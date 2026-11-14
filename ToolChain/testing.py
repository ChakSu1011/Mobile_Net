"""模块：Build 阶段测试辅助工具。

职责：
  构造小型 ``ModelGraph``、生成合成 safetensors 文件，供 converter、
  optimizer、serializer 与 pipeline 的单元测试复用。

边界：
  只用于测试，不进入生产流水线；不依赖 torch/transformers。
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from ToolChain.common.dtypes import element_bytes
from ToolChain.common.model_ir import (
    Dim,
    IOBinding,
    ModelGraph,
    OpMeta,
    TensorMeta,
)

S_HIDDEN = 16
S_INTERMEDIATE = 32


def make_tensor(
    name: str,
    kind: str,
    shape: tuple[int | str, ...],
    precision: str,
    *,
    reusable: bool = True,
    weight_source: str | None = None,
) -> TensorMeta:
    """构造张量元数据。

    参数：
      name：张量名；
      kind：input/constant/activation/output；
      shape：维度序列，元素为 int 或符号；
      precision：精度名；
      reusable：是否允许复用；
      weight_source：源权重名。

    返回：
      ``TensorMeta``；静态形状自动计算 byte_size。
    """
    dims = tuple(Dim(value) for value in shape)
    static_shape = (
        tuple(int(dim.value) for dim in dims)
        if all(not dim.is_symbol for dim in dims) else None
    )
    byte_size = 0
    if static_shape is not None and precision != "unknown":
        byte_size = (
            math.prod(static_shape) * element_bytes(precision))
    pitch = (
        static_shape
        if kind == "constant" and static_shape is not None else ()
    )
    return TensorMeta(
        name=name,
        kind=kind,
        shape=dims,
        precision=precision,
        layout="row_major",
        byte_size=byte_size,
        pitch=pitch,
        reusable=reusable,
        weight_source=weight_source,
    )


def build_tiny_model(
    precision: str = "bfloat16",
) -> ModelGraph:
    """构造 gather→linear 的最小可优化模型图。

    参数：
      precision：常量与激活精度。

    返回：
      通过 ``validate()`` 的 ``ModelGraph``。
    """
    tensors = {
        "input.input_ids": make_tensor(
            "input.input_ids", "input", ("B", "S"), "int64",
            reusable=False),
        "embedding.weight": make_tensor(
            "embedding.weight", "constant",
            (100, S_HIDDEN), precision,
            reusable=False,
            weight_source="model.embed_tokens.weight"),
        "embedding.out": make_tensor(
            "embedding.out", "activation",
            ("B", "S", S_HIDDEN), precision),
        "head.weight": make_tensor(
            "head.weight", "constant",
            (S_INTERMEDIATE, S_HIDDEN), precision,
            reusable=False,
            weight_source="lm_head.weight"),
        "head.out": make_tensor(
            "head.out", "activation",
            ("B", "S", S_INTERMEDIATE), precision,
            reusable=False),
    }
    ops = (
        OpMeta(
            name="embedding",
            op_type="gather",
            inputs=("input.input_ids", "embedding.weight"),
            outputs=("embedding.out",),
            attrs={
                "axis": 0,
                "precision": precision,
                "layout": "row_major",
            },
        ),
        OpMeta(
            name="head",
            op_type="linear",
            inputs=("embedding.out", "head.weight"),
            outputs=("head.out",),
            attrs={
                "in_features": S_HIDDEN,
                "out_features": S_INTERMEDIATE,
                "precision": precision,
                "layout": "row_major",
                "bias": False,
            },
        ),
    )
    return ModelGraph(
        name="tiny",
        adapter="test",
        inputs=("input.input_ids",),
        constants=("embedding.weight", "head.weight"),
        outputs=(IOBinding(
            name="output.logits", tensor="head.out"),),
        tensors=tensors,
        ops=ops,
        metadata={"model_config": {"tie_word_embeddings": False}},
    )


def build_attention_model(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int = 8,
    sequence: int = 4,
) -> ModelGraph:
    """构造仅含 mega_attention 的模型图，用于 GQA 校验。

    参数：
      num_heads：Query head 数；
      num_kv_heads：KV head 数；
      head_dim：单 head 宽度；
      sequence：序列长度。

    返回：
      ``ModelGraph``。
    """
    tensors = {
        "input.q": make_tensor(
            "input.q", "input",
            (1, sequence, num_heads * head_dim), "float16",
            reusable=False),
        "input.k": make_tensor(
            "input.k", "input",
            (1, sequence, num_kv_heads * head_dim), "float16",
            reusable=False),
        "input.v": make_tensor(
            "input.v", "input",
            (1, sequence, num_kv_heads * head_dim), "float16",
            reusable=False),
        "attention.out": make_tensor(
            "attention.out", "activation",
            (1, sequence, num_heads * head_dim), "float16",
            reusable=False),
    }
    op = OpMeta(
        name="attention",
        op_type="mega_attention",
        inputs=("input.q", "input.k", "input.v"),
        outputs=("attention.out",),
        attrs={
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "precision": "float16",
        },
    )
    return ModelGraph(
        name="attention",
        adapter="test",
        inputs=("input.q", "input.k", "input.v"),
        constants=(),
        outputs=(IOBinding(
            name="output.hidden", tensor="attention.out"),),
        tensors=tensors,
        ops=(op,),
        metadata={},
    )


def build_rms_norm_model(
    normalized_shape: int,
    last_dim: int,
) -> ModelGraph:
    """构造仅含 RMSNorm 的模型图，用于整除校验。

    参数：
      normalized_shape：归一化宽度；
      last_dim：输入末维。

    返回：
      ``ModelGraph``。
    """
    tensors = {
        "input.x": make_tensor(
            "input.x", "input", (1, last_dim), "float16",
            reusable=False),
        "norm.weight": make_tensor(
            "norm.weight", "constant", (normalized_shape,),
            "float16", reusable=False),
        "norm.out": make_tensor(
            "norm.out", "activation", (1, last_dim), "float16",
            reusable=False),
    }
    op = OpMeta(
        name="norm",
        op_type="rms_norm",
        inputs=("input.x", "norm.weight"),
        outputs=("norm.out",),
        attrs={
            "normalized_shape": normalized_shape,
            "precision": "float16",
        },
    )
    return ModelGraph(
        name="rms_norm",
        adapter="test",
        inputs=("input.x",),
        constants=("norm.weight",),
        outputs=(IOBinding(name="output.x", tensor="norm.out"),),
        tensors=tensors,
        ops=(op,),
        metadata={},
    )


def build_add_model(
    first_shape: tuple[int, ...],
    second_shape: tuple[int, ...],
) -> ModelGraph:
    """构造仅含逐元素加法的模型图。

    参数：
      first_shape：第一个输入形状；
      second_shape：第二个输入形状。

    返回：
      ``ModelGraph``。
    """
    tensors = {
        "input.a": make_tensor(
            "input.a", "input", first_shape, "float16",
            reusable=False),
        "input.b": make_tensor(
            "input.b", "input", second_shape, "float16",
            reusable=False),
        "add.out": make_tensor(
            "add.out", "activation", first_shape, "float16",
            reusable=False),
    }
    op = OpMeta(
        name="add",
        op_type="add",
        inputs=("input.a", "input.b"),
        outputs=("add.out",),
        attrs={"precision": "float16"},
    )
    return ModelGraph(
        name="add",
        adapter="test",
        inputs=("input.a", "input.b"),
        constants=(),
        outputs=(IOBinding(name="output.y", tensor="add.out"),),
        tensors=tensors,
        ops=(op,),
        metadata={},
    )


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    """写出合成 safetensors 文件。

    参数：
      path：输出文件路径；
      tensors：张量名到 ``(dtype, shape, payload)`` 的映射。

    返回：
      无。

    异常：
      OSError：写入失败。
    """
    header: dict[str, object] = {}
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        begin = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [begin, len(payload)],
        }
    header_bytes = json.dumps(
        header, ensure_ascii=False, sort_keys=True).encode("utf-8")
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding
    path.write_bytes(
        struct.pack("<Q", len(header_bytes))
        + header_bytes
        + bytes(payload)
    )
