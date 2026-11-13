"""模块：Qwen3 模型 Adapter。

职责：
  加载 Qwen3 配置，复用 ``Models/Qwen3/qwen3_graph_builder.py`` 的层结构，
  通过 ``RecordingGraph`` 构建 Graph_Fixed 图并补充权重来源、tied 别名与
  模型配置元数据。

边界：
  - 不读取或修改权重数值，只读取 safetensors 头信息；
  - 不实现 Qwen3 之外的模型结构；
  - 不执行 shape/precision 推导与序列化。
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from Build.common.errors import BuildError
from Build.common.graph_api import load_graph_api
from Build.common.model_ir import Dim, ModelGraph
from Build.common.safetensors import SafetensorsEntry, read_index
from Build.converter.recording_graph import RecordingGraph
from Build.converter.result import ConversionResult

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPOSITORY_ROOT / "Models" / "Qwen3"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from qwen3_config import Qwen3Config  # noqa: E402
from qwen3_graph_builder import (  # noqa: E402
    GRAPH_NAME,
    Qwen3GraphBuilder,
)

_LAYER_WEIGHT_MAP = {
    "attn_norm": "input_layernorm",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "q_norm": "self_attn.q_norm",
    "k_norm": "self_attn.k_norm",
    "mlp_norm": "post_attention_layernorm",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def huggingface_weight_name(graph_name: str) -> str | None:
    """把图内常量名转换为 Transformers 参数名。

    参数：
      graph_name：图内常量名，例如 ``layer.3.q_proj.weight``。

    返回：
      对应的 Transformers 参数名；无法映射时返回 None。
    """
    if graph_name == "embedding.weight":
        return "model.embed_tokens.weight"
    if graph_name == "final_norm.weight":
        return "model.norm.weight"
    if graph_name == "lm_head.weight":
        return "lm_head.weight"
    if not graph_name.startswith("layer."):
        return None
    parts = graph_name.split(".")
    if len(parts) != 4 or parts[3] != "weight":
        return None
    try:
        layer_index = int(parts[1])
    except ValueError:
        return None
    suffix = _LAYER_WEIGHT_MAP.get(parts[2])
    if suffix is None:
        return None
    return f"model.layers.{layer_index}.{suffix}.weight"


class Qwen3Adapter:
    """Qwen3-0.6B 转换 Adapter。

    功能：
      读取 config.json，构建 Graph_Fixed 图，记录 ModelGraph，并按可选的
      safetensors 权重目录登记权重可用性。

    边界：
      只支持 Dense Qwen3 prefill 图；不支持 MoE、VL 与 KV Cache。
    """

    name = "qwen3"

    def __init__(
        self,
        config_path: Path,
        weights_path: Path | None = None,
    ) -> None:
        """保存配置与权重路径。

        参数：
          config_path：Transformers 风格 Qwen3 config.json；
          weights_path：可选 safetensors 文件或目录。
        """
        self._config_path = Path(config_path)
        self._weights_path = (
            None if weights_path is None else Path(weights_path))

    def build(self) -> ConversionResult:
        """构建并返回转换结果。

        返回：
          ``ConversionResult``；图已排序，ModelGraph 已自校验。

        异常：
          ValueError：配置非法；
          BuildError：常量缺少权重映射、权重文件损坏或记录不一致；
          RuntimeError：Graph_Fixed 构图失败。
        """
        config = Qwen3Config.from_json(self._config_path)
        config.validate()
        graph_api = load_graph_api()
        recording = RecordingGraph(
            graph_api.Graph(GRAPH_NAME),
            name=GRAPH_NAME,
            adapter=self.name,
        )

        def graph_factory(name: str) -> RecordingGraph:
            """返回记录代理，并校验 Builder 请求的图名。"""
            if name != recording.name:
                raise BuildError(
                    "converter",
                    f"unexpected graph name: {name}",
                    target=GRAPH_NAME,
                )
            return recording

        builder = Qwen3GraphBuilder(
            config, graph_factory=graph_factory)
        builder.build()
        model_graph = recording.finish()
        weight_index = self._read_weight_index()
        self._annotate(
            model_graph, config, builder, weight_index)
        return ConversionResult(
            graph=recording.adapted_graph,
            model_graph=model_graph,
            adapter=self.name,
        )

    def _read_weight_index(
        self,
    ) -> dict[str, SafetensorsEntry] | None:
        """读取可选权重索引。

        返回：
          张量名到条目的映射；未提供权重或路径不存在时返回 None。

        异常：
          BuildError：权重文件存在但损坏。
        """
        if self._weights_path is None:
            return None
        if not self._weights_path.exists():
            return None
        return read_index(self._weights_path)

    def _annotate(
        self,
        model_graph: ModelGraph,
        config: Qwen3Config,
        builder: Qwen3GraphBuilder,
        weight_index: dict[str, SafetensorsEntry] | None,
    ) -> None:
        """补充输入形状、权重映射与模型配置元数据。

        参数：
          model_graph：待补充的模型图；
          config：Qwen3 配置；
          builder：已完成构图的 Builder；
          weight_index：可选权重索引。

        异常：
          BuildError：常量缺少 HF 映射。
        """
        for name in model_graph.inputs:
            tensor = model_graph.tensor(name)
            tensor.shape = (Dim("B"), Dim("S"))
            tensor.precision = "int64"
            tensor.layout = "row_major"
            tensor.pitch = ()

        issues: list[str] = []
        for name in model_graph.constants:
            tensor = model_graph.tensor(name)
            source = huggingface_weight_name(name)
            if source is None:
                raise BuildError(
                    "converter",
                    "no Hugging Face weight mapping for constant",
                    target=name,
                )
            tensor.weight_source = source
            tensor.weight_available = False
            if weight_index is None or source not in weight_index:
                continue
            entry = weight_index[source]
            static_shape = tensor.static_shape
            if (entry.shape != static_shape
                    or entry.precision != tensor.precision):
                expected = (
                    f"{tensor.precision}{list(static_shape or ())}")
                actual = f"{entry.precision}{list(entry.shape)}"
                issues.append(
                    f"{name}: expected {expected}, got {actual}")
                continue
            tensor.weight_available = True

        tied_weights: dict[str, str] = {}
        if config.tie_word_embeddings:
            tied_weights["lm_head.weight"] = "embedding.weight"
        model_graph.metadata.update({
            "architecture": "Qwen3ForCausalLM",
            "model_config": asdict(config),
            "tied_weights": tied_weights,
            "weight_shapes": {
                name: list(shape)
                for name, shape in builder.weight_shapes.items()
            },
            "weights_supplied": weight_index is not None,
            "weight_issues": issues,
        })
