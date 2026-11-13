"""模块：Qwen3-0.6B Runtime Graph 构建器。

职责：
  依据 Qwen3-0.6B 配置和 Transformers Qwen3 源码的层结构，使用
  Graph_Fixed 的 Python 绑定构造 runtime 静态图；图只描述 op、tensor、
  constant 和连接关系，不加载真实权重、不执行推理。

边界：
  - 不读取 safetensors 权重内容，constant 的 data_ptr 固定为 0；
  - 不实现 shape/precision 推导、显存规划、序列化和 kernel 执行；
  - 只支持 Qwen3 稠密模型的 prefill 图，不支持 MoE、VL、KV cache；
  - attention mask 由 kMegaAttention 的 causal 属性表达，不建模 padding。

依赖：
  ``mobilenet_runtime_graph``、``qwen3_config``、Python 3.10+ 标准库。

主要逻辑：
  1. 校验配置并创建 Graph；
  2. 按 Transformers 参数顺序登记全部 constant；
  3. 创建 input_ids/position_ids 输入和 embedding；
  4. 逐层创建 attention、MLP、residual 子图；
  5. 创建 final RMSNorm、LM Head 和 output，最后执行 sort_op()。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

from qwen3_config import Qwen3Config


def _load_graph_api() -> Any:
    """加载 Graph_Fixed 的 Python 公共包。

    优先使用当前 ``sys.path``；失败时把仓库的 ``Graph_Fixed/python``
    加入 ``sys.path`` 后重试。

    返回：
      ``mobilenet_runtime_graph`` 模块对象。

    异常：
      ModuleNotFoundError：公共包或 ``_graph_fixed`` 扩展不可用时抛出。
    """
    try:
        import mobilenet_runtime_graph as graph_api
        return graph_api
    except ModuleNotFoundError:
        repository_root = Path(__file__).resolve().parents[2]
        sys.path.insert(
            0, str(repository_root / "Graph_Fixed" / "python"))
        import mobilenet_runtime_graph as graph_api
        return graph_api


_GRAPH_API = _load_graph_api()
Graph = _GRAPH_API.Graph
Tensor = _GRAPH_API.Tensor
Precision = _GRAPH_API.Precision
Layout = _GRAPH_API.Layout
OpType = _GRAPH_API.OpType

GRAPH_NAME = "qwen3_0_6b"
_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
_DTYPE_TO_PRECISION = {
    "bfloat16": Precision.kBFloat16,
    "float16": Precision.kFloat16,
    "float32": Precision.kFloat32,
}
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


class Qwen3GraphBuilder:
    """按 Qwen3 层结构构造 Graph_Fixed 静态图。

    功能：
      从 Qwen3Config 生成完整 prefill 图：embedding、N 个 Decoder Layer、
      final RMSNorm 和 LM Head，返回已排序的 Graph。

    边界：
      - 权重只登记元信息，data_ptr=0；
      - 图输入只有 input_ids 和 position_ids；
      - LM Head 输出全序列 logits，不在图内做 last-token gather。

    主要逻辑：
      1. 创建全部 constant 和图输入；
      2. 逐层创建 attention/MLP 子图并连接 residual；
      3. 创建 final norm、lm_head 和 output；
      4. 调用 sort_op()，保证执行序列从输入到输出连续。
    """

    def __init__(
        self,
        config: Qwen3Config,
        graph_factory: Callable[[str], Any] | None = None,
    ) -> None:
        """保存配置并校验。

        参数：
          config：Qwen3 配置；必须通过 validate()；
          graph_factory：可选图工厂，接收图名并返回 Graph 兼容对象；
            为 None 时直接创建 Graph_Fixed 的 Graph。Build 阶段用它
            注入 RecordingGraph，不改变默认行为。

        异常：
          ValueError：配置超出当前实现范围。
        """
        config.validate()
        self._config = config
        self._graph_factory = graph_factory
        self._precision = _DTYPE_TO_PRECISION[config.dtype]
        self._element_bytes = _DTYPE_BYTES[config.dtype]
        self._graph: Graph | None = None
        self._constants: dict[str, Tensor] = {}
        self._weight_shapes: dict[str, tuple[int, ...]] = {}
        self._constant_shapes: dict[str, tuple[int, ...]] = {}

    @property
    def weight_shapes(self) -> dict[str, tuple[int, ...]]:
        """返回逻辑权重名到形状的映射，包含 tied lm_head 别名。"""
        return dict(self._weight_shapes)

    @property
    def graph_constant_shapes(self) -> dict[str, tuple[int, ...]]:
        """返回图中实际登记的 constant 形状，不含 tied 别名。"""
        return dict(self._constant_shapes)

    def build(self) -> Graph:
        """构建并返回 Qwen3 静态图。

        返回：
          已调用 ``sort_op()`` 的 Graph 对象；Tensor 句柄由 Graph 持有。

        异常：
          RuntimeError：Graph_Fixed 命名冲突、tensor 归属错误或排序失败。
        """
        self._graph = (
            Graph(GRAPH_NAME)
            if self._graph_factory is None
            else self._graph_factory(GRAPH_NAME)
        )
        self._constants = {}
        self._weight_shapes = {}
        self._constant_shapes = {}

        self._add_constants()
        input_ids, position_ids = self._add_inputs()
        hidden = self._add_embedding(input_ids)
        for layer_index in range(self._config.num_hidden_layers):
            hidden = self._add_decoder_layer(
                layer_index, hidden, position_ids)
        hidden = self._add_final_norm(hidden)
        logits = self._add_lm_head(hidden)
        self._graph.create_output("output.logits", logits)
        self._graph.sort_op()
        return self._graph

    def _add_constants(self) -> None:
        """按 Transformers 参数顺序登记全部权重常量。

        返回：
          无；常量写入 ``self._constants`` 和形状映射。

        主要逻辑：
          embedding 在最前，随后逐层登记 11 个权重，最后登记 final norm；
          tie_word_embeddings 为 true 时只登记一个 embedding constant。
        """
        config = self._config
        self._create_constant(
            "embedding.weight",
            (config.vocab_size, config.hidden_size),
        )
        for layer_index in range(config.num_hidden_layers):
            prefix = f"layer.{layer_index}"
            self._create_constant(
                f"{prefix}.attn_norm.weight",
                (config.hidden_size,),
            )
            self._create_constant(
                f"{prefix}.q_proj.weight",
                (config.q_width, config.hidden_size),
            )
            self._create_constant(
                f"{prefix}.k_proj.weight",
                (config.kv_width, config.hidden_size),
            )
            self._create_constant(
                f"{prefix}.v_proj.weight",
                (config.kv_width, config.hidden_size),
            )
            self._create_constant(
                f"{prefix}.o_proj.weight",
                (config.hidden_size, config.q_width),
            )
            self._create_constant(
                f"{prefix}.q_norm.weight",
                (config.head_dim,),
            )
            self._create_constant(
                f"{prefix}.k_norm.weight",
                (config.head_dim,),
            )
            self._create_constant(
                f"{prefix}.mlp_norm.weight",
                (config.hidden_size,),
            )
            self._create_constant(
                f"{prefix}.gate_proj.weight",
                (config.intermediate_size, config.hidden_size),
            )
            self._create_constant(
                f"{prefix}.up_proj.weight",
                (config.intermediate_size, config.hidden_size),
            )
            self._create_constant(
                f"{prefix}.down_proj.weight",
                (config.hidden_size, config.intermediate_size),
            )
        self._create_constant(
            "final_norm.weight",
            (config.hidden_size,),
        )
        if config.tie_word_embeddings:
            self._weight_shapes["lm_head.weight"] = (
                self._weight_shapes["embedding.weight"])
        else:
            self._create_constant(
                "lm_head.weight",
                (config.vocab_size, config.hidden_size),
            )

    def _add_inputs(self) -> tuple[Tensor, Tensor]:
        """创建 input_ids 和 position_ids 两个图输入。

        返回：
          ``(input_ids, position_ids)``，均为 Int64、RowMajor 的 Tensor。
        """
        input_ids = self._graph.create_input(
            "input.input_ids", Precision.kInt64, Layout.kRowMajor)
        position_ids = self._graph.create_input(
            "input.position_ids", Precision.kInt64, Layout.kRowMajor)
        return input_ids, position_ids

    def _add_embedding(self, input_ids: Tensor) -> Tensor:
        """创建 token embedding 节点。

        参数：
          input_ids：形状为 [batch, sequence] 的整数输入。

        返回：
          形状为 [batch, sequence, hidden_size] 的 hidden tensor。
        """
        return self._create_op(
            "embedding",
            [input_ids, self._constants["embedding.weight"]],
            OpType.kGather,
            {
                "axis": 0,
                "vocab_size": self._config.vocab_size,
                "hidden_size": self._config.hidden_size,
                "precision": self._config.dtype,
            },
        )

    def _add_decoder_layer(
        self,
        layer_index: int,
        hidden: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """创建一个完整的 Qwen3 Decoder Layer。

        参数：
          layer_index：层编号，用于生成名称和查找权重；
          hidden：层输入 [batch, sequence, hidden_size]；
          position_ids：RoPE 位置输入 [batch, sequence]。

        返回：
          层输出 [batch, sequence, hidden_size]。

        主要逻辑：
          严格按 Transformers Qwen3DecoderLayer.forward 的顺序创建
          attention norm、QKV、q/k norm、RoPE、attention、o_proj、
          residual、MLP norm、SwiGLU、down_proj 和 residual。
        """
        config = self._config
        prefix = f"layer.{layer_index}"

        attn_norm = self._rms_norm(
            f"{prefix}.attn_norm",
            hidden,
            self._constants[f"{prefix}.attn_norm.weight"],
            config.hidden_size,
        )
        q_proj = self._linear(
            f"{prefix}.q_proj",
            attn_norm,
            self._constants[f"{prefix}.q_proj.weight"],
            config.hidden_size,
            config.q_width,
        )
        k_proj = self._linear(
            f"{prefix}.k_proj",
            attn_norm,
            self._constants[f"{prefix}.k_proj.weight"],
            config.hidden_size,
            config.kv_width,
        )
        v_proj = self._linear(
            f"{prefix}.v_proj",
            attn_norm,
            self._constants[f"{prefix}.v_proj.weight"],
            config.hidden_size,
            config.kv_width,
        )
        q_norm = self._rms_norm(
            f"{prefix}.q_norm",
            q_proj,
            self._constants[f"{prefix}.q_norm.weight"],
            config.head_dim,
            num_heads=config.num_attention_heads,
        )
        k_norm = self._rms_norm(
            f"{prefix}.k_norm",
            k_proj,
            self._constants[f"{prefix}.k_norm.weight"],
            config.head_dim,
            num_kv_heads=config.num_key_value_heads,
        )
        q_rope = self._rope(
            f"{prefix}.q_rope",
            q_norm,
            position_ids,
            config.num_attention_heads,
        )
        k_rope = self._rope(
            f"{prefix}.k_rope",
            k_norm,
            position_ids,
            config.num_key_value_heads,
        )
        attn = self._attention(
            f"{prefix}.attn", q_rope, k_rope, v_proj)
        attn_out = self._linear(
            f"{prefix}.o_proj",
            attn,
            self._constants[f"{prefix}.o_proj.weight"],
            config.q_width,
            config.hidden_size,
        )
        hidden_mid = self._add(
            f"{prefix}.attn_residual", attn_out, hidden)

        mlp_norm = self._rms_norm(
            f"{prefix}.mlp_norm",
            hidden_mid,
            self._constants[f"{prefix}.mlp_norm.weight"],
            config.hidden_size,
        )
        mlp_act = self._linear_swiglu(
            f"{prefix}.mlp_gate_up",
            mlp_norm,
            self._constants[f"{prefix}.gate_proj.weight"],
            self._constants[f"{prefix}.up_proj.weight"],
        )
        mlp_down = self._linear(
            f"{prefix}.mlp_down",
            mlp_act,
            self._constants[f"{prefix}.down_proj.weight"],
            config.intermediate_size,
            config.hidden_size,
        )
        return self._add(
            f"{prefix}.mlp_residual", mlp_down, hidden_mid)

    def _add_final_norm(self, hidden: Tensor) -> Tensor:
        """创建 final RMSNorm 节点。

        参数：
          hidden：最后一层输出 [batch, sequence, hidden_size]。

        返回：
          final_norm 输出 [batch, sequence, hidden_size]。
        """
        return self._rms_norm(
            "final_norm",
            hidden,
            self._constants["final_norm.weight"],
            self._config.hidden_size,
        )

    def _add_lm_head(self, hidden: Tensor) -> Tensor:
        """创建 LM Head 线性投影。

        参数：
          hidden：final_norm 输出 [batch, sequence, hidden_size]。

        返回：
          logits [batch, sequence, vocab_size]。

        主要逻辑：
          tie_word_embeddings=true 时直接复用 embedding.weight 的 Tensor；
          否则使用单独登记的 lm_head.weight。
        """
        weight_name = (
            "embedding.weight"
            if self._config.tie_word_embeddings
            else "lm_head.weight"
        )
        return self._create_op(
            "lm_head",
            [hidden, self._constants[weight_name]],
            OpType.kLinear,
            {
                "in_features": self._config.hidden_size,
                "out_features": self._config.vocab_size,
                "transpose_weight": True,
                "bias": False,
                "tie_word_embeddings":
                    self._config.tie_word_embeddings,
                "logits_to_keep": 0,
                "precision": self._config.dtype,
            },
        )

    def _create_constant(
        self,
        name: str,
        dims: tuple[int, ...],
    ) -> Tensor:
        """登记一个只含元信息的 constant tensor。

        参数：
          name：权重名，必须全局唯一；
          dims：逻辑形状，采用 Transformers 权重布局。

        返回：
          constant Tensor；其 ``reusable`` 被 Graph_Fixed 强制为 false。

        主要逻辑：
          pitch 与 dims 相同，layout 固定 RowMajor，data_ptr 固定 0，
          byte_size 按 numel * dtype 字节数计算。
        """
        byte_size = math.prod(dims) * self._element_bytes
        tensor = self._graph.create_constant(
            name,
            dims=list(dims),
            pitch=list(dims),
            precision=self._precision,
            layout=Layout.kRowMajor,
            byte_size=byte_size,
            data_ptr=0,
        )
        self._constants[name] = tensor
        self._weight_shapes[name] = dims
        self._constant_shapes[name] = dims
        return tensor

    def _create_op(
        self,
        name: str,
        src_tensors: list[Tensor],
        op_type: Any,
        attrs: dict[str, Any],
    ) -> Tensor:
        """创建一个单输出计算 op。

        参数：
          name：op 名称；
          src_tensors：输入 tensor 列表；
          op_type：Graph_Fixed OpType；
          attrs：算子属性字典。

        返回：
          op 的输出 tensor，名称为 ``<name>.out``。
        """
        outputs = self._graph.create_op(
            name, src_tensors, 1, op_type, attrs)
        return outputs[0]

    def _rms_norm(
        self,
        name: str,
        input_tensor: Tensor,
        weight: Tensor,
        normalized_size: int,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
    ) -> Tensor:
        """创建一个 RMSNorm 节点。

        参数：
          name：op 名称；
          input_tensor：待归一化张量；
          weight：RMSNorm 权重；
          normalized_size：归一化维度大小；
          num_heads：Q head 数，q_norm 使用；
          num_kv_heads：KV head 数，k_norm 使用。

        返回：
          归一化后的 tensor。

        主要逻辑：
          normalized_size=head_dim 时表示 per-head RMSNorm；否则表示
          对整个 hidden_size 做 RMSNorm。
        """
        attrs: dict[str, Any] = {
            "eps": self._config.rms_norm_eps,
            "normalized_shape": normalized_size,
            "axis": -1,
            "precision": self._config.dtype,
        }
        if num_heads is not None:
            attrs["num_heads"] = num_heads
        if num_kv_heads is not None:
            attrs["num_kv_heads"] = num_kv_heads
        return self._create_op(
            name, [input_tensor, weight], OpType.kRmsNorm, attrs)

    def _linear(
        self,
        name: str,
        input_tensor: Tensor,
        weight: Tensor,
        in_features: int,
        out_features: int,
    ) -> Tensor:
        """创建一个 Linear 节点。

        参数：
          name：op 名称；
          input_tensor：输入 [..., in_features]；
          weight：权重 [out_features, in_features]；
          in_features：输入宽度；
          out_features：输出宽度。

        返回：
          输出 [..., out_features]。

        主要逻辑：
          ``transpose_weight=True`` 表示权重布局为 [out, in]，计算
          ``x @ weight.T``，与 Transformers nn.Linear 一致。
        """
        return self._create_op(
            name,
            [input_tensor, weight],
            OpType.kLinear,
            {
                "in_features": in_features,
                "out_features": out_features,
                "transpose_weight": True,
                "bias": False,
                "precision": self._config.dtype,
            },
        )

    def _rope(
        self,
        name: str,
        input_tensor: Tensor,
        position_ids: Tensor,
        num_heads: int,
    ) -> Tensor:
        """创建一个 NeoX RoPE 节点。

        参数：
          name：op 名称；
          input_tensor：q 或 k 投影结果；
          position_ids：位置输入 [batch, sequence]；
          num_heads：该 tensor 的 head 数。

        返回：
          旋转后的 q 或 k tensor。

        主要逻辑：
          kRope 融合 Qwen3RotaryEmbedding.forward 和
          apply_rotary_pos_emb；rope_layout=neox 对应 rotate_half。
        """
        return self._create_op(
            name,
            [input_tensor, position_ids],
            OpType.kRope,
            {
                "head_dim": self._config.head_dim,
                "num_heads": num_heads,
                "rope_theta": self._config.rope_theta,
                "rope_type": self._config.rope_type,
                "rope_layout": "neox",
                "scaling_factor": 1.0,
                "original_max_position_embeddings":
                    self._config.max_position_embeddings,
                "precision": self._config.dtype,
            },
        )

    def _attention(
        self,
        name: str,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """创建一个 causal GQA attention 节点。

        参数：
          name：op 名称；
          query：旋转后的 Q [batch, sequence, num_heads*head_dim]；
          key：旋转后的 K [batch, sequence, num_kv_heads*head_dim]；
          value：V [batch, sequence, num_kv_heads*head_dim]。

        返回：
          attention context [batch, sequence, num_heads*head_dim]。

        主要逻辑：
          kMegaAttention 覆盖 repeat_kv、QK^T、scale、causal mask、
          softmax 和 PV；has_output_projection=false 表示 o_proj 由
          独立 kLinear 节点完成；merge_heads=true 表示输出已合并 head。
        """
        config = self._config
        return self._create_op(
            name,
            [query, key, value],
            OpType.kMegaAttention,
            {
                "num_heads": config.num_attention_heads,
                "num_kv_heads": config.num_key_value_heads,
                "gqa_ratio": config.gqa_ratio,
                "head_dim": config.head_dim,
                "causal": True,
                "softmax_scale": 1.0 / math.sqrt(config.head_dim),
                "sliding_window": 0,
                "has_output_projection": False,
                "merge_heads": True,
                "use_kv_cache": False,
                "precision": config.dtype,
            },
        )

    def _add(
        self,
        name: str,
        left: Tensor,
        right: Tensor,
    ) -> Tensor:
        """创建一个逐元素加法节点。

        参数：
          name：op 名称；
          left：第一个加数；
          right：第二个加数，residual 路径通常传原始 hidden。

        返回：
          逐元素和。
        """
        return self._create_op(
            name,
            [left, right],
            OpType.kAdd,
            {"precision": self._config.dtype},
        )

    def _linear_swiglu(
        self,
        name: str,
        input_tensor: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
    ) -> Tensor:
        """创建一个 SwiGLU 前段融合节点。

        参数：
          name：op 名称；
          input_tensor：MLP norm 输出 [..., hidden_size]；
          gate_weight：gate_proj 权重 [intermediate, hidden]；
          up_weight：up_proj 权重 [intermediate, hidden]。

        返回：
          ``silu(x @ gate.T) * (x @ up.T)``，形状
          [..., intermediate_size]。

        主要逻辑：
          kLinearSwiGlu 融合 gate_proj、up_proj、SiLU 和逐元素乘，
          与 Qwen3MLP.forward 的前半段一致。
        """
        return self._create_op(
            name,
            [input_tensor, gate_weight, up_weight],
            OpType.kLinearSwiGlu,
            {
                "hidden_size": self._config.hidden_size,
                "intermediate_size":
                    self._config.intermediate_size,
                "activation": "silu",
                "transpose_weight": True,
                "bias": False,
                "precision": self._config.dtype,
            },
        )


def build_qwen3_graph(config: Qwen3Config | None = None) -> Graph:
    """构建 Qwen3-0.6B 静态图。

    参数：
      config：可选配置；为 None 时使用 Qwen3-0.6B 默认值。

    返回：
      已调用 ``sort_op()`` 的 Graph。

    异常：
      ValueError：配置非法；
      RuntimeError：Graph_Fixed 构图或排序失败。
    """
    if config is None:
        config = Qwen3Config()
    return Qwen3GraphBuilder(config).build()


def graph_summary(graph: Graph) -> str:
    """返回图的名称和四段计数。

    参数：
      graph：已排序的 Graph。

    返回：
      多行文本，包含 graph 名称、inputs、constants、ops、outputs 计数。
    """
    lines = graph.dump_ir().splitlines()
    selected = [lines[0]]
    for line in lines[1:]:
        if line.startswith(
                ("inputs ", "constants ", "ops ", "outputs ")):
            selected.append(line)
    return "\n".join(selected)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：构建 Qwen3 图并输出摘要或完整 IR。

    参数：
      argv：命令行参数列表；None 表示使用 ``sys.argv[1:]``。

    返回：
      进程退出码，正常为 0。

    异常：
      ValueError：配置非法；
      RuntimeError：Graph_Fixed 构图失败；
      OSError：配置文件或输出文件不可读写。
    """
    parser = argparse.ArgumentParser(
        description="Build the Qwen3-0.6B Graph_Fixed static graph")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Qwen3 config.json path",
    )
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="print the full IR text",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the full IR text to a file",
    )
    args = parser.parse_args(argv)

    config = Qwen3Config.from_json(args.config)
    graph = build_qwen3_graph(config)
    if args.output is not None:
        args.output.write_text(graph.dump_ir(), encoding="utf-8")
        print(f"IR written to {args.output}")
    print(graph_summary(graph))
    if args.dump_ir:
        print(graph.dump_ir())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
