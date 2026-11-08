"""模块：Qwen3-0.6B Python 侧参考图构建器。

职责：
  使用 ``torch.Tensor`` / ``torch.dtype`` 与 Python 绑定（``_graph``）构造
  Qwen3-0.6B 的融合算子图，作为 C++ 侧 ``BuildQwen3ReferenceGraph`` 的
  逐字节一致性基准。

边界：
  - 权重与激活张量使用 meta device，只保留形状/dtype 元数据；
  - I/O host_tensor 使用小型 CPU 张量，不加载真实权重；
  - 权重、节点、边的插入顺序与属性必须与 C++ 夹具完全一致，任何顺序变化
    都会改变 Canonical JSON，必须同步修改 ``qwen3_fixture.cc``。

依赖：
  ``mobilenet_graph.graph_tools``、``mobilenet_graph`` 导出的图类型、
  ``torch``。
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .. import GraphBuilder, ModelConfig, Node, OpType, RopeLayout
from .. import graph_tools as tools

# Qwen3-0.6B 超参数；来源：models/Qwen3-0.6B/config.json。
NUM_LAYERS = 28
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_ATTENTION_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 151936
MAX_CONTEXT_LENGTH = 40960
NORM_EPS = 1e-6
ROPE_THETA = 1000000.0
BATCH_EXAMPLE = 1
SEQUENCE_EXAMPLE = 8


def _meta(
    shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """创建 meta 张量，只保留形状与 dtype。"""
    return torch.empty(shape, dtype=dtype, device="meta")


def _cpu(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """创建 CPU 小张量，用于 I/O host_tensor。"""
    return torch.empty(shape, dtype=dtype, device="cpu")


def _layer_prefix(layer: int) -> str:
    """返回层内名称前缀。"""
    return f"layer.{layer}"


def _layer_source(layer: int, suffix: str) -> str:
    """返回 safetensors 参数路径。"""
    return f"model.layers.{layer}.{suffix}"


def build_qwen3_python_graph() -> Any:
    """构造 Qwen3-0.6B Python 参考图。

    返回：
      ``_graph.Graph``；已通过 Finalize 校验。
    异常：
      ``_graph.GraphError``：命名冲突、边缺失、校验失败等。
    """
    builder = GraphBuilder()
    builder.set_model_config(_make_model_config())
    weights = _add_weights(builder)
    input_ids, output_ids = _add_nodes(builder)
    _add_edges(builder, weights)
    builder.set_input_nodes(input_ids)
    builder.set_output_nodes(output_ids)
    return builder.finalize()


def _make_model_config() -> Any:
    """构造 Qwen3 ModelConfig，数值与 config.json 一致。"""
    config = ModelConfig()
    config.model_family = "qwen3"
    config.num_layers = NUM_LAYERS
    config.hidden_size = HIDDEN_SIZE
    config.intermediate_size = INTERMEDIATE_SIZE
    config.num_attention_heads = NUM_ATTENTION_HEADS
    config.num_kv_heads = NUM_KV_HEADS
    config.head_dim = HEAD_DIM
    config.vocab_size = VOCAB_SIZE
    config.max_context_length = MAX_CONTEXT_LENGTH
    config.norm_eps = NORM_EPS
    config.activation = "silu"
    config.tie_word_embeddings = True
    config.dtype = torch.bfloat16
    config.rope.theta = ROPE_THETA
    config.rope.layout = RopeLayout.kNeox
    config.rope.scaling_type = "none"
    config.rope.scaling_factor = 1.0
    config.rope.original_max_position_embeddings = MAX_CONTEXT_LENGTH
    return config


def _add_weights(builder: Any) -> dict[str, torch.Tensor]:
    """按固定顺序添加全部权重，返回引用名到张量的映射。"""
    weights: dict[str, torch.Tensor] = {}
    weights["embedding.weight"] = _meta((VOCAB_SIZE, HIDDEN_SIZE))
    builder.add_weight(
        tools.weight_desc(
            "embedding.weight",
            "model.embed_tokens.weight",
            weights["embedding.weight"],
        )
    )
    for layer in range(NUM_LAYERS):
        prefix = _layer_prefix(layer)
        definitions = (
            ("attn_norm.weight", "input_layernorm.weight", (HIDDEN_SIZE,)),
            (
                "q.weight",
                "self_attn.q_proj.weight",
                (NUM_ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE),
            ),
            (
                "k.weight",
                "self_attn.k_proj.weight",
                (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
            ),
            (
                "v.weight",
                "self_attn.v_proj.weight",
                (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
            ),
            (
                "o.weight",
                "self_attn.o_proj.weight",
                (HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM),
            ),
            ("q_norm.weight", "self_attn.q_norm.weight", (HEAD_DIM,)),
            ("k_norm.weight", "self_attn.k_norm.weight", (HEAD_DIM,)),
            (
                "mlp_norm.weight",
                "post_attention_layernorm.weight",
                (HIDDEN_SIZE,),
            ),
            (
                "gate.weight",
                "mlp.gate_proj.weight",
                (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            ),
            (
                "up.weight",
                "mlp.up_proj.weight",
                (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            ),
            (
                "down.weight",
                "mlp.down_proj.weight",
                (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            ),
        )
        for suffix, source_suffix, shape in definitions:
            ref = prefix + "." + suffix
            tensor = _meta(shape)
            weights[ref] = tensor
            builder.add_weight(
                tools.weight_desc(
                    ref, _layer_source(layer, source_suffix), tensor
                )
            )
    weights["final_norm.weight"] = _meta((HIDDEN_SIZE,))
    builder.add_weight(
        tools.weight_desc(
            "final_norm.weight",
            "model.norm.weight",
            weights["final_norm.weight"],
        )
    )
    # tie_word_embeddings=true：LM Head 与 embedding 共享同一张量。
    weights["lm_head.weight"] = weights["embedding.weight"]
    builder.add_weight(
        tools.weight_desc(
            "lm_head.weight", "lm_head.weight", weights["lm_head.weight"]
        )
    )
    return weights


def _add_nodes(builder: Any) -> tuple[list[int], list[int]]:
    """按固定顺序添加全部节点，返回输入/输出节点 ID。"""
    input_ids = [
        builder.add_node(
            tools.input_node(
                "input.input_ids",
                "input_ids",
                "input_ids",
                _cpu((BATCH_EXAMPLE, SEQUENCE_EXAMPLE), torch.int64),
            )
        ),
        builder.add_node(
            tools.input_node(
                "input.position_ids",
                "position_ids",
                "position_ids",
                _cpu((BATCH_EXAMPLE, SEQUENCE_EXAMPLE), torch.int64),
            )
        ),
        builder.add_node(
            tools.input_node(
                "input.last_token_indices",
                "last_token_indices",
                "last_token_indices",
                _cpu((BATCH_EXAMPLE,), torch.int64),
            )
        ),
    ]

    builder.add_node(
        Node(
            name="embedding",
            op_type=OpType.kEmbedding,
            fused_ops=[
                tools.fused_op("model.embed_tokens", "embedding_lookup")
            ],
            input_edges=["input_ids", "weight.embedding"],
            output_edges=["hidden"],
            precision=torch.bfloat16,
            kernel_key="embedding",
        )
    )

    for layer in range(NUM_LAYERS):
        prefix = _layer_prefix(layer)
        hidden_in = "hidden" if layer == 0 else f"layer.{layer - 1}.hidden_out"
        norm_attributes = {"eps": NORM_EPS, "precision": "bfloat16"}
        builder.add_node(
            Node(
                name=prefix + ".attn_norm",
                op_type=OpType.kRmsNorm,
                fused_ops=[
                    tools.fused_op(
                        _layer_source(layer, "input_layernorm"), "rmsnorm"
                    )
                ],
                attributes=norm_attributes,
                input_edges=[hidden_in, "weight." + prefix + ".attn_norm"],
                output_edges=[prefix + ".attn_normed"],
                precision=torch.bfloat16,
                kernel_key="rms_norm",
                fusion_rule_id="qwen3.block.norm.attn.v1",
            )
        )
        builder.add_node(
            Node(
                name=prefix + ".qkv",
                op_type=OpType.kQkvProjection,
                fused_ops=[
                    tools.fused_op(
                        _layer_source(layer, "self_attn.q_proj"), "q_linear"
                    ),
                    tools.fused_op(
                        _layer_source(layer, "self_attn.k_proj"), "k_linear"
                    ),
                    tools.fused_op(
                        _layer_source(layer, "self_attn.v_proj"), "v_linear"
                    ),
                    tools.fused_op(
                        _layer_source(layer, "self_attn.q_norm"), "qk_rmsnorm"
                    ),
                    tools.fused_op(
                        _layer_source(layer, "self_attn.rotary_emb"), "rope"
                    ),
                ],
                attributes={
                    "num_heads": NUM_ATTENTION_HEADS,
                    "num_kv_heads": NUM_KV_HEADS,
                    "head_dim": HEAD_DIM,
                    "rope_theta": ROPE_THETA,
                    "rope_layout": "neox",
                    "has_qk_norm": True,
                    "qk_norm_eps": NORM_EPS,
                    "bias": False,
                },
                input_edges=[
                    prefix + ".attn_normed",
                    "weight." + prefix + ".q",
                    "weight." + prefix + ".k",
                    "weight." + prefix + ".v",
                    "weight." + prefix + ".q_norm",
                    "weight." + prefix + ".k_norm",
                    "position_ids",
                ],
                output_edges=[prefix + ".q", prefix + ".k", prefix + ".v"],
                precision=torch.bfloat16,
                kernel_key="qkv_projection",
                fusion_rule_id="qwen3.block.qkv.v1",
            )
        )
        builder.add_node(
            Node(
                name=prefix + ".attn",
                op_type=OpType.kAttention,
                fused_ops=[
                    tools.fused_op("kv_cache", "kv_cache_write"),
                    tools.fused_op(
                        _layer_source(layer, "self_attn"), "qk_matmul"
                    ),
                    tools.fused_op(
                        _layer_source(layer, "self_attn.softmax"), "softmax"
                    ),
                    tools.fused_op(
                        _layer_source(layer, "self_attn"), "pv_matmul"
                    ),
                    tools.fused_op(
                        _layer_source(layer, "self_attn.o_proj"),
                        "output_projection",
                    ),
                ],
                attributes={
                    "causal": True,
                    "softmax_scale": 1.0 / math.sqrt(float(HEAD_DIM)),
                    "sliding_window": 0,
                    "has_output_projection": True,
                },
                input_edges=[
                    prefix + ".q",
                    prefix + ".k",
                    prefix + ".v",
                    "weight." + prefix + ".o",
                ],
                output_edges=[prefix + ".attn_out"],
                precision=torch.bfloat16,
                kernel_key="attention",
                fusion_rule_id="qwen3.block.attention.v1",
            )
        )
        builder.add_node(
            Node(
                name=prefix + ".attn_residual",
                op_type=OpType.kResidualAdd,
                fused_ops=[tools.fused_op(prefix, "add")],
                attributes={"precision": "bfloat16"},
                input_edges=[prefix + ".attn_out", hidden_in],
                output_edges=[prefix + ".hidden_mid"],
                precision=torch.bfloat16,
                kernel_key="residual_add",
                fusion_rule_id="qwen3.block.residual.attn.v1",
            )
        )
        builder.add_node(
            Node(
                name=prefix + ".mlp_norm",
                op_type=OpType.kRmsNorm,
                fused_ops=[
                    tools.fused_op(
                        _layer_source(layer, "post_attention_layernorm"),
                        "rmsnorm",
                    )
                ],
                attributes={"eps": NORM_EPS, "precision": "bfloat16"},
                input_edges=[
                    prefix + ".hidden_mid",
                    "weight." + prefix + ".mlp_norm",
                ],
                output_edges=[prefix + ".mlp_normed"],
                precision=torch.bfloat16,
                kernel_key="rms_norm",
                fusion_rule_id="qwen3.block.norm.mlp.v1",
            )
        )
        builder.add_node(
            Node(
                name=prefix + ".mlp_gate_up",
                op_type=OpType.kMlpGateUpAct,
                fused_ops=[
                    tools.fused_op(
                        _layer_source(layer, "mlp.gate_proj"), "gate_linear"
                    ),
                    tools.fused_op(
                        _layer_source(layer, "mlp.up_proj"), "up_linear"
                    ),
                    tools.fused_op(_layer_source(layer, "mlp.act_fn"), "silu"),
                    tools.fused_op(_layer_source(layer, "mlp"), "mul"),
                ],
                attributes={
                    "activation": "silu",
                    "intermediate_size": INTERMEDIATE_SIZE,
                },
                input_edges=[
                    prefix + ".mlp_normed",
                    "weight." + prefix + ".gate",
                    "weight." + prefix + ".up",
                ],
                output_edges=[prefix + ".mlp_act"],
                precision=torch.bfloat16,
                kernel_key="mlp_gate_up_act",
                fusion_rule_id="qwen3.block.gate_up.v1",
            )
        )
        builder.add_node(
            Node(
                name=prefix + ".mlp_down_residual",
                op_type=OpType.kMlpDownResidual,
                fused_ops=[
                    tools.fused_op(
                        _layer_source(layer, "mlp.down_proj"), "down_linear"
                    ),
                    tools.fused_op(prefix, "add"),
                ],
                attributes={"precision": "bfloat16"},
                input_edges=[
                    prefix + ".mlp_act",
                    "weight." + prefix + ".down",
                    prefix + ".hidden_mid",
                ],
                output_edges=[prefix + ".hidden_out"],
                precision=torch.bfloat16,
                kernel_key="mlp_down_residual",
                fusion_rule_id="qwen3.block.down_residual.v1",
            )
        )

    last_hidden = f"layer.{NUM_LAYERS - 1}.hidden_out"
    builder.add_node(
        Node(
            name="final_norm",
            op_type=OpType.kRmsNorm,
            fused_ops=[tools.fused_op("model.norm", "rmsnorm")],
            attributes={"eps": NORM_EPS, "precision": "bfloat16"},
            input_edges=[last_hidden, "weight.final_norm"],
            output_edges=["final_normed"],
            precision=torch.bfloat16,
            kernel_key="rms_norm",
        )
    )
    builder.add_node(
        Node(
            name="lm_head",
            op_type=OpType.kLmHead,
            fused_ops=[
                tools.fused_op("lm_head", "last_token_gather"),
                tools.fused_op("lm_head", "linear"),
            ],
            attributes={"gather_last_token": True},
            input_edges=[
                "final_normed",
                "last_token_indices",
                "weight.lm_head",
            ],
            output_edges=["logits"],
            precision=torch.bfloat16,
            kernel_key="lm_head",
        )
    )
    builder.add_node(
        Node(
            name="sampling",
            op_type=OpType.kSampling,
            fused_ops=[tools.fused_op("sampling", "sampling")],
            input_edges=["logits"],
            output_edges=["token_ids"],
            precision=torch.int64,
            kernel_key="sampling",
        )
    )
    output_ids = [
        builder.add_node(
            tools.output_node(
                "output.token_ids",
                "token_ids",
                "token_ids",
                _cpu((BATCH_EXAMPLE,), torch.int64),
            )
        )
    ]
    return input_ids, output_ids


def _add_edges(builder: Any, weights: dict[str, torch.Tensor]) -> None:
    """按固定顺序添加全部边。"""
    hidden = tools.hidden_shape(BATCH_EXAMPLE, SEQUENCE_EXAMPLE, HIDDEN_SIZE)
    builder.add_edge(
        tools.activation_edge(
            "input_ids",
            tools.concrete_shape((BATCH_EXAMPLE, SEQUENCE_EXAMPLE)),
            torch.int64,
        )
    )
    builder.add_edge(
        tools.activation_edge(
            "position_ids",
            tools.concrete_shape((BATCH_EXAMPLE, SEQUENCE_EXAMPLE)),
            torch.int64,
        )
    )
    builder.add_edge(
        tools.activation_edge(
            "last_token_indices",
            tools.concrete_shape((BATCH_EXAMPLE,)),
            torch.int64,
        )
    )
    builder.add_edge(
        tools.weight_edge(
            "weight.embedding", weights["embedding.weight"], "embedding.weight"
        )
    )
    builder.add_edge(tools.activation_edge("hidden", hidden, torch.bfloat16))

    qkv_width = NUM_ATTENTION_HEADS * HEAD_DIM
    kv_width = NUM_KV_HEADS * HEAD_DIM
    for layer in range(NUM_LAYERS):
        prefix = _layer_prefix(layer)
        builder.add_edge(
            tools.weight_edge(
                "weight." + prefix + ".attn_norm",
                weights[prefix + ".attn_norm.weight"],
                prefix + ".attn_norm.weight",
            )
        )
        builder.add_edge(
            tools.activation_edge(
                prefix + ".attn_normed", hidden, torch.bfloat16
            )
        )
        for suffix in ("q", "k", "v", "q_norm", "k_norm"):
            ref = prefix + "." + suffix + ".weight"
            builder.add_edge(
                tools.weight_edge(
                    "weight." + prefix + "." + suffix, weights[ref], ref
                )
            )
        builder.add_edge(
            tools.activation_edge(
                prefix + ".q",
                tools.sequence_shape(
                    BATCH_EXAMPLE, SEQUENCE_EXAMPLE, qkv_width
                ),
                torch.bfloat16,
            )
        )
        builder.add_edge(
            tools.activation_edge(
                prefix + ".k",
                tools.sequence_shape(BATCH_EXAMPLE, SEQUENCE_EXAMPLE, kv_width),
                torch.bfloat16,
            )
        )
        builder.add_edge(
            tools.activation_edge(
                prefix + ".v",
                tools.sequence_shape(BATCH_EXAMPLE, SEQUENCE_EXAMPLE, kv_width),
                torch.bfloat16,
            )
        )
        builder.add_edge(
            tools.weight_edge(
                "weight." + prefix + ".o",
                weights[prefix + ".o.weight"],
                prefix + ".o.weight",
            )
        )
        builder.add_edge(
            tools.activation_edge(
                prefix + ".attn_out", hidden, torch.bfloat16
            )
        )
        builder.add_edge(
            tools.activation_edge(
                prefix + ".hidden_mid", hidden, torch.bfloat16
            )
        )
        builder.add_edge(
            tools.weight_edge(
                "weight." + prefix + ".mlp_norm",
                weights[prefix + ".mlp_norm.weight"],
                prefix + ".mlp_norm.weight",
            )
        )
        builder.add_edge(
            tools.activation_edge(
                prefix + ".mlp_normed", hidden, torch.bfloat16
            )
        )
        for suffix in ("gate", "up"):
            ref = prefix + "." + suffix + ".weight"
            builder.add_edge(
                tools.weight_edge(
                    "weight." + prefix + "." + suffix, weights[ref], ref
                )
            )
        builder.add_edge(
            tools.activation_edge(
                prefix + ".mlp_act",
                tools.sequence_shape(
                    BATCH_EXAMPLE, SEQUENCE_EXAMPLE, INTERMEDIATE_SIZE
                ),
                torch.bfloat16,
            )
        )
        builder.add_edge(
            tools.weight_edge(
                "weight." + prefix + ".down",
                weights[prefix + ".down.weight"],
                prefix + ".down.weight",
            )
        )
        builder.add_edge(
            tools.activation_edge(
                prefix + ".hidden_out", hidden, torch.bfloat16
            )
        )

    builder.add_edge(
        tools.weight_edge(
            "weight.final_norm",
            weights["final_norm.weight"],
            "final_norm.weight",
        )
    )
    builder.add_edge(
        tools.activation_edge("final_normed", hidden, torch.bfloat16)
    )
    builder.add_edge(
        tools.weight_edge(
            "weight.lm_head", weights["lm_head.weight"], "lm_head.weight"
        )
    )
    builder.add_edge(
        tools.activation_edge(
            "logits",
            tools.logits_shape(BATCH_EXAMPLE, VOCAB_SIZE),
            torch.bfloat16,
        )
    )
    builder.add_edge(
        tools.activation_edge(
            "token_ids", tools.concrete_shape((BATCH_EXAMPLE,)), torch.int64
        )
    )
