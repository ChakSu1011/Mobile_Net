"""模块：融合算子组成、类别与颜色映射。

职责：
  为 ``OpMeta.op_type`` 提供稳定类别、散算子组成说明和节点颜色。

边界：
  - 只描述“显示用”的融合组成，不执行算子；
  - 未登记类型返回 ``other`` 和空组成，由调用方标记警告；
  - 组成表不改变 ModelGraph 或 Runtime 语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_OP_TYPE_LABELS = {
    "gather": "kGather",
    "expand_embedding": "kExpandEmbedding",
    "rms_norm": "kRmsNorm",
    "add_rms_norm": "kAddRmsNorm",
    "linear": "kLinear",
    "linear_mul": "kLinearMul",
    "linear_ape": "kLinearApe",
    "linear_rms_norm": "kLinearRmsNorm",
    "linear_rms_norm_rope": "kLinearRmsNormRope",
    "linear_rms_norm_rope_quant_paged_kv_store":
        "kLinearRmsNormRopeQuantPagedKVStore",
    "linear_rope_rotate_quant": "kLinearRopeRotateQuant",
    "rms_norm_rope_quant_paged_kv_store":
        "kRmsNormRopeQuantPagedKVStore",
    "rope_quant_linear": "kRopeQuantLinear",
    "mega_attention": "kMegaAttention",
    "sparse_mla": "kSparseMla",
    "linear_swi_glu": "kLinearSwiGlu",
    "ffn": "kFfn",
    "mega_moe": "kMegaMoe",
    "add": "kAdd",
    "rope": "kRope",
    "rotate": "kRotate",
    "deepseek_v3_router": "kDeepseekV3Router",
    "hash_router": "kHashRouter",
    "score_router": "kScoreRouter",
    "hc_pre": "kHcPre",
    "hc_post": "kHcPost",
    "hc_head": "kHcHead",
    "quant": "kQuant",
    "kv_compress": "kKvCompress",
    "index_score": "kIndexScore",
    "top_k": "kTopK",
    "reshape": "kReshape",
    "pack": "kPack",
    "unpack": "kUnpack",
    "input": "kInput",
    "output": "kOutput",
    "unknown": "kUnknown",
}

_CATEGORY_COLORS = {
    "io": "#2563eb",
    "embedding": "#0891b2",
    "norm": "#0f766e",
    "linear": "#475569",
    "attention": "#7c3aed",
    "mlp": "#b45309",
    "elementwise": "#334155",
    "position": "#c026d3",
    "routing": "#be123c",
    "quant": "#ca8a04",
    "reshape": "#64748b",
    "other": "#64748b",
}

_TABLE = {
    "gather": ("embedding", ("gather",)),
    "expand_embedding": ("embedding", ("embedding_expand",)),
    "rms_norm": (
        "norm",
        ("pow", "mean", "add(eps)", "rsqrt", "mul(weight)"),
    ),
    "add_rms_norm": ("norm", ("add", "rms_norm")),
    "linear": ("linear", ("matmul",)),
    "linear_mul": ("linear", ("linear", "mul")),
    "linear_ape": ("linear", ("linear", "ape")),
    "linear_rms_norm": ("linear", ("linear", "rms_norm")),
    "linear_rms_norm_rope": (
        "linear",
        ("linear", "rms_norm", "rope"),
    ),
    "linear_rms_norm_rope_quant_paged_kv_store": (
        "attention",
        ("linear", "rms_norm", "rope", "quant", "paged_kv_store"),
    ),
    "linear_rope_rotate_quant": (
        "linear",
        ("linear", "rope", "rotate", "quant"),
    ),
    "rms_norm_rope_quant_paged_kv_store": (
        "attention",
        ("rms_norm", "rope", "quant", "paged_kv_store"),
    ),
    "rope_quant_linear": (
        "linear",
        ("rope", "quant", "linear"),
    ),
    "mega_attention": (
        "attention",
        (
            "repeat_kv",
            "QK^T",
            "scale",
            "causal_mask",
            "softmax",
            "PV",
            "merge_heads",
        ),
    ),
    "sparse_mla": (
        "attention",
        ("q_proj", "kv_compress", "sparse_attention", "o_proj"),
    ),
    "linear_swi_glu": (
        "mlp",
        ("linear(gate)", "linear(up)", "silu", "mul"),
    ),
    "ffn": ("mlp", ("gate/up", "activation", "down")),
    "mega_moe": ("mlp", ("router", "expert_ffn", "combine")),
    "add": ("elementwise", ("add",)),
    "rope": ("position", ("cos/sin", "rotate_half", "mul", "add")),
    "rotate": ("position", ("rotate",)),
    "deepseek_v3_router": ("routing", ("router",)),
    "hash_router": ("routing", ("router",)),
    "score_router": ("routing", ("router",)),
    "hc_pre": ("other", ("hc_pre",)),
    "hc_post": ("other", ("hc_post",)),
    "hc_head": ("other", ("hc_head",)),
    "quant": ("quant", ("quantize",)),
    "kv_compress": ("quant", ("compress",)),
    "index_score": ("other", ("index_score",)),
    "top_k": ("other", ("top_k",)),
    "reshape": ("reshape", ("reshape",)),
    "pack": ("reshape", ("pack",)),
    "unpack": ("reshape", ("unpack",)),
    "input": ("io", ()),
    "output": ("io", ()),
    "unknown": ("other", ()),
}


@dataclass(frozen=True)
class OpDescription:
    """融合算子的显示描述。

    属性：
      category：类别名；
      composition：散算子组成；空元组表示未登记。
    """

    category: str
    composition: tuple[str, ...]


def describe_op(op_type: str, attrs: dict[str, Any]) -> OpDescription:
    """返回融合算子的类别与散算子组成。

    参数：
      op_type：ModelGraph 中的稳定 op 类型名，如 ``linear_swi_glu``；
      attrs：算子属性；当前仅用于 ``linear.bias`` 等少量变体。

    返回：
      ``OpDescription``；未知类型返回 ``other`` 与空组成。
    """
    category, composition = _TABLE.get(op_type, ("other", ()))
    if op_type == "linear" and bool(attrs.get("bias")):
        return OpDescription(category, ("matmul", "add(bias)"))
    return OpDescription(category, composition)


def op_type_label(op_type: str) -> str:
    """返回 Graph C++ 枚举风格的 op 类型名。

    参数：
      op_type：稳定 snake_case 类型名。

    返回：
      形如 ``kLinearSwiGlu`` 的显示名；未知类型原样返回。
    """
    return _OP_TYPE_LABELS.get(op_type, op_type)


def category_color(category: str) -> str:
    """返回类别对应的主色。

    参数：
      category：类别名。

    返回：
      十六进制颜色；未知类别使用中性灰。
    """
    return _CATEGORY_COLORS.get(category, _CATEGORY_COLORS["other"])


def is_known_op_type(op_type: str) -> bool:
    """判断 op 类型是否已登记组成表。

    参数：
      op_type：稳定 op 类型名。

    返回：
      已登记返回 true，否则 false。
    """
    return op_type in _TABLE
