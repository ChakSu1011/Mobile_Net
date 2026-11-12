"""模块：Qwen3 模型配置读取与校验。

职责：
  定义 Qwen3-0.6B 的超参数数据类，读取 Transformers 风格的
  ``config.json``，并在组图前校验当前实现支持的字段范围。

边界：
  - 只负责配置解析和静态校验，不读取权重、不创建 Graph；
  - 不依赖 transformers/torch，运行时只使用 Python 标准库；
  - 只支持 Qwen3 稠密模型，不支持 MoE、VL、Next 等变体。

输入输出：
  ``Qwen3Config.from_json()`` 输入 JSON 路径，返回不可变配置对象；
  非法字段抛出 ``ValueError``，文件不存在时由 ``open()`` 抛出
  ``FileNotFoundError``。

主要逻辑：
  1. 兼容 Transformers 5.9 的 ``rope_parameters`` 和旧版 ``rope_theta``；
  2. 把 JSON 字段转换为强类型 dataclass 字段；
  3. 在 ``validate()`` 中集中检查正数、整除、dtype、activation 等约束。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
_SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")
_SUPPORTED_ROPE_TYPES = ("default",)


@dataclass(frozen=True)
class Qwen3Config:
    """Qwen3 模型超参数。

    功能：
      保存组图所需的全部静态配置，并提供 q/kv 投影宽度等派生值。

    边界：
      不保存 tokenizer、权重路径或运行时设备信息；不执行 shape 推导。

    字段：
      model_type：模型族，当前只接受 ``qwen3``；
      vocab_size：词表大小；
      hidden_size：residual stream 宽度；
      intermediate_size：MLP 中间宽度；
      num_hidden_layers：Decoder Layer 数量；
      num_attention_heads：Q 头数；
      num_key_value_heads：K/V 头数；
      head_dim：每个头的宽度；
      hidden_act：MLP 激活函数，当前只支持 ``silu``；
      max_position_embeddings：最大位置数；
      rms_norm_eps：RMSNorm 的 epsilon；
      rope_theta：RoPE 基频；
      rope_type：RoPE 类型，当前只支持 ``default``；
      tie_word_embeddings：LM Head 是否复用 embedding 权重；
      attention_bias：q/k/v/o 是否带 bias，当前只支持 False；
      use_sliding_window：是否启用 sliding attention，当前只支持 False；
      sliding_window：sliding window 大小，关闭时可为 None；
      dtype：权重和激活声明精度。
    """

    model_type: str = "qwen3"
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_type: str = "default"
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = None
    dtype: str = "bfloat16"

    @property
    def q_width(self) -> int:
        """返回 Q 投影输出宽度。"""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        """返回 K/V 投影输出宽度。"""
        return self.num_key_value_heads * self.head_dim

    @property
    def gqa_ratio(self) -> int:
        """返回 Q head 到 KV head 的复制倍数。"""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3Config":
        """从 JSON 文件读取配置。

        参数：
          path：Transformers 风格 config.json 路径。

        返回：
          Qwen3Config 实例；字段缺失时使用 Qwen3-0.6B 默认值。

        异常：
          FileNotFoundError：文件不存在；
          ValueError：JSON 不是对象、rope_parameters 类型错误或
            rope_scaling 非空。
        """
        with open(path, "r", encoding="utf-8") as config_file:
            data = json.load(config_file)
        if not isinstance(data, dict):
            raise ValueError("config.json must contain a JSON object")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3Config":
        """从 Transformers 风格字典构造配置。

        参数：
          data：config.json 解析后的字典。

        返回：
          Qwen3Config 实例。

        异常：
          ValueError：字段类型或 rope 配置不合法。
        """
        rope_theta = float(data.get("rope_theta", 1_000_000.0))
        rope_type = "default"
        rope_parameters = data.get("rope_parameters")
        if rope_parameters is not None:
            if not isinstance(rope_parameters, dict):
                raise ValueError(
                    "rope_parameters must be a dict or null")
            rope_theta = float(
                rope_parameters.get("rope_theta", rope_theta))
            rope_type = str(
                rope_parameters.get("rope_type", rope_type))
        if data.get("rope_scaling"):
            raise ValueError("rope_scaling is not supported")

        num_key_value_heads = data.get(
            "num_key_value_heads", cls.num_key_value_heads)
        if num_key_value_heads is None:
            num_key_value_heads = data.get(
                "num_attention_heads", cls.num_attention_heads)

        raw_sliding_window = data.get("sliding_window")
        sliding_window = (
            None if raw_sliding_window is None
            else int(raw_sliding_window)
        )
        dtype = data.get("dtype", data.get("torch_dtype", "bfloat16"))
        return cls(
            model_type=str(data.get("model_type", cls.model_type)),
            vocab_size=int(data.get("vocab_size", cls.vocab_size)),
            hidden_size=int(data.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(
                data.get("intermediate_size", cls.intermediate_size)),
            num_hidden_layers=int(
                data.get("num_hidden_layers", cls.num_hidden_layers)),
            num_attention_heads=int(
                data.get("num_attention_heads",
                         cls.num_attention_heads)),
            num_key_value_heads=int(num_key_value_heads),
            head_dim=int(data.get("head_dim", cls.head_dim)),
            hidden_act=str(data.get("hidden_act", cls.hidden_act)),
            max_position_embeddings=int(
                data.get("max_position_embeddings",
                         cls.max_position_embeddings)),
            rms_norm_eps=float(
                data.get("rms_norm_eps", cls.rms_norm_eps)),
            rope_theta=rope_theta,
            rope_type=rope_type,
            tie_word_embeddings=bool(
                data.get("tie_word_embeddings",
                         cls.tie_word_embeddings)),
            attention_bias=bool(
                data.get("attention_bias", cls.attention_bias)),
            use_sliding_window=bool(
                data.get("use_sliding_window",
                         cls.use_sliding_window)),
            sliding_window=sliding_window,
            dtype=str(dtype),
        )

    def validate(self) -> None:
        """校验配置是否在当前组图能力范围内。

        返回：
          无；校验通过时直接返回。

        异常：
          ValueError：字段非正、GQA 不能整除、activation/bias/dtype/
            RoPE/sliding window 不受支持。
        """
        if self.model_type != "qwen3":
            raise ValueError(
                f"unsupported model_type: {self.model_type}")

        positive_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.hidden_act != "silu":
            raise ValueError(
                f"unsupported hidden_act: {self.hidden_act}")
        if self.attention_bias:
            raise ValueError("attention_bias must be false")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.rope_type not in _SUPPORTED_ROPE_TYPES:
            raise ValueError(
                f"unsupported rope_type: {self.rope_type}")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads")
        if self.use_sliding_window:
            raise ValueError("use_sliding_window is not supported")
        if self.sliding_window is not None and self.sliding_window < 0:
            raise ValueError("sliding_window must not be negative")
