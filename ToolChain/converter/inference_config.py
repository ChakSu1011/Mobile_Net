"""模块：模型运行时输入配置。

职责：
  读取 ``Models/<model>/input/inference_config.json``，与官方
  ``config.json``、``generation_config.json`` 合并为引擎运行参数，
  并完成类型、范围与重复声明校验。

边界：
  - 不执行 Tokenization、采样或推理；
  - 不读取权重，不参与构图；
  - 新配置文件缺失时返回 None，保持旧模型目录可转换。

主要逻辑：
  1. 官方 config 已有参数（context、vocab、采样、eos 等）只从官方文件
     读取，新配置只接受引擎补充字段；
  2. generation_config 缺失时按缺省行为取值；
  3. 合并为 ``InferenceConfig`` 并校验与 vocab/context 的关系；
  4. 由 Adapter 写入 ``ModelGraph.metadata``，供 Serializer 与开发者读取。

输入输出：
  ``load_inference_config()`` 输入模型目录与官方配置路径，返回有效配置
  或 None；非法字段、缺失文件（显式传入时）抛 ``BuildError``。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ToolChain.common.errors import BuildError
from ToolChain.common.jsonio import read_json

INFERENCE_DIR = "input"
INFERENCE_FILE = "inference_config.json"

_ENGINE_FIELDS = frozenset({
    "max_num_tokens",
    "max_batch_size",
    "repetition_penalty",
    "seed",
})

_DERIVED_FIELDS = {
    "max_new_tokens": "context_length - prompt_length",
}

_REPEATED_FIELDS = {
    "context_length": "config.json.max_position_embeddings",
    "max_position_embeddings": "config.json.max_position_embeddings",
    "vocab_size": "config.json.vocab_size",
    "bos_token_id": "config.json or generation_config.json",
    "eos_token_id": "config.json or generation_config.json",
    "pad_token_id": "config.json or generation_config.json",
    "do_sample": "generation_config.json.do_sample",
    "temperature": "generation_config.json.temperature",
    "top_k": "generation_config.json.top_k",
    "top_p": "generation_config.json.top_p",
    "stop_token_ids": "generation_config.json.eos_token_id",
}


@dataclass(frozen=True)
class InferenceConfig:
    """合并后的有效运行参数。

    功能：
      把官方 config 字段与引擎补充字段合并为单个不可变对象，供
      ``ModelGraph.metadata``、转换报告与流水线默认值使用。

    边界：
      只描述参数，不执行采样或 Tokenization；不包含 tokenizer 文件路径。

    属性：
      context_length：单请求上下文上限，来自官方 config；
      vocab_size：词表大小，来自官方 config；
      max_num_tokens：一次推理 Step 内所有请求累加的 token 预算；
      max_batch_size：规划 batch 上界；
      do_sample：是否采样，来自 generation_config；
      temperature：采样温度；
      top_k：top-k 阈值，0 表示关闭；
      top_p：nucleus 采样阈值；
      repetition_penalty：重复惩罚；
      seed：随机种子，0 表示非固定种子；
      stop_token_ids：停止 token，来自官方 eos；
      bos_token_id：起始 token；
      pad_token_id：填充 token。
    """

    context_length: int
    vocab_size: int
    max_num_tokens: int
    max_batch_size: int
    do_sample: bool
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    seed: int
    stop_token_ids: tuple[int, ...]
    bos_token_id: int | None
    pad_token_id: int | None

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 原生的有效配置。

        返回：
          字段顺序固定；``stop_token_ids`` 转为 list。
        """
        return {
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "max_num_tokens": self.max_num_tokens,
            "max_batch_size": self.max_batch_size,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "stop_token_ids": list(self.stop_token_ids),
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
        }


def _error(source: str, key: str, message: str) -> BuildError:
    """构造带文件与字段位置的配置错误。"""
    return BuildError(
        "converter",
        message,
        target=f"{source}:{key}",
    )


def _require_int(
    data: dict[str, Any],
    key: str,
    source: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """读取并校验整数字段。

    参数：
      data：配置字典；
      key：字段名；
      source：错误信息中的来源；
      minimum：最小值（含）；
      maximum：最大值（含）；None 表示不限制。

    返回：
      整数值。

    异常：
      BuildError：类型错误或越界。
    """
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(source, key, "must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}"
        if maximum is not None:
            bound += f" and <= {maximum}"
        raise _error(source, key, f"must be {bound}")
    return value


def _require_number(
    data: dict[str, Any],
    key: str,
    source: str,
    *,
    minimum: float,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    """读取并校验数值字段。

    参数：
      data：配置字典；
      key：字段名；
      source：错误信息中的来源；
      minimum：下界；
      maximum：上界（含）；None 表示不限制；
      exclusive_minimum：为 True 时下界为开区间。

    返回：
      float 值。

    异常：
      BuildError：类型、非有限值或越界。
    """
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(source, key, "must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise _error(source, key, "must be finite")
    too_small = (
        number <= minimum if exclusive_minimum else number < minimum)
    if too_small or (maximum is not None and number > maximum):
        bound = f"> {minimum}" if exclusive_minimum else f">= {minimum}"
        if maximum is not None:
            bound += f" and <= {maximum}"
        raise _error(source, key, f"must be {bound}")
    return number


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象文件。

    参数：
      path：文件路径。

    返回：
      解析后的字典。

    异常：
      BuildError：文件不存在、JSON 非法或内容不是对象。
    """
    try:
        data = read_json(path)
    except OSError as error:
        raise BuildError(
            "converter",
            f"cannot read config: {error}",
            target=str(path),
        ) from error
    except json.JSONDecodeError as error:
        raise BuildError(
            "converter",
            f"invalid JSON: {error}",
            target=str(path),
        ) from error
    except UnicodeDecodeError as error:
        raise BuildError(
            "converter",
            f"invalid UTF-8: {error}",
            target=str(path),
        ) from error
    if not isinstance(data, dict):
        raise BuildError(
            "converter",
            "config must contain a JSON object",
            target=str(path),
        )
    return data


def _read_optional_json_object(path: Path) -> dict[str, Any] | None:
    """读取可选 JSON 对象文件；不存在时返回 None。"""
    if not path.is_file():
        return None
    return _read_json_object(path)


def _token_id(
    value: Any,
    *,
    source: str,
    key: str,
    vocab_size: int,
) -> int:
    """校验单个 Token ID。

    参数：
      value：待校验值；
      source：错误信息中的来源；
      key：字段名；
      vocab_size：词表大小。

    返回：
      合法 Token ID。

    异常：
      BuildError：不是整数或超出词表。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(source, key, "must be an integer")
    if value < 0 or value >= vocab_size:
        raise _error(
            source, key, f"must be within [0, {vocab_size})")
    return value


def _optional_token_id(
    generation: dict[str, Any],
    official: dict[str, Any],
    key: str,
    *,
    generation_source: str,
    official_source: str,
    vocab_size: int,
) -> int | None:
    """从 generation_config 读取可选 Token ID，缺失/null 时回退官方 config。"""
    if key in generation and generation[key] is not None:
        source = generation_source
        value = generation[key]
    elif key in official and official[key] is not None:
        source = official_source
        value = official[key]
    else:
        return None
    return _token_id(
        value, source=source, key=key, vocab_size=vocab_size)


def _stop_token_ids(
    generation: dict[str, Any],
    official: dict[str, Any],
    *,
    generation_source: str,
    official_source: str,
    vocab_size: int,
) -> tuple[int, ...]:
    """从官方 eos_token_id 规范化停止 Token 列表。

    参数：
      generation：generation_config 字典；
      official：config.json 字典；
      generation_source：generation_config 路径文本；
      official_source：config.json 路径文本；
      vocab_size：词表大小。

    返回：
      去重且保持顺序的 Token ID 元组；官方未声明时为空。

    异常：
      BuildError：不是 int/int[] 或 Token ID 越界。
    """
    if generation.get("eos_token_id") is not None:
        source = generation_source
        raw = generation["eos_token_id"]
    elif official.get("eos_token_id") is not None:
        source = official_source
        raw = official["eos_token_id"]
    else:
        return ()
    values = raw if isinstance(raw, list) else [raw]
    result: list[int] = []
    for item in values:
        token = _token_id(
            item, source=source, key="eos_token_id",
            vocab_size=vocab_size)
        if token not in result:
            result.append(token)
    return tuple(result)


def _validate_authored(
    data: dict[str, Any],
    *,
    source: str,
    context_length: int,
) -> dict[str, Any]:
    """校验新配置文件并返回引擎补充字段。

    参数：
      data：新配置解析结果；
      source：文件路径文本；
      context_length：官方上下文上限。

    返回：
      只含引擎字段的字典。

    异常：
      BuildError：字段缺失、未知、重复声明官方字段、类型错误或越界。
    """
    keys = set(data)
    missing = sorted(_ENGINE_FIELDS - keys)
    if missing:
        raise BuildError(
            "converter",
            f"missing field(s): {', '.join(missing)}",
            target=source,
        )
    repeated = sorted(keys & set(_REPEATED_FIELDS))
    if repeated:
        key = repeated[0]
        raise _error(
            source,
            key,
            "official parameter must be read from "
            f"{_REPEATED_FIELDS[key]}, not redeclared",
        )
    derived = sorted(keys & set(_DERIVED_FIELDS))
    if derived:
        key = derived[0]
        raise _error(
            source,
            key,
            "is derived at runtime from "
            f"{_DERIVED_FIELDS[key]}",
        )
    unknown = sorted(
        keys - _ENGINE_FIELDS - set(_REPEATED_FIELDS)
        - set(_DERIVED_FIELDS))
    if unknown:
        raise _error(
            source, unknown[0], "unknown field")

    max_num_tokens = _require_int(
        data, "max_num_tokens", source,
        minimum=1, maximum=context_length)
    max_batch_size = _require_int(
        data, "max_batch_size", source, minimum=1)
    repetition_penalty = _require_number(
        data, "repetition_penalty", source,
        minimum=0.0, exclusive_minimum=True)
    seed = _require_int(data, "seed", source, minimum=0)
    return {
        "max_num_tokens": max_num_tokens,
        "max_batch_size": max_batch_size,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
    }


def _sampling_value(
    data: dict[str, Any],
    key: str,
    *,
    source: str,
    default: Any,
    kind: str,
    minimum: float,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> Any:
    """读取官方采样字段；缺失或为 null 时返回缺省值。"""
    if key not in data or data[key] is None:
        return default
    if kind == "bool":
        value = data[key]
        if not isinstance(value, bool):
            raise _error(source, key, "must be a boolean")
        return value
    if kind == "int":
        return _require_int(
            data, key, source,
            minimum=int(minimum),
            maximum=None if maximum is None else int(maximum))
    return _require_number(
        data, key, source,
        minimum=minimum,
        maximum=maximum,
        exclusive_minimum=exclusive_minimum,
    )


def load_inference_config(
    model_dir: Path,
    *,
    config_path: Path,
    vocab_size: int,
    max_position_embeddings: int,
    inference_config_path: Path | None = None,
) -> InferenceConfig | None:
    """加载并合并模型运行参数。

    参数：
      model_dir：模型目录，用于解析相对路径；
      config_path：官方 config.json 路径；
      vocab_size：官方词表大小；
      max_position_embeddings：官方最大位置数，作为 context_length；
      inference_config_path：可选显式新配置路径；省略时按
        ``model_dir/input/inference_config.json`` 自动发现。

    返回：
      合并后的 ``InferenceConfig``；自动发现时文件不存在返回 None。

    异常：
      BuildError：显式文件不存在、JSON 非法、字段非法或官方
        generation_config 非法。
    """
    model_dir = Path(model_dir)
    config_path = Path(config_path)
    official = _read_json_object(config_path)
    official_source = str(config_path)

    generation_path = model_dir / "generation_config.json"
    generation = _read_optional_json_object(generation_path)
    generation_source = str(generation_path)
    if generation is None:
        generation = {}

    if inference_config_path is None:
        authored_path = model_dir / INFERENCE_DIR / INFERENCE_FILE
        if not authored_path.is_file():
            return None
    else:
        authored_path = Path(inference_config_path)
        if not authored_path.is_absolute():
            authored_path = model_dir / authored_path
        if not authored_path.is_file():
            raise BuildError(
                "converter",
                "inference config file not found",
                target=str(authored_path),
            )

    authored = _read_json_object(authored_path)
    authored_source = str(authored_path)
    engine = _validate_authored(
        authored,
        source=authored_source,
        context_length=max_position_embeddings,
    )

    do_sample = _sampling_value(
        generation, "do_sample",
        source=generation_source, default=False, kind="bool",
        minimum=0, maximum=None)
    temperature = _sampling_value(
        generation, "temperature",
        source=generation_source, default=1.0, kind="number",
        minimum=0.0, exclusive_minimum=True)
    top_k = _sampling_value(
        generation, "top_k",
        source=generation_source, default=0, kind="int",
        minimum=0, maximum=vocab_size)
    top_p = _sampling_value(
        generation, "top_p",
        source=generation_source, default=1.0, kind="number",
        minimum=0.0, maximum=1.0, exclusive_minimum=True)
    bos_token_id = _optional_token_id(
        generation, official, "bos_token_id",
        generation_source=generation_source,
        official_source=official_source,
        vocab_size=vocab_size)
    pad_token_id = _optional_token_id(
        generation, official, "pad_token_id",
        generation_source=generation_source,
        official_source=official_source,
        vocab_size=vocab_size)
    stop_token_ids = _stop_token_ids(
        generation, official,
        generation_source=generation_source,
        official_source=official_source,
        vocab_size=vocab_size)

    return InferenceConfig(
        context_length=max_position_embeddings,
        vocab_size=vocab_size,
        max_num_tokens=engine["max_num_tokens"],
        max_batch_size=engine["max_batch_size"],
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=engine["repetition_penalty"],
        seed=engine["seed"],
        stop_token_ids=stop_token_ids,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
    )
