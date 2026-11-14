"""模块：模型转换编排。

职责：
  按 Adapter 名称选择模型适配器，执行转换，并把 ``ModelGraph`` 与转换
  报告写入指定目录。

边界：
  不实现具体模型结构；Qwen3 逻辑位于 ``qwen3_adapter.py``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ToolChain.common.errors import BuildError
from ToolChain.common.jsonio import write_json
from ToolChain.converter.qwen3_adapter import Qwen3Adapter
from ToolChain.converter.result import ConversionResult

_ADAPTERS = {
    Qwen3Adapter.name: Qwen3Adapter,
}


def convert(
    adapter_name: str,
    config_path: Path,
    *,
    weights_path: Path | None = None,
) -> ConversionResult:
    """按 Adapter 名称执行模型转换。

    参数：
      adapter_name：已注册的 Adapter 名称，首期只支持 ``qwen3``；
      config_path：模型配置路径；
      weights_path：可选 safetensors 文件或目录。

    返回：
      ``ConversionResult``。

    异常：
      BuildError：Adapter 未注册；
      ValueError/RuntimeError：构图或配置失败。
    """
    adapter_type = _ADAPTERS.get(adapter_name)
    if adapter_type is None:
        raise BuildError(
            "converter",
            f"unsupported adapter: {adapter_name}",
        )
    return adapter_type(config_path, weights_path).build()


def convert_qwen3(
    config_path: Path,
    *,
    weights_path: Path | None = None,
) -> ConversionResult:
    """转换 Qwen3 模型的便捷入口。"""
    return convert(
        Qwen3Adapter.name,
        config_path,
        weights_path=weights_path,
    )


def conversion_report(result: ConversionResult) -> dict[str, Any]:
    """生成确定性转换报告。

    参数：
      result：转换结果。

    返回：
      JSON 可序列化报告；不包含时间戳。

    异常：
      BuildError：元数据缺少 IR 哈希。
    """
    model = result.model_graph
    metadata = model.metadata
    ir_sha256 = metadata.get("source_ir_sha256")
    if not isinstance(ir_sha256, str):
        raise BuildError(
            "converter",
            "model graph metadata has no source IR hash",
            target=model.name,
        )
    constants = []
    for name in model.constants:
        tensor = model.tensor(name)
        constants.append({
            "name": name,
            "shape": [
                dim.to_json() for dim in tensor.shape],
            "precision": tensor.precision,
            "byte_size": tensor.byte_size,
            "weight_source": tensor.weight_source,
            "weight_available": tensor.weight_available,
        })
    return {
        "adapter": result.adapter,
        "graph": model.name,
        "summary": model.summary(),
        "ir_sha256": ir_sha256,
        "ir_line_count": metadata.get("source_ir_line_count"),
        "model_config": metadata.get("model_config", {}),
        "tied_weights": metadata.get("tied_weights", {}),
        "weights_supplied": metadata.get(
            "weights_supplied", False),
        "weight_issues": metadata.get("weight_issues", []),
        "constants": constants,
    }


def write_conversion_outputs(
    result: ConversionResult,
    output_dir: Path,
) -> dict[str, Path]:
    """把模型图与转换报告写入目录。

    参数：
      result：转换结果；
      output_dir：输出目录；不存在时创建。

    返回：
      文件角色到路径的映射：``model_graph``、``report``。

    异常：
      OSError：目录或文件写入失败。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model_graph.json"
    report_path = output_dir / "conversion_report.json"
    result.model_graph.save_json(model_path)
    write_json(report_path, conversion_report(result))
    return {
        "model_graph": model_path,
        "report": report_path,
    }
