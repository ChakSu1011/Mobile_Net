"""模块：Build 阶段端到端流水线。

职责：
  串联 Converter、Optimizer、Serializer，生成模型包后立即加载并重建
  Graph，校验 IR 哈希，输出构建摘要。

边界：
  - 只支持已注册 Adapter（首期为 Qwen3）；
  - 不训练、不量化、不执行推理；
  - 中间产物写入独立 work 目录，模型包目录只保留格式文件。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Build.common.errors import BuildError
from Build.converter.converter import (
    convert,
    write_conversion_outputs,
)
from Build.converter.result import ConversionResult
from Build.optimizer import Optimizer, OptimizeConfig
from Build.optimizer.optimizer import (
    OptimizeResult,
    write_optimization_outputs,
)
from Build.serializer import (
    ModelPackage,
    SerializationResult,
    load_package,
    package_summary,
    rebuild_graph,
    serialize_package,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPOSITORY_ROOT / "Models" / "Qwen3" / "config.json"
_DEFAULT_ARTIFACT = (
    _REPOSITORY_ROOT / "Build" / "artifacts" / "qwen3_0_6b")


@dataclass(frozen=True)
class PipelineResult:
    """端到端构建结果。

    属性：
      conversion：转换结果；
      optimization：优化结果；
      serialization：序列化结果；
      package：重新加载的模型包；
      graph：重建的 Graph；
      work_dir：中间产物目录。
    """

    conversion: ConversionResult
    optimization: OptimizeResult
    serialization: SerializationResult
    package: ModelPackage
    graph: Any
    work_dir: Path


def run_pipeline(
    *,
    config_path: Path,
    output_dir: Path,
    work_dir: Path | None = None,
    adapter: str = "qwen3",
    weights_path: Path | None = None,
    optimize_config: OptimizeConfig | None = None,
    overwrite: bool = True,
) -> PipelineResult:
    """执行完整 Build 流水线。

    参数：
      config_path：模型 config.json；
      output_dir：模型包输出目录；
      work_dir：中间产物目录；省略时为 ``<output>_work``；
      adapter：Adapter 名称；
      weights_path：可选 safetensors 文件或目录；
      optimize_config：优化配置；省略时使用默认值；
      overwrite：是否覆盖已有模型包。

    返回：
      ``PipelineResult``。

    异常：
      BuildError：任一阶段失败；
      OSError：文件读写失败。
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    if work_dir is None:
        work_dir = output_dir.with_name(output_dir.name + "_work")
    work_dir = Path(work_dir)
    config = optimize_config or OptimizeConfig()

    conversion = convert(
        adapter, config_path, weights_path=weights_path)
    write_conversion_outputs(
        conversion, work_dir / "converter")
    optimization = Optimizer(config).run(conversion.model_graph)
    write_optimization_outputs(
        optimization, work_dir / "optimizer")
    serialization = serialize_package(
        optimization,
        output_dir,
        weights_path=weights_path,
        overwrite=overwrite,
    )
    package = load_package(output_dir, verify=True)
    graph = rebuild_graph(package, verify_ir=True)
    return PipelineResult(
        conversion=conversion,
        optimization=optimization,
        serialization=serialization,
        package=package,
        graph=graph,
        work_dir=work_dir,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析命令行参数。

    参数：
      argv：参数列表；None 表示使用 ``sys.argv[1:]``。

    返回：
      解析结果。
    """
    parser = argparse.ArgumentParser(
        prog="python -m Build.pipeline",
        description=(
            "Convert, optimize, serialize, and verify a model package."
        ),
    )
    parser.add_argument(
        "--adapter", default="qwen3", help="Registered adapter name.")
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="Model config.json path.")
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_ARTIFACT,
        help="Model package output directory.")
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Intermediate output directory; defaults to <output>_work.")
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Optional safetensors file or directory.")
    parser.add_argument(
        "--target-precision", default="float16",
        choices=("float16", "bfloat16", "float32", "keep"),
        help="Target precision for floating point tensors.")
    parser.add_argument(
        "--max-num-tokens", type=int, default=256,
        help="Planning bound for symbolic dimension S.")
    parser.add_argument(
        "--context-length", type=int, default=40960,
        help="Maximum context length per request.")
    parser.add_argument(
        "--alignment", type=int, default=256,
        help="Memory alignment in bytes; must be a power of two.")
    parser.add_argument(
        "--disable-reuse", action="store_true",
        help="Disable activation memory reuse.")
    parser.add_argument(
        "--no-overwrite", action="store_true",
        help="Fail when the package directory is not empty.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行端到端构建命令行。

    参数：
      argv：参数列表；None 表示使用 ``sys.argv[1:]``。

    返回：
      进程退出码；0 表示成功，1 表示构建错误。
    """
    args = _parse_args(argv)
    try:
        result = run_pipeline(
            config_path=args.config,
            output_dir=args.output,
            work_dir=args.work_dir,
            adapter=args.adapter,
            weights_path=args.weights,
            optimize_config=OptimizeConfig(
                target_precision=args.target_precision,
                max_num_tokens=args.max_num_tokens,
                context_length=args.context_length,
                alignment=args.alignment,
                enable_reuse=not args.disable_reuse,
            ),
            overwrite=not args.no_overwrite,
        )
    except (BuildError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    summary = package_summary(result.package)
    print(f"adapter {summary['adapter']}")
    print(f"model {summary['model']}")
    print(f"tensors {summary['tensors']}")
    print(f"ops {summary['ops']}")
    print(f"constants {summary['constants']}")
    print(f"weights_available {summary['weights_available']}")
    print(f"weights_total_bytes {summary['weights_total_bytes']}")
    print(f"ir_sha256 {summary['ir_sha256']}")
    print(f"manifest_sha256 {summary['manifest_sha256']}")
    print(f"package {result.serialization.package_dir}")
    print(f"work_dir {result.work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
