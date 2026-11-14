"""模块：Optimizer 命令行入口。

职责：
  读取 Converter 输出的 ``model_graph.json``，执行优化并写出优化后的
  模型图与 ``optimize_report.json``。

边界：
  不执行构图、不序列化最终模型包；这些由 Converter 与 Serializer 负责。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ToolChain.common.errors import BuildError
from ToolChain.common.model_ir import ModelGraph
from ToolChain.optimizer.config import OptimizeConfig
from ToolChain.optimizer.optimizer import (
    Optimizer,
    write_optimization_outputs,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析命令行参数。

    参数：
      argv：参数列表；None 表示使用 ``sys.argv[1:]``。

    返回：
      解析结果。
    """
    parser = argparse.ArgumentParser(
        prog="python -m ToolChain.optimizer",
        description="Optimize a converter model_graph.json.",
    )
    parser.add_argument(
        "--model-graph",
        type=Path,
        required=True,
        help="Path to model_graph.json produced by converter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for optimized graph and report.",
    )
    parser.add_argument(
        "--target-precision",
        default="float16",
        choices=("float16", "bfloat16", "float32", "keep"),
        help="Target precision for floating point tensors.",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=256,
        help="Planning bound for symbolic dimension S.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=40960,
        help="Maximum context length per request.",
    )
    parser.add_argument(
        "--alignment",
        type=int,
        default=256,
        help="Memory alignment in bytes; must be a power of two.",
    )
    parser.add_argument(
        "--disable-reuse",
        action="store_true",
        help="Disable activation memory reuse for comparison.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行优化命令行。

    参数：
      argv：参数列表；None 表示使用 ``sys.argv[1:]``。

    返回：
      进程退出码；0 表示成功，1 表示构建错误。
    """
    args = _parse_args(argv)
    try:
        config = OptimizeConfig(
            target_precision=args.target_precision,
            max_num_tokens=args.max_num_tokens,
            context_length=args.context_length,
            alignment=args.alignment,
            enable_reuse=not args.disable_reuse,
        )
        model_graph = ModelGraph.load_json(args.model_graph)
        result = Optimizer(config).run(model_graph)
        paths = write_optimization_outputs(result, args.output)
    except (BuildError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    summary = result.model_graph.summary()
    print(f"model {summary['name']}")
    print(f"tensors {summary['total_tensors']}")
    print(f"activations {summary['activation_tensors']}")
    print(f"target_precision {config.target_precision}")
    print(f"arena_bytes {result.memory_plan.arena_bytes}")
    print(f"peak_bytes {result.memory_plan.peak_bytes}")
    print(f"constants_bytes {result.memory_plan.constants_bytes}")
    print(f"model_graph {paths['model_graph']}")
    print(f"report {paths['optimize_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
