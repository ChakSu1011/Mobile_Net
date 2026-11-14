"""模块：Serializer 命令行入口。

职责：
  提供 ``inspect`` 与 ``rebuild`` 子命令，用于检查模型包并验证图重建。

边界：
  不生成模型包；包由 Pipeline 调用 ``serialize_package`` 生成。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ToolChain.common.errors import BuildError
from ToolChain.serializer.loader import (
    dump_manifest,
    load_package,
    package_summary,
    rebuild_graph,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析命令行参数。

    参数：
      argv：参数列表；None 表示使用 ``sys.argv[1:]``。

    返回：
      解析结果。
    """
    parser = argparse.ArgumentParser(
        prog="python -m ToolChain.serializer",
        description="Inspect or rebuild a MobileNet model package.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Print package summary.")
    inspect_parser.add_argument(
        "--package", type=Path, required=True,
        help="Directory containing model.mgf.")
    inspect_parser.add_argument(
        "--dump-manifest", action="store_true",
        help="Print the full manifest JSON.")

    rebuild_parser = subparsers.add_parser(
        "rebuild", help="Rebuild Graph graph from package.")
    rebuild_parser.add_argument(
        "--package", type=Path, required=True,
        help="Directory containing model.mgf.")
    rebuild_parser.add_argument(
        "--no-verify-ir", action="store_true",
        help="Skip IR SHA-256 verification.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行命令行。

    参数：
      argv：参数列表；None 表示使用 ``sys.argv[1:]``。

    返回：
      进程退出码；0 表示成功，1 表示构建错误。
    """
    args = _parse_args(argv)
    try:
        if args.command == "inspect":
            package = load_package(args.package, verify=True)
            summary = package_summary(package)
            for key, value in summary.items():
                print(f"{key} {value}")
            if args.dump_manifest:
                print(dump_manifest(args.package))
            return 0
        graph = rebuild_graph(
            load_package(args.package, verify=True),
            verify_ir=not args.no_verify_ir,
        )
        ir_text = graph.dump_ir()
        print(f"ir_lines {len(ir_text.splitlines())}")
        print(f"rebuild ok")
        return 0
    except (BuildError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
