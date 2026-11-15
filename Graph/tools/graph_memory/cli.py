"""模块：显存排布可视化命令行入口。

职责：
  解析参数、加载 memory plan、构建视图、布局并写出 SVG。

边界：
  不实现布局/渲染细节；错误转换为非零退出码。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ToolChain.common.errors import BuildError

from graph_memory.layout import layout_memory
from graph_memory.package import (
    LoadedPackage,
    load_model_package,
    resolve_package_dir,
)
from graph_memory.svg import render_memory_svg
from graph_memory.view_model import build_memory_view


def default_output_path(package_dir: Path) -> Path:
    """推导默认 SVG 输出路径。

    参数：
      package_dir：模型包目录。

    返回：
      ``<model-name>_memory.svg`` 路径。
    """
    package_dir = Path(package_dir)
    return package_dir.parent / f"{package_dir.name}_memory.svg"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析命令行参数。

    参数：
      argv：参数列表。

    返回：
      argparse 命名空间。
    """
    parser = argparse.ArgumentParser(
        prog="visualize_memory",
        description="Render model memory layout as an SVG diagram.",
    )
    parser.add_argument(
        "--package",
        type=Path,
        required=True,
        help="model.mgf file or package directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output SVG path",
    )
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    """执行显存可视化流程。

    参数：
      args：已解析参数。

    返回：
      进程退出码。
    """
    package: LoadedPackage = load_model_package(args.package)
    view = build_memory_view(package)
    layout = layout_memory(view)
    output = args.output
    if output is None:
        output = default_output_path(resolve_package_dir(args.package))
    render_memory_svg(view, layout, output)
    print(f"wrote {output}")
    print(f"activations {len(view.activation_blocks)}")
    print(f"weights {len(view.weight_blocks)}")
    print(f"planned_weights {view.planned_weights}")
    for warning in view.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    参数：
      argv：参数列表。

    返回：
      进程退出码。
    """
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    try:
        return _run(args)
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
