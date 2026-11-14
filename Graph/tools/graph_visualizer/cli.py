"""模块：Graph 静态图可视化命令行入口。

职责：
  解析参数、加载模型包、构建视图、计算布局并写出 SVG。

边界：
  - 不实现布局、渲染或模型包解析细节；
  - 只输出一个 SVG 文件与终端摘要；
  - 错误统一转换为非零退出码。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ToolChain.common.errors import BuildError

from graph_visualizer.layout import layout_view
from graph_visualizer.package import (
    LoadedPackage,
    load_model_package,
    resolve_package_dir,
)
from graph_visualizer.svg import render_svg
from graph_visualizer.view_model import build_view_model


def default_output_path(package_dir: Path) -> Path:
    """推导默认 SVG 输出路径。

    参数：
      package_dir：模型包目录，例如
        ``Models/Qwen3/artifacts/qwen3_0_6b``。

    返回：
      ``Models/Qwen3/artifacts/qwen3_0_6b_graph.svg`` 形式的路径。
    """
    package_dir = Path(package_dir)
    return package_dir.parent / f"{package_dir.name}_graph.svg"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析命令行参数。

    参数：
      argv：参数列表；None 表示使用 ``sys.argv[1:]``。

    返回：
      argparse 命名空间。
    """
    parser = argparse.ArgumentParser(
        prog="visualize_graph",
        description="Render a serialized Graph model as an SVG diagram.",
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
        help="output SVG path (default: <artifacts>/<model>_graph.svg)",
    )
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    """执行可视化流程。

    参数：
      args：已解析参数。

    返回：
      进程退出码；成功为 0。
    """
    package: LoadedPackage = load_model_package(args.package)
    view = build_view_model(package)
    layout = layout_view(view)
    output = args.output
    if output is None:
        output = default_output_path(resolve_package_dir(args.package))
    render_svg(view, layout, output)
    print(f"wrote {output}")
    print(f"nodes {len(view.nodes)}")
    print(f"tensor_edges {len(view.tensors)}")
    print(f"links {len(view.links)}")
    for warning in view.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    参数：
      argv：参数列表；None 表示使用 ``sys.argv[1:]``。

    返回：
      进程退出码：0 成功，2 参数错误，3 模型包错误，4 输出错误。
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
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
