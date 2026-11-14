"""模块：Graph 静态图可视化工具公共接口。

职责：
  导出模型包读取、视图模型构建与 SVG 渲染入口，供 CLI 和测试使用。

边界：
  不解析命令行、不写文件；这些逻辑位于 cli.py。
"""

from __future__ import annotations

from graph_visualizer.cli import main
from graph_visualizer.package import LoadedPackage, load_model_package
from graph_visualizer.svg import render_svg
from graph_visualizer.view_model import ViewModel, build_view_model

__all__ = [
    "LoadedPackage",
    "ViewModel",
    "build_view_model",
    "load_model_package",
    "main",
    "render_svg",
]
