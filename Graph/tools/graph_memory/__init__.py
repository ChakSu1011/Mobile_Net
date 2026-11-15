"""模块：Graph 显存排布可视化工具公共接口。

职责：
  导出模型包读取、内存视图构建与 SVG 渲染入口。

边界：
  不解析命令行、不写文件；这些逻辑位于 cli.py。
"""

from __future__ import annotations

from graph_memory.cli import main
from graph_memory.package import LoadedPackage, load_model_package
from graph_memory.svg import render_memory_svg
from graph_memory.view_model import MemoryViewModel, build_memory_view

__all__ = [
    "LoadedPackage",
    "MemoryViewModel",
    "build_memory_view",
    "load_model_package",
    "main",
    "render_memory_svg",
]
