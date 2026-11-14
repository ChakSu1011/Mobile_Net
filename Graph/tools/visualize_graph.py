"""模块：Graph 静态图可视化 CLI 启动器。

职责：
  配置仓库根目录与工具目录的导入路径，然后调用 graph_visualizer.cli。

边界：
  不实现参数解析或绘图逻辑；只负责 Python import path 和入口转发。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from graph_visualizer.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
