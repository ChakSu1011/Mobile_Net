"""模块：显存排布可视化 CLI 启动器。

职责：
  配置导入路径并调用 graph_memory.cli。

边界：
  不实现参数解析或绘图逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for _path in (_REPOSITORY_ROOT, _TOOLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from graph_memory.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
