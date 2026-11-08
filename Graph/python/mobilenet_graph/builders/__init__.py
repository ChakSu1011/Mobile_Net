"""模块：Python 侧模型融合算子图构建器集合。

职责：
  按模型族组织 Python 端融合算子图构建器；当前包含 Qwen3-0.6B。

边界：
  构建器只调用 Graph IR 的 Python 绑定，不定义第二套图结构；
  不加载真实权重，不执行推理。

依赖：
  ``mobilenet_graph`` 包与各模型构建器。
"""

from __future__ import annotations

from . import qwen3

__all__ = ["qwen3"]
