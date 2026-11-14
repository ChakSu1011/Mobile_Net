"""模块：MobileNet 离线工具链（Build）。

职责：
  组织 converter、optimizer、serializer 三个离线阶段模块，并提供
  Qwen3-0.6B 端到端构建入口。

边界：
  只负责主机侧模型转换、优化与序列化，不执行设备端推理。
"""
