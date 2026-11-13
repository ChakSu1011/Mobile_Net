"""模块：Build 阶段统一异常。

职责：
  定义 ``BuildError``，让 Converter、Optimizer、Serializer 的错误信息
  统一携带阶段名、对象名与原因，便于命令行与测试定位。

边界：
  只承载错误信息，不负责日志输出、错误码映射或重试策略。
"""

from __future__ import annotations


class BuildError(Exception):
    """Build 阶段可诊断错误。

    功能：
      表示模型转换、优化、序列化或加载过程中的可预期失败。

    边界：
      不捕获第三方异常；调用方负责把底层异常转换为 ``BuildError``，
      或在需要保留原始异常时使用 ``raise ... from``。

    属性：
      stage：失败阶段，如 ``converter``、``optimizer``、``serializer``；
      target：失败对象，如 op 名、张量名或文件路径；
      message：人类可读的失败原因。
    """

    def __init__(self, stage: str, message: str, *, target: str = "") -> None:
        """构造错误对象。

        参数：
          stage：失败阶段，不能为空字符串；
          message：失败原因，不能为空字符串；
          target：可选失败对象名称或路径。

        异常：
          ValueError：stage 或 message 为空时抛出。
        """
        if not stage:
            raise ValueError("stage must not be empty")
        if not message:
            raise ValueError("message must not be empty")
        self.stage = stage
        self.target = target
        self.message = message
        detail = f"[{stage}]"
        if target:
            detail += f" {target}:"
        detail += f" {message}"
        super().__init__(detail)

    def __str__(self) -> str:
        """返回带阶段与对象的错误文本。"""
        return self.args[0]
