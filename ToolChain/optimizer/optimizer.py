"""模块：Optimizer 编排与结果封装。

职责：
  在 ``ModelGraph`` 深拷贝上依次执行 Precision、Shape、Memory 三个 Pass，
  汇总报告并返回 ``OptimizeResult``，供 Serializer 使用。

边界：
  - 不修改调用方传入的模型图；
  - 不执行真实推理，不分配真实显存；
  - 任一 Pass 失败时不返回半成品。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ToolChain.common.jsonio import write_json
from ToolChain.common.model_ir import MemoryPlan, ModelGraph
from ToolChain.optimizer.config import OptimizeConfig
from ToolChain.optimizer.memory import MemoryPass
from ToolChain.optimizer.precision import PrecisionPass
from ToolChain.optimizer.report import build_report
from ToolChain.optimizer.shape import ShapePass


@dataclass(frozen=True)
class OptimizeResult:
    """优化结果。

    属性：
      model_graph：优化后的模型图（深拷贝）；
      config：本次优化配置；
      memory_plan：激活显存计划；
      report：结构化优化报告。
    """

    model_graph: ModelGraph
    config: OptimizeConfig
    memory_plan: MemoryPlan
    report: dict[str, Any]


class Optimizer:
    """Build 阶段优化器。"""

    def __init__(self, config: OptimizeConfig | None = None) -> None:
        """保存配置。

        参数：
          config：优化配置；省略时使用默认值。
        """
        self._config = config if config is not None else OptimizeConfig()

    @property
    def config(self) -> OptimizeConfig:
        """返回当前配置。"""
        return self._config

    def run(self, model_graph: ModelGraph) -> OptimizeResult:
        """执行完整优化流程。

        参数：
          model_graph：Converter 输出的模型图；必须已通过 ``validate()``。

        返回：
          优化结果；输入对象不被修改。

        异常：
          BuildError：配置非法、形状规则缺失、精度/显存不变量失败。
        """
        model_graph.validate()
        optimized = model_graph.clone()
        precision_report = PrecisionPass(self._config).run(optimized)
        shape_report = ShapePass(self._config).run(optimized)
        memory_plan = MemoryPass(self._config).run(optimized)
        optimized.metadata["optimization"] = {
            "config": self._config.to_dict(),
            "passes": ["precision", "shape", "memory"],
        }
        report = build_report(
            optimized,
            self._config,
            precision_report,
            shape_report,
            memory_plan,
        )
        return OptimizeResult(
            model_graph=optimized,
            config=self._config,
            memory_plan=memory_plan,
            report=report,
        )


def write_optimization_outputs(
    result: OptimizeResult,
    output_dir: Path,
) -> dict[str, Path]:
    """把优化后的模型图与报告写入目录。

    参数：
      result：优化结果；
      output_dir：输出目录；不存在时创建。

    返回：
      键为 ``model_graph``/``optimize_report`` 的路径字典。

    异常：
      OSError：目录创建或文件写入失败。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model_graph.json"
    report_path = output_dir / "optimize_report.json"
    result.model_graph.save_json(model_path)
    write_json(report_path, result.report)
    return {
        "model_graph": model_path,
        "optimize_report": report_path,
    }
