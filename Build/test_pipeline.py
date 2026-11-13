"""模块：Build 端到端流水线测试。

职责：
  验证 convert→optimize→serialize→load→rebuild 串联、产物布局、
  包字节确定性、显存复用开关与 CLI 入口。

边界：
  依赖 Graph_Fixed 与 Qwen3 config.json；不依赖真实权重。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from Build.common.hashing import sha256_file
from Build.optimizer import OptimizeConfig
from Build.pipeline import main, run_pipeline

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_QWEN3_CONFIG = (
    _REPOSITORY_ROOT / "Models" / "Qwen3" / "config.json")


class PipelineTest(unittest.TestCase):
    """端到端流水线测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        """执行一次完整 Qwen3 构建。"""
        cls.directory = Path(tempfile.mkdtemp(prefix="pipeline_"))
        cls.result = run_pipeline(
            config_path=_QWEN3_CONFIG,
            output_dir=cls.directory / "package",
            work_dir=cls.directory / "work",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """删除临时目录。"""
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_package_counts(self) -> None:
        """Qwen3 包计数、IR 哈希与重建一致。"""
        summary = self.result.package.manifest["statistics"]
        self.assertEqual(summary["tensors"], 735)
        self.assertEqual(summary["ops"], 423)
        self.assertEqual(summary["constants"], 310)
        self.assertEqual(summary["activation_tensors"], 423)
        self.assertEqual(
            self.result.serialization.graph_ir_sha256,
            self.result.package.manifest["graph"]["ir_sha256"],
        )
        self.assertEqual(
            len(self.result.graph.dump_ir().splitlines()),
            self.result.package.manifest["graph"]["ir_line_count"],
        )

    def test_package_layout(self) -> None:
        """模型包目录只包含格式文件。"""
        names = sorted(
            path.name
            for path in self.result.serialization.package_dir.iterdir()
        )
        self.assertEqual(
            names, ["compile_report.json", "model.mgf"])
        self.assertIsNone(self.result.serialization.weights_file)

    def test_work_artifacts(self) -> None:
        """中间产物完整。"""
        work = self.result.work_dir
        self.assertTrue(
            (work / "converter" / "model_graph.json").is_file())
        self.assertTrue(
            (work / "converter" / "conversion_report.json").is_file())
        self.assertTrue(
            (work / "optimizer" / "model_graph.json").is_file())
        self.assertTrue(
            (work / "optimizer" / "optimize_report.json").is_file())

    def test_determinism(self) -> None:
        """相同输入生成相同 model.mgf。"""
        other = run_pipeline(
            config_path=_QWEN3_CONFIG,
            output_dir=self.directory / "package2",
            work_dir=self.directory / "work2",
        )
        self.assertEqual(
            sha256_file(self.result.serialization.model_file),
            sha256_file(other.serialization.model_file),
        )

    def test_disable_reuse(self) -> None:
        """关闭复用后 arena 变大。"""
        other = run_pipeline(
            config_path=_QWEN3_CONFIG,
            output_dir=self.directory / "package3",
            work_dir=self.directory / "work3",
            optimize_config=OptimizeConfig(enable_reuse=False),
        )
        self.assertGreater(
            other.optimization.memory_plan.arena_bytes,
            self.result.optimization.memory_plan.arena_bytes,
        )

    def test_cli(self) -> None:
        """CLI 入口返回成功。"""
        code = main([
            "--config", str(_QWEN3_CONFIG),
            "--output", str(self.directory / "package4"),
            "--work-dir", str(self.directory / "work4"),
        ])
        self.assertEqual(code, 0)
        self.assertTrue(
            (self.directory / "package4" / "model.mgf").is_file())


if __name__ == "__main__":
    unittest.main()
