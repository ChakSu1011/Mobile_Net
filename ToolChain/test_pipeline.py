"""模块：Build 端到端流水线测试。

职责：
  验证 convert→optimize→serialize→load→rebuild 串联、产物布局、
  包字节确定性、显存复用开关与 CLI 入口。

边界：
  依赖 Graph 与 Qwen3 config.json；不依赖真实权重。
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ToolChain.common.hashing import sha256_file
from ToolChain.optimizer import OptimizeConfig
from ToolChain.pipeline import main, run_pipeline

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
        self.assertEqual(summary["tensors"], 623)
        self.assertEqual(summary["ops"], 311)
        self.assertEqual(summary["constants"], 310)
        self.assertEqual(summary["activation_tensors"], 311)
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

    def _write_inference_config(self, root: Path) -> Path:
        """写出带输入配置的临时模型目录。

        参数：
          root：临时模型目录。

        返回：
          config.json 路径。
        """
        data = json.loads(
            _QWEN3_CONFIG.read_text(encoding="utf-8"))
        config_path = root / "config.json"
        config_path.write_text(json.dumps(data), encoding="utf-8")
        input_dir = root / "input"
        input_dir.mkdir()
        (input_dir / "inference_config.json").write_text(
            json.dumps({
                "max_num_tokens": 8,
                "max_batch_size": 2,
                "repetition_penalty": 1.0,
                "seed": 0,
            }),
            encoding="utf-8",
        )
        return config_path

    def test_inference_config_drives_planning(self) -> None:
        """输入配置的 runtime 作为优化默认值。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_inference_config(root)
            result = run_pipeline(
                config_path=config_path,
                output_dir=root / "package",
                work_dir=root / "work",
            )
        config = result.optimization.config
        self.assertEqual(config.max_num_tokens, 8)
        self.assertEqual(config.context_length, 40960)
        self.assertEqual(config.batch_size, 2)

    def test_optimize_overrides(self) -> None:
        """显式覆盖字段优先，其余字段仍取输入配置。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_inference_config(root)
            result = run_pipeline(
                config_path=config_path,
                output_dir=root / "package",
                work_dir=root / "work",
                optimize_overrides={"max_num_tokens": 4},
            )
        config = result.optimization.config
        self.assertEqual(config.max_num_tokens, 4)
        self.assertEqual(config.context_length, 40960)
        self.assertEqual(config.batch_size, 2)


if __name__ == "__main__":
    unittest.main()
