"""模块：Serializer 单元与集成测试。

职责：
  覆盖固定头解析、manifest/权重哈希校验、截断与版本错误、小型图往返、
  合成权重打包、确定性与 Qwen3 metadata-only 模型包。

边界：
  依赖 Graph_Fixed 重建图；不依赖真实权重、GPU、torch 或 transformers。
"""

from __future__ import annotations

import shutil
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from Build.common.errors import BuildError
from Build.common.hashing import sha256_file
from Build.converter.converter import convert
from Build.optimizer import OptimizeConfig, Optimizer
from Build.serializer import (
    load_package,
    package_summary,
    rebuild_graph,
    rebuild_graph_from_model,
    serialize_package,
)
from Build.serializer.format import (
    HEADER_SIZE,
    unpack_header,
)
from Build.testing import build_tiny_model, write_safetensors

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_QWEN3_CONFIG = (
    _REPOSITORY_ROOT / "Models" / "Qwen3" / "config.json")
_SMALL_CONFIG = OptimizeConfig(max_num_tokens=4)


def _optimize_tiny():
    """返回小型图优化结果。"""
    return Optimizer(_SMALL_CONFIG).run(
        build_tiny_model("float16"))


def _payload(size: int, seed: int) -> bytes:
    """生成指定长度的确定性字节串。

    参数：
      size：字节数；
      seed：起始值。

    返回：
      字节串。
    """
    return bytes((seed + index) % 256 for index in range(size))


def _write_tiny_weights(path: Path) -> dict[str, bytes]:
    """写出与小型图匹配的合成 safetensors。

    参数：
      path：输出文件路径。

    返回：
      张量名到原始字节的映射。
    """
    embedding = _payload(100 * 16 * 2, 1)
    head = _payload(32 * 16 * 2, 7)
    write_safetensors(path, {
        "model.embed_tokens.weight": (
            "F16", (100, 16), embedding),
        "lm_head.weight": ("F16", (32, 16), head),
    })
    return {
        "embedding.weight": embedding,
        "head.weight": head,
    }


class SerializerRoundtripTest(unittest.TestCase):
    """metadata-only 模型包测试。"""

    def setUp(self) -> None:
        """创建临时目录并序列化小型图。"""
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.result = _optimize_tiny()
        self.serialized = serialize_package(
            self.result, self.directory / "package")

    def tearDown(self) -> None:
        """删除临时目录。"""
        self._temporary.cleanup()

    def test_header(self) -> None:
        """UT-SER-001：固定头字段与 manifest 哈希正确。"""
        data = self.serialized.model_file.read_bytes()
        header = unpack_header(data)
        self.assertEqual(header.version, 1)
        self.assertFalse(header.has_weights)
        self.assertEqual(header.manifest_offset, HEADER_SIZE)
        self.assertEqual(header.weights_length, 0)
        manifest = data[
            header.manifest_offset:
            header.manifest_offset + header.manifest_length]
        self.assertEqual(
            header.manifest_sha256.hex(),
            self.serialized.manifest_sha256,
        )
        self.assertGreater(len(manifest), 0)
        self.assertEqual(
            header.manifest_length, len(manifest))

    def test_metadata_only(self) -> None:
        """UT-SER-006：无权重包可加载且不生成 weights.bin。"""
        self.assertIsNone(self.serialized.weights_file)
        self.assertFalse(
            (self.directory / "package" / "weights.bin").exists())
        summary = package_summary(
            load_package(self.serialized.package_dir))
        self.assertEqual(summary["weights_available"], 0)
        self.assertEqual(summary["weights_total_bytes"], 0)
        self.assertEqual(summary["model"], "tiny")

    def test_graph_roundtrip(self) -> None:
        """UT-SER-007：加载并重建图后 IR 与优化图一致。"""
        package = load_package(
            self.serialized.package_dir, verify=True)
        graph = rebuild_graph(package, verify_ir=True)
        expected = rebuild_graph_from_model(
            self.result.model_graph)
        self.assertEqual(graph.dump_ir(), expected.dump_ir())
        self.assertEqual(
            package.manifest["graph"]["ir_sha256"],
            self.serialized.graph_ir_sha256,
        )

    def test_determinism(self) -> None:
        """UT-SER-008：两次序列化 model.mgf 字节一致。"""
        other = serialize_package(
            self.result, self.directory / "package2")
        self.assertEqual(
            sha256_file(self.serialized.model_file),
            sha256_file(other.model_file),
        )

    def test_manifest_tamper(self) -> None:
        """UT-SER-002：篡改 manifest 后加载报错。"""
        model_file = self.serialized.model_file
        data = bytearray(model_file.read_bytes())
        data[HEADER_SIZE + 10] ^= 0xFF
        model_file.write_bytes(bytes(data))
        with self.assertRaises(BuildError) as context:
            load_package(self.serialized.package_dir)
        self.assertIn("manifest SHA-256", str(context.exception))

    def test_truncated_header(self) -> None:
        """UT-SER-004：头部截断时报错。"""
        self.serialized.model_file.write_bytes(b"MNET")
        with self.assertRaises(BuildError) as context:
            load_package(self.serialized.package_dir)
        self.assertIn("too short", str(context.exception))

    def test_truncated_manifest(self) -> None:
        """UT-SER-004：manifest 截断时报错。"""
        data = self.serialized.model_file.read_bytes()
        self.serialized.model_file.write_bytes(
            data[:HEADER_SIZE + 10])
        with self.assertRaises(BuildError) as context:
            load_package(self.serialized.package_dir)
        self.assertIn("beyond file", str(context.exception))

    def test_version_mismatch(self) -> None:
        """UT-SER-005：容器版本不支持时报错。"""
        data = bytearray(self.serialized.model_file.read_bytes())
        struct.pack_into("<I", data, 8, 99)
        self.serialized.model_file.write_bytes(bytes(data))
        with self.assertRaises(BuildError) as context:
            load_package(self.serialized.package_dir)
        self.assertIn("version", str(context.exception))

    def test_ir_verification_failure(self) -> None:
        """UT-SER-010：manifest IR 哈希被篡改时重建报错。"""
        package = load_package(self.serialized.package_dir)
        manifest = dict(package.manifest)
        graph = dict(manifest["graph"])
        graph["ir_sha256"] = "0" * 64
        manifest["graph"] = graph
        tampered = replace(package, manifest=manifest)
        with self.assertRaises(BuildError) as context:
            rebuild_graph(tampered, verify_ir=True)
        self.assertIn("IR SHA-256", str(context.exception))

    def test_unoptimized_rejected(self) -> None:
        """UT-SER-011：未优化图因形状未解析被拒绝。"""
        class _Result:
            model_graph = build_tiny_model("float16")
            config = _SMALL_CONFIG
            memory_plan = _optimize_tiny().memory_plan

        with self.assertRaises(BuildError) as context:
            serialize_package(
                _Result(), self.directory / "raw")
        self.assertIn("not resolved", str(context.exception))


class SerializerWeightsTest(unittest.TestCase):
    """合成权重包测试。"""

    def setUp(self) -> None:
        """创建带权重的模型包。"""
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.weights_path = self.directory / "weights.safetensors"
        self.payloads = _write_tiny_weights(self.weights_path)
        self.result = _optimize_tiny()
        self.serialized = serialize_package(
            self.result,
            self.directory / "package",
            weights_path=self.weights_path,
        )

    def tearDown(self) -> None:
        """删除临时目录。"""
        self._temporary.cleanup()

    def test_synthetic_weights(self) -> None:
        """UT-SER-009：权重字节一致且偏移对齐。"""
        self.assertIsNotNone(self.serialized.weights_file)
        package = load_package(self.serialized.package_dir)
        self.assertTrue(package.manifest["weights"]["total_bytes"] > 0)
        data = self.serialized.weights_file.read_bytes()
        for entry in package.manifest["weights"]["entries"]:
            self.assertTrue(entry["available"], entry["name"])
            self.assertEqual(entry["offset"] % 256, 0, entry["name"])
            begin = entry["offset"]
            end = begin + entry["byte_size"]
            self.assertEqual(
                data[begin:end],
                self.payloads[entry["name"]],
                entry["name"],
            )
        summary = package_summary(package)
        self.assertEqual(summary["weights_available"], 2)

    def test_weight_tamper(self) -> None:
        """UT-SER-003：篡改 weights.bin 后加载报错。"""
        data = bytearray(self.serialized.weights_file.read_bytes())
        data[0] ^= 0xFF
        self.serialized.weights_file.write_bytes(bytes(data))
        with self.assertRaises(BuildError) as context:
            load_package(self.serialized.package_dir)
        self.assertIn("weights SHA-256", str(context.exception))


class Qwen3PackageTest(unittest.TestCase):
    """Qwen3 metadata-only 模型包集成测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        """生成一次 Qwen3 模型包并缓存。"""
        cls.directory = Path(tempfile.mkdtemp(prefix="qwen3_pkg_"))
        conversion = convert("qwen3", _QWEN3_CONFIG)
        result = Optimizer(OptimizeConfig()).run(
            conversion.model_graph)
        cls.serialized = serialize_package(
            result, cls.directory / "package")

    @classmethod
    def tearDownClass(cls) -> None:
        """删除临时目录。"""
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_qwen3_package(self) -> None:
        """UT-SER-012：Qwen3 包可加载并重建图。"""
        package = load_package(
            self.serialized.package_dir, verify=True)
        graph = rebuild_graph(package, verify_ir=True)
        summary = package_summary(package)
        self.assertEqual(summary["tensors"], 735)
        self.assertEqual(summary["ops"], 423)
        self.assertEqual(summary["constants"], 310)
        self.assertEqual(summary["weights_available"], 0)
        ir_text = graph.dump_ir()
        self.assertEqual(
            package.manifest["graph"]["ir_line_count"],
            len(ir_text.splitlines()),
        )
        self.assertEqual(
            package.manifest["graph"]["ir_sha256"],
            self.serialized.graph_ir_sha256,
        )
        self.assertEqual(
            sorted(path.name for path in
                   self.serialized.package_dir.iterdir()),
            ["compile_report.json", "model.mgf"],
        )


if __name__ == "__main__":
    unittest.main()
