"""模块：Qwen3-0.6B Graph_Fixed 组图测试。

职责：
  验证 Qwen3-0.6B 静态图的配置解析、图规模、拓扑顺序、权重形状、
  tied/untied LM Head 行为和错误路径；可选使用 Transformers 的
  Qwen3ForCausalLM meta 模型做权重形状对照。

边界：
  - 不加载真实权重、不执行 kernel、不验证数值；
  - 图测试只依赖 Graph_Fixed 的 Python 绑定；
  - Transformers/torch 对照测试在依赖缺失时自动 skip。

依赖：
  ``mobilenet_runtime_graph``、``qwen3_config``、
  ``qwen3_graph_builder``；可选 torch、transformers。
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(
    0, str(REPOSITORY_ROOT / "Graph_Fixed" / "python"))

from qwen3_config import Qwen3Config  # noqa: E402
from qwen3_graph_builder import (  # noqa: E402
    Qwen3GraphBuilder,
    graph_summary,
)

CONFIG_PATH = MODEL_DIR / "config.json"
EXPECTED_OPS = 426
EXPECTED_CONSTANTS = 310
EXPECTED_INPUTS = 2
EXPECTED_OUTPUTS = 1
EXPECTED_PARAMS = 596_049_920
EXPECTED_BYTES = 1_192_099_840


class Qwen3GraphTest(unittest.TestCase):
    """验证 Qwen3-0.6B 静态图的结构、权重和错误路径。"""

    @classmethod
    def setUpClass(cls) -> None:
        """加载配置并构建一次默认图，供各用例复用。"""
        cls.config = Qwen3Config.from_json(CONFIG_PATH)
        cls.builder = Qwen3GraphBuilder(cls.config)
        cls.graph = cls.builder.build()
        cls.ir = cls.graph.dump_ir()
        cls.lines = cls.ir.splitlines()

    def _op_line(self, name: str) -> str:
        """返回指定 op 的 dump 行。

        参数：
          name：op 名称。

        返回：
          包含该 op 的完整行。

        异常：
          AssertionError：op 不存在时由 ``self.fail`` 抛出。
        """
        marker = f"] {name} "
        for line in self.lines:
            if line.startswith("  [") and marker in line:
                return line
        self.fail(f"op not found: {name}")

    def _op_names(self) -> list[str]:
        """按拓扑序返回全部 op 名称。"""
        names = []
        for line in self.lines:
            if line.startswith("  ["):
                names.append(line.split()[1])
        return names

    def test_config_defaults(self) -> None:
        """配置字段与 Qwen3-0.6B 一致，派生宽度正确。"""
        self.assertEqual(self.config.model_type, "qwen3")
        self.assertEqual(self.config.vocab_size, 151936)
        self.assertEqual(self.config.hidden_size, 1024)
        self.assertEqual(self.config.intermediate_size, 3072)
        self.assertEqual(self.config.num_hidden_layers, 28)
        self.assertEqual(self.config.num_attention_heads, 16)
        self.assertEqual(self.config.num_key_value_heads, 8)
        self.assertEqual(self.config.head_dim, 128)
        self.assertEqual(self.config.q_width, 2048)
        self.assertEqual(self.config.kv_width, 1024)
        self.assertEqual(self.config.gqa_ratio, 2)
        self.assertTrue(self.config.tie_word_embeddings)
        self.assertFalse(self.config.attention_bias)
        self.assertFalse(self.config.use_sliding_window)
        self.assertEqual(self.config.dtype, "bfloat16")

    def test_graph_counts(self) -> None:
        """图名称和 inputs/constants/ops/outputs 计数符合设计。"""
        self.assertEqual(self.lines[0], "graph qwen3_0_6b")
        self.assertIn(f"inputs {EXPECTED_INPUTS}", self.lines)
        self.assertIn(f"constants {EXPECTED_CONSTANTS}", self.lines)
        self.assertIn(f"ops {EXPECTED_OPS}", self.lines)
        self.assertIn(f"outputs {EXPECTED_OUTPUTS}", self.lines)
        summary = graph_summary(self.graph)
        self.assertIn("graph qwen3_0_6b", summary)
        self.assertIn(f"ops {EXPECTED_OPS}", summary)

    def test_constants_and_tied_weight(self) -> None:
        """默认图登记 310 个 constant，LM Head 复用 embedding 权重。"""
        self.assertEqual(
            len(self.builder.graph_constant_shapes),
            EXPECTED_CONSTANTS,
        )
        self.assertIn("lm_head.weight", self.builder.weight_shapes)
        self.assertEqual(
            self.builder.weight_shapes["lm_head.weight"],
            (151936, 1024),
        )
        self.assertNotIn(
            "lm_head.weight", self.builder.graph_constant_shapes)
        self.assertIn(
            "embedding.weight storage(precision=bfloat16,"
            "layout=row_major,device=host,shape=[151936,1024],"
            "offset=0,bytes=311164928,data=unset,reusable=false)",
            self.ir,
        )
        self.assertNotIn("lm_head.weight storage(", self.ir)

    def test_weight_shapes_and_parameter_count(self) -> None:
        """权重形状、总参数和 bfloat16 字节数与 Qwen3-0.6B 一致。"""
        self.assertEqual(
            self.builder.graph_constant_shapes[
                "layer.0.q_proj.weight"],
            (2048, 1024),
        )
        self.assertEqual(
            self.builder.graph_constant_shapes[
                "layer.0.k_proj.weight"],
            (1024, 1024),
        )
        self.assertEqual(
            self.builder.graph_constant_shapes[
                "layer.0.down_proj.weight"],
            (1024, 3072),
        )
        total_params = sum(
            math.prod(shape)
            for shape in self.builder.graph_constant_shapes.values()
        )
        self.assertEqual(total_params, EXPECTED_PARAMS)
        self.assertEqual(total_params * 2, EXPECTED_BYTES)

    def test_layer_zero_topology(self) -> None:
        """layer.0 的 15 个 op 顺序和关键连接符合源码结构。"""
        names = self._op_names()
        layer_prefix = "layer.0."
        layer_ops = [
            name for name in names if name.startswith(layer_prefix)]
        self.assertEqual(len(layer_ops), 15)
        self.assertEqual(len(set(layer_ops)), 15)

        def index(name: str) -> int:
            """返回 op 在拓扑序中的位置。"""
            return names.index(name)

        dependencies = [
            ("layer.0.attn_norm", "layer.0.q_proj"),
            ("layer.0.attn_norm", "layer.0.k_proj"),
            ("layer.0.attn_norm", "layer.0.v_proj"),
            ("layer.0.q_proj", "layer.0.q_norm"),
            ("layer.0.k_proj", "layer.0.k_norm"),
            ("layer.0.q_norm", "layer.0.q_rope"),
            ("layer.0.k_norm", "layer.0.k_rope"),
            ("layer.0.q_rope", "layer.0.attn"),
            ("layer.0.k_rope", "layer.0.attn"),
            ("layer.0.v_proj", "layer.0.attn"),
            ("layer.0.attn", "layer.0.o_proj"),
            ("layer.0.o_proj", "layer.0.attn_residual"),
            ("layer.0.attn_residual", "layer.0.mlp_norm"),
            ("layer.0.mlp_norm", "layer.0.mlp_gate_up"),
            ("layer.0.mlp_gate_up", "layer.0.mlp_down"),
            ("layer.0.mlp_down", "layer.0.mlp_residual"),
            ("layer.0.mlp_residual", "layer.1.attn_norm"),
        ]
        for producer, consumer in dependencies:
            self.assertLess(index(producer), index(consumer))
        layer_zero_positions = [index(name) for name in layer_ops]
        self.assertEqual(
            sorted(layer_zero_positions), layer_zero_positions)

        self.assertIn(
            "src=[embedding.out,layer.0.attn_norm.weight]",
            self._op_line("layer.0.attn_norm"),
        )
        self.assertIn(
            "src=[layer.0.attn_norm.out,layer.0.q_proj.weight]",
            self._op_line("layer.0.q_proj"),
        )
        self.assertIn(
            "src=[layer.0.q_proj.out,layer.0.q_norm.weight]",
            self._op_line("layer.0.q_norm"),
        )
        self.assertIn(
            "src=[layer.0.q_norm.out,input.position_ids]",
            self._op_line("layer.0.q_rope"),
        )
        self.assertIn(
            "src=[layer.0.q_rope.out,layer.0.k_rope.out,"
            "layer.0.v_proj.out]",
            self._op_line("layer.0.attn"),
        )
        self.assertIn(
            "src=[layer.0.attn.out,layer.0.o_proj.weight]",
            self._op_line("layer.0.o_proj"),
        )
        self.assertIn(
            "src=[layer.0.o_proj.out,embedding.out]",
            self._op_line("layer.0.attn_residual"),
        )
        self.assertIn(
            "src=[layer.0.mlp_norm.out,layer.0.gate_proj.weight,"
            "layer.0.up_proj.weight]",
            self._op_line("layer.0.mlp_gate_up"),
        )
        self.assertIn(
            "src=[layer.0.mlp_gate_up.out,"
            "layer.0.down_proj.weight]",
            self._op_line("layer.0.mlp_down"),
        )
        self.assertIn(
            "src=[layer.0.mlp_down.out,"
            "layer.0.attn_residual.out]",
            self._op_line("layer.0.mlp_residual"),
        )

    def test_final_lm_head_and_output(self) -> None:
        """最后一层、final_norm、lm_head 和 output 的连接正确。"""
        self.assertIn(
            "src=[layer.27.mlp_residual.out,final_norm.weight]",
            self._op_line("final_norm"),
        )
        self.assertIn(
            "src=[final_norm.out,embedding.weight]",
            self._op_line("lm_head"),
        )
        self.assertIn(
            "output.logits <- [lm_head.out]", self.ir)
        self.assertEqual(self._op_names()[-1], "output.logits")

    def test_untied_lm_head(self) -> None:
        """tie_word_embeddings=false 时新增独立 lm_head.weight。"""
        config = replace(
            self.config, tie_word_embeddings=False)
        builder = Qwen3GraphBuilder(config)
        graph = builder.build()
        ir = graph.dump_ir()
        self.assertIn("constants 311", ir.splitlines())
        self.assertIn("lm_head.weight", builder.graph_constant_shapes)
        self.assertIn(
            "lm_head.weight storage(precision=bfloat16,"
            "layout=row_major,device=host,shape=[151936,1024],"
            "offset=0,bytes=311164928,data=unset,reusable=false)",
            ir,
        )
        self.assertIn(
            "src=[final_norm.out,lm_head.weight]",
            self._op_line_from_ir(ir, "lm_head"),
        )

    def _op_line_from_ir(self, ir: str, name: str) -> str:
        """从给定 IR 文本中查找指定 op 行。

        参数：
          ir：dump_ir() 输出；
          name：op 名称。

        返回：
          匹配的 op 行。

        异常：
          AssertionError：op 不存在时由 ``self.fail`` 抛出。
        """
        marker = f"] {name} "
        for line in ir.splitlines():
            if line.startswith("  [") and marker in line:
                return line
        self.fail(f"op not found: {name}")

    def test_invalid_configs(self) -> None:
        """非法配置在 builder 初始化阶段抛出 ValueError。"""
        invalid_configs = [
            replace(self.config, hidden_act="relu"),
            replace(self.config, attention_bias=True),
            replace(self.config, dtype="int8"),
            replace(self.config, use_sliding_window=True),
            replace(self.config, num_key_value_heads=7),
            replace(self.config, rope_type="linear"),
            replace(self.config, vocab_size=0),
        ]
        for config in invalid_configs:
            with self.assertRaises(ValueError):
                Qwen3GraphBuilder(config)

    def test_dump_is_stable(self) -> None:
        """重复调用 dump_ir() 输出完全一致。"""
        self.assertEqual(self.graph.dump_ir(), self.ir)
        self.assertEqual(len(self._op_names()), EXPECTED_OPS)

    def test_all_layers_have_fifteen_ops(self) -> None:
        """28 个 Decoder Layer 各包含 15 个唯一 op。"""
        names = self._op_names()
        for layer in range(self.config.num_hidden_layers):
            prefix = f"layer.{layer}."
            layer_ops = [
                name for name in names if name.startswith(prefix)]
            self.assertEqual(len(layer_ops), 15, prefix)
            self.assertEqual(len(set(layer_ops)), 15, prefix)

    def test_config_file_matches_transformers(self) -> None:
        """config.json 可被 Transformers Qwen3Config 直接加载。"""
        try:
            from transformers import Qwen3Config as TransformersConfig
        except ImportError:
            self.skipTest("transformers not installed")

        data = json.loads(
            CONFIG_PATH.read_text(encoding="utf-8"))
        config = TransformersConfig(**data)
        self.assertEqual(config.model_type, "qwen3")
        self.assertEqual(config.num_hidden_layers, 28)
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.intermediate_size, 3072)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.rope_parameters["rope_theta"], 1e6)
        self.assertEqual(
            config.rope_parameters["rope_type"], "default")
        self.assertTrue(config.tie_word_embeddings)

    def test_transformers_weight_shapes(self) -> None:
        """与 Transformers Qwen3ForCausalLM meta 模型的权重形状一致。"""
        try:
            import torch
            from transformers import Qwen3Config as TransformersConfig
            from transformers import Qwen3ForCausalLM
        except ImportError:
            self.skipTest("torch/transformers not installed")

        transformers_config = TransformersConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            hidden_act=self.config.hidden_act,
            max_position_embeddings=
                self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            tie_word_embeddings=self.config.tie_word_embeddings,
            attention_bias=self.config.attention_bias,
            use_sliding_window=self.config.use_sliding_window,
            sliding_window=self.config.sliding_window,
            rope_parameters={
                "rope_type": self.config.rope_type,
                "rope_theta": self.config.rope_theta,
            },
            dtype=self.config.dtype,
        )
        with torch.device("meta"):
            model = Qwen3ForCausalLM(transformers_config)
        shapes = {
            name: tuple(parameter.shape)
            for name, parameter in model.named_parameters(
                remove_duplicate=False)
        }

        expected_names = {
            "embedding.weight": "model.embed_tokens.weight",
            "final_norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for layer in range(self.config.num_hidden_layers):
            prefix = f"layer.{layer}"
            hf_prefix = f"model.layers.{layer}"
            expected_names.update({
                f"{prefix}.attn_norm.weight":
                    f"{hf_prefix}.input_layernorm.weight",
                f"{prefix}.q_proj.weight":
                    f"{hf_prefix}.self_attn.q_proj.weight",
                f"{prefix}.k_proj.weight":
                    f"{hf_prefix}.self_attn.k_proj.weight",
                f"{prefix}.v_proj.weight":
                    f"{hf_prefix}.self_attn.v_proj.weight",
                f"{prefix}.o_proj.weight":
                    f"{hf_prefix}.self_attn.o_proj.weight",
                f"{prefix}.q_norm.weight":
                    f"{hf_prefix}.self_attn.q_norm.weight",
                f"{prefix}.k_norm.weight":
                    f"{hf_prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp_norm.weight":
                    f"{hf_prefix}.post_attention_layernorm.weight",
                f"{prefix}.gate_proj.weight":
                    f"{hf_prefix}.mlp.gate_proj.weight",
                f"{prefix}.up_proj.weight":
                    f"{hf_prefix}.mlp.up_proj.weight",
                f"{prefix}.down_proj.weight":
                    f"{hf_prefix}.mlp.down_proj.weight",
            })
        for graph_name, hf_name in expected_names.items():
            self.assertEqual(
                self.builder.weight_shapes[graph_name],
                shapes[hf_name],
                graph_name,
            )
        total_params = sum(
            parameter.numel() for parameter in model.parameters())
        self.assertEqual(total_params, EXPECTED_PARAMS)
        self.assertIs(
            model.lm_head.weight, model.model.embed_tokens.weight)


if __name__ == "__main__":
    unittest.main()
