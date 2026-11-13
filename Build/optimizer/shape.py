"""模块：Shape Infer Pass。

职责：
  把符号维 ``B``/``S`` 绑定为规划值，按算子规则推导全部激活张量形状，
  计算字节数，并校验算子形状契约。

边界：
  - 不执行真实计算，不做数值校验；
  - 只支持首期 Qwen3 使用的算子集合；
  - 不修改 Graph对象，只修改 ``ModelGraph``。
"""

from __future__ import annotations

import math
from typing import Any

from Build.common.dtypes import element_bytes
from Build.common.errors import BuildError
from Build.common.model_ir import Dim, ModelGraph, OpMeta
from Build.optimizer.config import OptimizeConfig


class ShapePass:
    """静态形状推导。"""

    def __init__(self, config: OptimizeConfig) -> None:
        """保存规划边界。

        参数：
          config：优化配置；构造时已校验。
        """
        self._config = config
        self._symbols = {
            "B": config.batch_size,
            "S": config.max_num_tokens,
        }
        self._shapes: dict[str, tuple[int, ...]] = {}

    def run(self, model_graph: ModelGraph) -> dict[str, Any]:
        """执行形状推导。

        参数：
          model_graph：待修改的模型图；调用方应保证是深拷贝。

        返回：
          形状报告字典。

        异常：
          BuildError：符号未知、算子规则不支持或形状契约不满足。
        """
        self._shapes = {}
        self._bind_external_tensors(model_graph)
        inferred = 0
        for op in model_graph.topological_order():
            inputs = [
                self._shape_of(model_graph, name)
                for name in op.inputs
            ]
            output_shape = self._infer_op(op, inputs)
            if len(op.outputs) != 1:
                raise BuildError(
                    "optimizer",
                    "shape inference supports single-output ops only",
                    target=op.name,
                )
            output_name = op.outputs[0]
            tensor = model_graph.tensor(output_name)
            precision = self._resolve_precision(op, model_graph, inputs)
            tensor.shape = tuple(
                Dim(int(value)) for value in output_shape)
            tensor.precision = precision
            tensor.byte_size = (
                math.prod(output_shape)
                * element_bytes(precision)
            )
            tensor.pitch = ()
            self._shapes[output_name] = output_shape
            inferred += 1

        for binding in model_graph.outputs:
            tensor = model_graph.tensor(binding.tensor)
            if not tensor.shape:
                raise BuildError(
                    "optimizer",
                    "output tensor shape is not resolved",
                    target=binding.tensor,
                )
            tensor.reusable = False

        model_graph.metadata["planning"] = {
            "batch": self._config.batch_size,
            "sequence": self._config.max_num_tokens,
            "context_length": self._config.context_length,
        }
        resolved_activations = sum(
            1 for tensor in model_graph.tensors.values()
            if tensor.kind == "activation"
            and tensor.static_shape is not None
        )
        return {
            "batch": self._config.batch_size,
            "sequence": self._config.max_num_tokens,
            "context_length": self._config.context_length,
            "resolved_tensors": resolved_activations,
            "inferred_ops": inferred,
            "unresolved": [],
        }

    def _bind_external_tensors(self, model_graph: ModelGraph) -> None:
        """绑定输入与常量的静态形状并校验常量字节数。

        参数：
          model_graph：模型图。

        异常：
          BuildError：符号未定义、常量形状非法或字节数不一致。
        """
        for name, tensor in model_graph.tensors.items():
            if tensor.kind not in ("input", "constant"):
                continue
            shape = self._resolve_dims(tensor.shape)
            tensor.shape = tuple(Dim(int(value)) for value in shape)
            if tensor.precision == "unknown":
                raise BuildError(
                    "optimizer",
                    "external tensor precision is unknown",
                    target=name,
                )
            expected_bytes = (
                math.prod(shape) * element_bytes(tensor.precision))
            if tensor.kind == "constant":
                if tensor.byte_size != expected_bytes:
                    raise BuildError(
                        "optimizer",
                        "constant byte_size does not match shape/precision: "
                        f"expected {expected_bytes}, "
                        f"got {tensor.byte_size}",
                        target=name,
                    )
            else:
                tensor.byte_size = expected_bytes
                tensor.pitch = ()
            self._shapes[name] = shape

    def _resolve_dims(
        self, dims: tuple[Dim, ...]) -> tuple[int, ...]:
        """把符号维替换为规划值。

        参数：
          dims：维度元组。

        返回：
          静态维度元组。

        异常：
          BuildError：符号未定义。
        """
        resolved = []
        for dim in dims:
            if dim.is_symbol:
                symbol = str(dim.value)
                if symbol not in self._symbols:
                    raise BuildError(
                        "optimizer",
                        f"unknown symbolic dimension: {symbol}",
                    )
                resolved.append(self._symbols[symbol])
            else:
                resolved.append(int(dim.value))
        return tuple(resolved)

    def _shape_of(
        self, model_graph: ModelGraph, name: str) -> tuple[int, ...]:
        """返回已解析的张量形状。

        参数：
          model_graph：模型图；
          name：张量名。

        返回：
          静态形状。

        异常：
          BuildError：张量未登记或形状尚未解析。
        """
        if name in self._shapes:
            return self._shapes[name]
        tensor = model_graph.tensors.get(name)
        if tensor is None:
            raise BuildError(
                "optimizer", "tensor not found", target=name)
        raise BuildError(
            "optimizer",
            "tensor shape is not resolved",
            target=name,
        )

    def _infer_op(
        self,
        op: OpMeta,
        inputs: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """按 op 类型推导输出形状。

        参数：
          op：算子元数据；
          inputs：输入张量静态形状。

        返回：
          输出静态形状。

        异常：
          BuildError：op 不受支持或形状契约不满足。
        """
        if op.op_type == "gather":
            return self._infer_gather(op, inputs)
        if op.op_type == "linear":
            return self._infer_linear(op, inputs)
        if op.op_type == "linear_swi_glu":
            return self._infer_linear_swi_glu(op, inputs)
        if op.op_type == "rms_norm":
            return self._infer_rms_norm(op, inputs)
        if op.op_type == "rope":
            return self._infer_rope(op, inputs)
        if op.op_type == "mega_attention":
            return self._infer_attention(op, inputs)
        if op.op_type == "add":
            return self._infer_add(op, inputs)
        raise BuildError(
            "optimizer",
            f"unsupported op type for shape inference: {op.op_type}",
            target=op.name,
        )

    def _infer_gather(
        self,
        op: OpMeta,
        inputs: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """推导 gather/embedding 输出形状。

        参数：
          op：gather op；
          inputs：indices 与 weight 形状。

        返回：
          ``indices.shape + (weight[-1],)``。

        异常：
          BuildError：输入数量、秩或 axis 非法。
        """
        if len(inputs) < 2:
            raise BuildError(
                "optimizer", "gather requires two inputs", target=op.name)
        indices, weight = inputs[0], inputs[1]
        if len(indices) != 2:
            raise BuildError(
                "optimizer",
                f"gather indices must be rank 2, got {indices}",
                target=op.name,
            )
        if len(weight) != 2:
            raise BuildError(
                "optimizer",
                f"gather weight must be rank 2, got {weight}",
                target=op.name,
            )
        axis = int(op.attrs.get("axis", 0))
        if axis != 0:
            raise BuildError(
                "optimizer",
                f"gather only supports axis 0, got {axis}",
                target=op.name,
            )
        return indices + (weight[1],)

    def _infer_linear(
        self,
        op: OpMeta,
        inputs: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """推导 linear 输出形状。

        参数：
          op：linear op；
          inputs：输入与权重形状。

        返回：
          ``input[:-1] + (out_features,)``。

        异常：
          BuildError：输入/权重形状与属性不一致。
        """
        if len(inputs) != 2:
            raise BuildError(
                "optimizer",
                "linear requires input and weight",
                target=op.name,
            )
        input_shape, weight_shape = inputs
        if not input_shape:
            raise BuildError(
                "optimizer",
                "linear input must have at least one dimension",
                target=op.name,
            )
        in_features = self._int_attr(op, "in_features")
        out_features = self._int_attr(op, "out_features")
        transpose = bool(op.attrs.get("transpose_weight", True))
        expected_weight = (
            (out_features, in_features)
            if transpose else (in_features, out_features)
        )
        if weight_shape != expected_weight:
            raise BuildError(
                "optimizer",
                f"linear weight {weight_shape} != {expected_weight}",
                target=op.name,
            )
        if input_shape[-1] != in_features:
            raise BuildError(
                "optimizer",
                f"linear input last dim {input_shape[-1]} "
                f"!= in_features {in_features}",
                target=op.name,
            )
        return input_shape[:-1] + (out_features,)

    def _infer_linear_swi_glu(
        self,
        op: OpMeta,
        inputs: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """推导 SwiGLU 融合输出形状。

        参数：
          op：linear_swi_glu op；
          inputs：输入、gate 权重、up 权重形状。

        返回：
          ``input[:-1] + (intermediate_size,)``。

        异常：
          BuildError：输入/权重形状与属性不一致。
        """
        if len(inputs) != 3:
            raise BuildError(
                "optimizer",
                "linear_swi_glu requires three inputs",
                target=op.name,
            )
        input_shape, gate_shape, up_shape = inputs
        if not input_shape:
            raise BuildError(
                "optimizer",
                "linear_swi_glu input must have at least one dimension",
                target=op.name,
            )
        hidden_size = self._int_attr(op, "hidden_size")
        intermediate = self._int_attr(op, "intermediate_size")
        transpose = bool(op.attrs.get("transpose_weight", True))
        expected = (
            (intermediate, hidden_size)
            if transpose else (hidden_size, intermediate)
        )
        if gate_shape != expected or up_shape != expected:
            raise BuildError(
                "optimizer",
                "linear_swi_glu weight shapes do not match "
                f"{expected}",
                target=op.name,
            )
        if input_shape[-1] != hidden_size:
            raise BuildError(
                "optimizer",
                f"linear_swi_glu input last dim {input_shape[-1]} "
                f"!= hidden_size {hidden_size}",
                target=op.name,
            )
        return input_shape[:-1] + (intermediate,)

    def _infer_rms_norm(
        self,
        op: OpMeta,
        inputs: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """推导 RMSNorm 输出形状。

        参数：
          op：rms_norm op；
          inputs：输入与权重形状。

        返回：
          与输入同形。

        异常：
          BuildError：归一化维度不能整除或 head 数不匹配。
        """
        if len(inputs) != 2:
            raise BuildError(
                "optimizer",
                "rms_norm requires input and weight",
                target=op.name,
            )
        input_shape, weight_shape = inputs
        if not input_shape:
            raise BuildError(
                "optimizer",
                "rms_norm input must have at least one dimension",
                target=op.name,
            )
        normalized = self._int_attr(op, "normalized_shape")
        if weight_shape != (normalized,):
            raise BuildError(
                "optimizer",
                f"rms_norm weight {weight_shape} != ({normalized},)",
                target=op.name,
            )
        last_dim = input_shape[-1]
        if last_dim % normalized != 0:
            raise BuildError(
                "optimizer",
                f"rms_norm last dim {last_dim} is not divisible by "
                f"{normalized}",
                target=op.name,
            )
        for key in ("num_heads", "num_kv_heads"):
            if key in op.attrs:
                heads = int(op.attrs[key])
                if last_dim != heads * normalized:
                    raise BuildError(
                        "optimizer",
                        f"rms_norm last dim {last_dim} != "
                        f"{key} {heads} * {normalized}",
                        target=op.name,
                    )
        return input_shape

    def _infer_rope(
        self,
        op: OpMeta,
        inputs: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """推导 RoPE 输出形状。

        参数：
          op：rope op；
          inputs：输入与位置张量形状。

        返回：
          与输入同形。

        异常：
          BuildError：末维与 head 配置不匹配或位置形状不一致。
        """
        if len(inputs) != 2:
            raise BuildError(
                "optimizer",
                "rope requires input and position ids",
                target=op.name,
            )
        input_shape, position_shape = inputs
        if not input_shape:
            raise BuildError(
                "optimizer",
                "rope input must have at least one dimension",
                target=op.name,
            )
        head_dim = self._int_attr(op, "head_dim")
        num_heads = self._int_attr(op, "num_heads")
        if input_shape[-1] != num_heads * head_dim:
            raise BuildError(
                "optimizer",
                f"rope input last dim {input_shape[-1]} != "
                f"{num_heads} * {head_dim}",
                target=op.name,
            )
        if position_shape != input_shape[:-1]:
            raise BuildError(
                "optimizer",
                f"rope position shape {position_shape} != "
                f"{input_shape[:-1]}",
                target=op.name,
            )
        return input_shape

    def _infer_attention(
        self,
        op: OpMeta,
        inputs: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """推导 GQA attention 输出形状。

        参数：
          op：mega_attention op；
          inputs：q、k、v 形状。

        返回：
          与 q 同形。

        异常：
          BuildError：head 宽度或 GQA 比例不匹配。
        """
        if len(inputs) != 3:
            raise BuildError(
                "optimizer",
                "mega_attention requires q, k, v",
                target=op.name,
            )
        query, key, value = inputs
        if not query:
            raise BuildError(
                "optimizer",
                "mega_attention query must have at least one dimension",
                target=op.name,
            )
        num_heads = self._int_attr(op, "num_heads")
        num_kv_heads = self._int_attr(op, "num_kv_heads")
        head_dim = self._int_attr(op, "head_dim")
        if num_heads % num_kv_heads != 0:
            raise BuildError(
                "optimizer",
                f"num_heads {num_heads} is not divisible by "
                f"num_kv_heads {num_kv_heads}",
                target=op.name,
            )
        ratio = num_heads // num_kv_heads
        if "gqa_ratio" in op.attrs:
            attribute_ratio = int(op.attrs["gqa_ratio"])
            if attribute_ratio != ratio:
                raise BuildError(
                    "optimizer",
                    f"gqa_ratio {attribute_ratio} != {ratio}",
                    target=op.name,
                )
        if query[-1] != num_heads * head_dim:
            raise BuildError(
                "optimizer",
                f"query last dim {query[-1]} != "
                f"{num_heads} * {head_dim}",
                target=op.name,
            )
        for label, shape in (("key", key), ("value", value)):
            if shape[:-1] != query[:-1]:
                raise BuildError(
                    "optimizer",
                    f"{label} prefix {shape[:-1]} != {query[:-1]}",
                    target=op.name,
                )
            if shape[-1] != num_kv_heads * head_dim:
                raise BuildError(
                    "optimizer",
                    f"{label} last dim {shape[-1]} != "
                    f"{num_kv_heads} * {head_dim}",
                    target=op.name,
                )
        return query

    def _infer_add(
        self,
        op: OpMeta,
        inputs: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """推导逐元素加法输出形状。

        参数：
          op：add op；
          inputs：加数形状。

        返回：
          与第一个输入同形。

        异常：
          BuildError：输入数量不足或形状不一致。
        """
        if len(inputs) < 2:
            raise BuildError(
                "optimizer", "add requires two inputs", target=op.name)
        first = inputs[0]
        for index, shape in enumerate(inputs[1:], start=1):
            if shape != first:
                raise BuildError(
                    "optimizer",
                    f"add input {index} shape {shape} != {first}",
                    target=op.name,
                )
        return first

    @staticmethod
    def _int_attr(op: OpMeta, name: str) -> int:
        """读取整数属性。

        参数：
          op：算子；
          name：属性名。

        返回：
          整数值。

        异常：
          BuildError：属性缺失或不是整数。
        """
        if name not in op.attrs:
            raise BuildError(
                "optimizer",
                f"missing attribute: {name}",
                target=op.name,
            )
        value = op.attrs[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise BuildError(
                "optimizer",
                f"attribute {name} must be an int",
                target=op.name,
            )
        return value

    @staticmethod
    def _resolve_precision(
        op: OpMeta,
        model_graph: ModelGraph,
        inputs: list[tuple[int, ...]],
    ) -> str:
        """确定输出张量精度。

        参数：
          op：算子；
          model_graph：模型图；
          inputs：输入形状（仅用于保持签名一致）。

        返回：
          精度名。

        异常：
          BuildError：无法确定精度。
        """
        del inputs
        attribute = op.attrs.get("precision")
        if isinstance(attribute, str) and attribute != "unknown":
            return attribute
        for name in op.inputs:
            tensor = model_graph.tensor(name)
            if tensor.precision != "unknown":
                return tensor.precision
        raise BuildError(
            "optimizer",
            "cannot resolve output precision",
            target=op.name,
        )
