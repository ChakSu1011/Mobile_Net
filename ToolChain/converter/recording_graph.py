"""模块：Graph 构图代理与元数据记录。

职责：
  以透明代理方式包装 ``Graph.Graph`` 的构图接口，在保持图语义
  不变的前提下旁路记录 ``ModelGraph`` 所需的输入、常量、激活、算子与
  输出绑定元数据。

边界：
  - 不实现第二套可执行图；
  - 不推导 shape/precision，不修改 op 连接；
  - 不持久化数据指针指向的内容；
  - 记录失败时抛出 ``BuildError``，不返回半记录状态。
"""

from __future__ import annotations

from typing import Any

from ToolChain.common.errors import BuildError
from ToolChain.common.graph_api import (
    layout_name,
    op_type_name,
    precision_name,
)
from ToolChain.common.hashing import sha256_bytes
from ToolChain.common.model_ir import (
    Dim,
    IOBinding,
    ModelGraph,
    OpMeta,
    TensorMeta,
    normalize_json_value,
)


class RecordingGraph:
    """Graph的记录代理。

    功能：
      每个构图方法先转发到真实 Graph，成功后再记录元数据；``finish()``
      汇总为 ``ModelGraph`` 并执行自校验。

    边界：
      - 只支持 Graph Python 绑定公开的构图方法；
      - Tensor 句柄仍由 Graph 持有，代理不延长其生命周期；
      - 同一实例只应构建一张图。
    """

    def __init__(self, graph: Any, name: str, adapter: str) -> None:
        """保存真实图并初始化记录表。

        参数：
          graph：``mobilenet_runtime_graph.Graph`` 实例；
          name：图名称，必须非空；
          adapter：模型 Adapter 名称，必须非空。

        异常：
          ValueError：名称或 Adapter 为空。
        """
        if not name:
            raise ValueError("graph name must not be empty")
        if not adapter:
            raise ValueError("adapter must not be empty")
        self._graph = graph
        self._name = name
        self._adapter = adapter
        self._tensors: dict[str, TensorMeta] = {}
        self._ops: list[OpMeta] = []
        self._outputs: list[IOBinding] = []
        self._sorted = False

    @property
    def name(self) -> str:
        """返回图名称。"""
        return self._name

    @property
    def adapted_graph(self) -> Any:
        """返回被包装的 Graph。"""
        return self._graph

    def create_input(
        self,
        name: str,
        precision: Any = None,
        layout: Any = None,
    ) -> Any:
        """创建图输入并记录元数据。

        参数：
          name：输入张量名；
          precision：可选 Precision 枚举；省略时由 Graph 使用
            unknown 精度；
          layout：可选 Layout 枚举。

        返回：
          Graph Tensor 句柄。

        异常：
          RuntimeError：Graph 拒绝创建；
          BuildError：返回张量名与预期不符或名称重复。
        """
        if precision is None and layout is None:
            tensor = self._graph.create_input(name)
            precision_text = "unknown"
            layout_text = "unknown"
        else:
            if precision is None or layout is None:
                raise BuildError(
                    "converter",
                    "precision and layout must be provided together",
                    target=name,
                )
            tensor = self._graph.create_input(name, precision, layout)
            precision_text = precision_name(precision)
            layout_text = layout_name(layout)
        self._record_tensor(
            TensorMeta(
                name=tensor.name,
                kind="input",
                precision=precision_text,
                layout=layout_text,
                reusable=False,
            ),
            expected_name=name,
        )
        return tensor

    def create_constant(
        self,
        name: str,
        dims: list[int] | tuple[int, ...],
        pitch: list[int] | tuple[int, ...],
        precision: Any,
        layout: Any,
        byte_size: int,
        data_ptr: int,
    ) -> Any:
        """创建常量并记录元数据。

        参数：
          name：常量名；
          dims：逻辑形状；
          pitch：物理形状；
          precision：Precision 枚举；
          layout：Layout 枚举；
          byte_size：字节数；
          data_ptr：主机数据地址；0 表示无数据。

        返回：
          Graph Tensor 句柄。

        异常：
          RuntimeError：Graph 拒绝创建；
          BuildError：名称重复或元数据非法。
        """
        tensor = self._graph.create_constant(
            name,
            dims=list(dims),
            pitch=list(pitch),
            precision=precision,
            layout=layout,
            byte_size=byte_size,
            data_ptr=data_ptr,
        )
        self._record_tensor(
            TensorMeta(
                name=tensor.name,
                kind="constant",
                shape=tuple(Dim(int(item)) for item in dims),
                precision=precision_name(precision),
                layout=layout_name(layout),
                byte_size=int(byte_size),
                pitch=tuple(int(item) for item in pitch),
                data_ptr=int(data_ptr),
                reusable=False,
            ),
            expected_name=name,
        )
        return tensor

    def create_op(
        self,
        name: str,
        src_tensors: list[Any] | tuple[Any, ...],
        out_cnt: int,
        op_type: Any,
        attr: dict[str, Any] | None = None,
    ) -> list[Any]:
        """创建计算 op 并记录输入、输出与属性。

        参数：
          name：op 名称；
          src_tensors：输入 Tensor 句柄列表；
          out_cnt：输出张量数量，必须大于 0；
          op_type：OpType 枚举；
          attr：属性字典，必须可 JSON 序列化。

        返回：
          Graph 输出 Tensor 列表。

        异常：
          RuntimeError：Graph 拒绝创建；
          BuildError：out_cnt 非法、属性非法、输入未登记或输出名不符。
        """
        if out_cnt <= 0:
            raise BuildError(
                "converter",
                "out_cnt must be greater than zero",
                target=name,
            )
        source_names = [tensor.name for tensor in src_tensors]
        for tensor_name in source_names:
            if tensor_name not in self._tensors:
                raise BuildError(
                    "converter",
                    "source tensor is not recorded",
                    target=tensor_name,
                )
        expected_outputs = self._output_names(name, out_cnt)
        raw_attr = {} if attr is None else attr
        try:
            normalized_attr = normalize_json_value(
                raw_attr, context=f"op {name}.attrs")
        except ValueError as error:
            raise BuildError(
                "converter", str(error), target=name) from error
        if not isinstance(normalized_attr, dict):
            raise BuildError(
                "converter", "attrs must be a dict", target=name)

        outputs = self._graph.create_op(
            name,
            list(src_tensors),
            out_cnt,
            op_type,
            normalized_attr,
        )
        actual_outputs = [tensor.name for tensor in outputs]
        if actual_outputs != expected_outputs:
            raise BuildError(
                "converter",
                "output tensor names differ: "
                f"expected {expected_outputs}, got {actual_outputs}",
                target=name,
            )

        output_precision = str(
            normalized_attr.get("precision", "unknown"))
        output_layout = str(
            normalized_attr.get("layout", "row_major"))
        for tensor_name in actual_outputs:
            self._record_tensor(
                TensorMeta(
                    name=tensor_name,
                    kind="activation",
                    precision=output_precision,
                    layout=output_layout,
                ),
                expected_name=tensor_name,
            )
        self._ops.append(
            OpMeta(
                name=name,
                op_type=op_type_name(op_type),
                inputs=tuple(source_names),
                outputs=tuple(actual_outputs),
                attrs=normalized_attr,
            )
        )
        return list(outputs)

    def create_output(self, name: str, tensor: Any) -> None:
        """创建图输出绑定并记录。

        参数：
          name：输出 op 名称；
          tensor：被输出的 Tensor 句柄。

        异常：
          RuntimeError：Graph 拒绝创建；
          BuildError：张量未登记或输出名重复。
        """
        tensor_name = tensor.name
        if tensor_name not in self._tensors:
            raise BuildError(
                "converter",
                "output tensor is not recorded",
                target=tensor_name,
            )
        if any(item.name == name for item in self._outputs):
            raise BuildError(
                "converter", "duplicate output name", target=name)
        self._graph.create_output(name, tensor)
        self._tensors[tensor_name].reusable = False
        self._outputs.append(IOBinding(name=name, tensor=tensor_name))

    def sort_op(self) -> None:
        """调用 Graph 拓扑排序并标记已排序。"""
        self._graph.sort_op()
        self._sorted = True

    def dump_ir(self) -> str:
        """返回 Graph 的 IR 文本。"""
        return self._graph.dump_ir()

    def finish(self) -> ModelGraph:
        """生成并返回模型图元数据。

        返回：
          ``ModelGraph``；包含 IR 哈希、行数与 Adapter 信息。

        异常：
          BuildError：尚未调用 ``sort_op()`` 或元数据自校验失败。
        """
        if not self._sorted:
            raise BuildError(
                "converter",
                "sort_op() must be called before finish()",
                target=self._name,
            )
        ir_text = self._graph.dump_ir()
        metadata = {
            "adapter": self._adapter,
            "graph_name": self._name,
            "source_ir_sha256": sha256_bytes(
                ir_text.encode("utf-8")),
            "source_ir_line_count": len(ir_text.splitlines()),
        }
        inputs = tuple(
            name for name, tensor in self._tensors.items()
            if tensor.kind == "input")
        constants = tuple(
            name for name, tensor in self._tensors.items()
            if tensor.kind == "constant")
        try:
            return ModelGraph(
                name=self._name,
                adapter=self._adapter,
                inputs=inputs,
                constants=constants,
                outputs=tuple(self._outputs),
                tensors=self._tensors,
                ops=tuple(self._ops),
                metadata=metadata,
            )
        except (ValueError, BuildError) as error:
            raise BuildError(
                "converter",
                f"invalid model graph: {error}",
                target=self._name,
            ) from error

    def _record_tensor(
        self,
        tensor: TensorMeta,
        *,
        expected_name: str,
    ) -> None:
        """登记张量元数据并校验名称。

        参数：
          tensor：待登记元数据；
          expected_name：调用方期望的名称。

        异常：
          BuildError：名称不符或重复。
        """
        if tensor.name != expected_name:
            raise BuildError(
                "converter",
                f"tensor name mismatch: expected {expected_name}, "
                f"got {tensor.name}",
            )
        if tensor.name in self._tensors:
            raise BuildError(
                "converter",
                "duplicate tensor name",
                target=tensor.name,
            )
        self._tensors[tensor.name] = tensor

    @staticmethod
    def _output_names(name: str, out_cnt: int) -> list[str]:
        """按 Graph 规则生成输出张量名。

        参数：
          name：op 名称；
          out_cnt：输出数量。

        返回：
          单输出为 ``<name>.out``，多输出为 ``<name>.out.<i>``。
        """
        if out_cnt == 1:
            return [f"{name}.out"]
        return [f"{name}.out.{index}" for index in range(out_cnt)]
