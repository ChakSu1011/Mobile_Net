"""模块：Build 阶段模型 IR 数据类。

职责：
  定义 Converter 记录的 ``ModelGraph``、张量元数据 ``TensorMeta``、
  算子元数据 ``OpMeta``、I/O 绑定、显存计划数据结构与 JSON 往返。

边界：
  - 不创建 Graph，不执行构图；
  - 不推导 shape/precision，不分配显存；
  - 图对象以名称引用，TensorMeta 是图元数据的旁路副本。

主要逻辑：
  1. 数据类在构造时做字段与命名校验；
  2. ``ModelGraph.validate()`` 校验引用完整性与无环；
  3. ``topological_order()`` 复刻 Graph 的 post-DFS 顺序；
  4. ``to_dict()/from_dict()`` 提供确定性的 JSON 往返。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ToolChain.common.errors import BuildError
from ToolChain.common.jsonio import read_json, write_json

SCHEMA_VERSION = 1
_TENSOR_KINDS = ("input", "constant", "activation", "output")
_LAYOUTS = ("row_major", "column_major", "unknown")


def normalize_json_value(value: Any, *, context: str) -> Any:
    """把属性值递归转换为 JSON 原生类型。

    参数：
      value：待转换值，允许 None/bool/int/float/str/list/tuple/dict；
      context：错误信息中的位置描述，例如 ``op.attr``。

    返回：
      JSON 原生对象；None 保留为 JSON null，元组转换为列表，
      字典键必须为字符串。

    异常：
      ValueError：类型不支持、浮点为 NaN/Inf 或字典键非字符串。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{context}: float value must be finite")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [
            normalize_json_value(item, context=context)
            for item in value
        ]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{context}: dict keys must be strings")
            result[key] = normalize_json_value(
                item, context=context)
        return result
    raise ValueError(
        f"{context}: unsupported value type "
        f"{type(value).__name__}")


@dataclass(frozen=True)
class Dim:
    """张量维度。

    功能：
      表示一个静态维度（int）或符号维度（str，如 ``B``、``S``）。

    边界：
      只描述单维，不表达 stride、offset 或动态取值范围。
    """

    value: int | str

    def __post_init__(self) -> None:
        """校验维度取值。"""
        if isinstance(self.value, bool):
            raise ValueError("dim must not be bool")
        if isinstance(self.value, int):
            if self.value < 0:
                raise ValueError("dim must not be negative")
            return
        if isinstance(self.value, str):
            if not self.value:
                raise ValueError("symbolic dim must not be empty")
            return
        raise ValueError("dim must be int or str")

    @property
    def is_symbol(self) -> bool:
        """返回是否为符号维。"""
        return isinstance(self.value, str)

    def to_json(self) -> int | str:
        """返回 JSON 原生表示。"""
        return self.value

    @classmethod
    def from_json(cls, value: int | str) -> "Dim":
        """从 JSON 原生值构造维度。"""
        return cls(value=value)


@dataclass
class TensorMeta:
    """张量元数据。

    功能：
      描述输入、常量、激活或输出张量的名称、形状、精度与存储信息。

    边界：
      不持有真实数据指针；``data_ptr`` 只用于记录，不参与序列化加载。
    """

    name: str
    kind: str
    shape: tuple[Dim, ...] = ()
    precision: str = "unknown"
    layout: str = "row_major"
    byte_size: int = 0
    pitch: tuple[int, ...] = ()
    data_ptr: int = 0
    reusable: bool = True
    weight_source: str | None = None
    weight_available: bool = False
    quantization: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """校验字段取值。"""
        if not self.name:
            raise ValueError("tensor name must not be empty")
        if self.kind not in _TENSOR_KINDS:
            raise ValueError(f"unsupported tensor kind: {self.kind}")
        if not self.precision:
            raise ValueError("precision must not be empty")
        if self.layout not in _LAYOUTS:
            raise ValueError(f"unsupported layout: {self.layout}")
        if self.byte_size < 0:
            raise ValueError("byte_size must not be negative")
        if self.data_ptr < 0:
            raise ValueError("data_ptr must not be negative")
        for dim in self.shape:
            if not isinstance(dim, Dim):
                raise ValueError("shape elements must be Dim")
        if self.pitch:
            if len(self.pitch) != len(self.shape):
                raise ValueError("pitch length must match shape length")
            for value in self.pitch:
                if value < 0:
                    raise ValueError("pitch must not be negative")
        if self.quantization is not None:
            normalize_json_value(
                self.quantization, context=f"tensor {self.name}.quantization")

    @property
    def static_shape(self) -> tuple[int, ...] | None:
        """返回全部为静态值的形状；含符号维时返回 None。"""
        if any(dim.is_symbol for dim in self.shape):
            return None
        return tuple(int(dim.value) for dim in self.shape)

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 原生表示。"""
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "shape": [dim.to_json() for dim in self.shape],
            "precision": self.precision,
            "layout": self.layout,
            "byte_size": self.byte_size,
            "pitch": list(self.pitch),
            "data_ptr": self.data_ptr,
            "reusable": self.reusable,
            "weight_source": self.weight_source,
            "weight_available": self.weight_available,
        }
        if self.quantization is not None:
            result["quantization"] = self.quantization
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorMeta":
        """从 JSON 字典构造张量元数据。

        异常：
          KeyError/ValueError：字段缺失或非法。
        """
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            shape=tuple(
                Dim.from_json(item) for item in data.get("shape", [])),
            precision=str(data.get("precision", "unknown")),
            layout=str(data.get("layout", "row_major")),
            byte_size=int(data.get("byte_size", 0)),
            pitch=tuple(int(item) for item in data.get("pitch", [])),
            data_ptr=int(data.get("data_ptr", 0)),
            reusable=bool(data.get("reusable", True)),
            weight_source=(
                None if data.get("weight_source") is None
                else str(data["weight_source"])
            ),
            weight_available=bool(
                data.get("weight_available", False)),
            quantization=data.get("quantization"),
        )


@dataclass
class OpMeta:
    """算子元数据。

    功能：
      记录 op 名称、类型、输入/输出张量名与属性。

    边界：
      不执行算子；attrs 必须可 JSON 序列化。
    """

    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)
    origin: str | None = None

    def __post_init__(self) -> None:
        """校验字段并规范化属性。"""
        if not self.name:
            raise ValueError("op name must not be empty")
        if not self.op_type:
            raise ValueError("op_type must not be empty")
        if not self.outputs:
            raise ValueError("op must have at least one output")
        normalized = normalize_json_value(
            self.attrs, context=f"op {self.name}.attrs")
        if not isinstance(normalized, dict):
            raise ValueError("attrs must be a dict")
        self.attrs = normalized

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 原生表示。"""
        result: dict[str, Any] = {
            "name": self.name,
            "op_type": self.op_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": self.attrs,
        }
        if self.origin is not None:
            result["origin"] = self.origin
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpMeta":
        """从 JSON 字典构造算子元数据。"""
        return cls(
            name=str(data["name"]),
            op_type=str(data["op_type"]),
            inputs=tuple(str(item) for item in data["inputs"]),
            outputs=tuple(str(item) for item in data["outputs"]),
            attrs=dict(data.get("attrs", {})),
            origin=(
                None if data.get("origin") is None
                else str(data["origin"])
            ),
        )


@dataclass(frozen=True)
class IOBinding:
    """图输出绑定。

    功能：
      记录输出 op 名称与被输出的张量名。
    """

    name: str
    tensor: str

    def __post_init__(self) -> None:
        """校验名称非空。"""
        if not self.name:
            raise ValueError("output name must not be empty")
        if not self.tensor:
            raise ValueError("output tensor must not be empty")

    def to_dict(self) -> dict[str, str]:
        """返回 JSON 原生表示。"""
        return {"name": self.name, "tensor": self.tensor}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IOBinding":
        """从 JSON 字典构造输出绑定。"""
        return cls(name=str(data["name"]), tensor=str(data["tensor"]))


@dataclass(frozen=True)
class MemoryEntry:
    """单个张量的显存排版条目。"""

    name: str
    offset: int
    size: int
    birth: int
    last_use: int
    reusable: bool

    def __post_init__(self) -> None:
        """校验偏移与生命周期。"""
        if self.offset < 0 or self.size < 0:
            raise ValueError("memory offset/size must not be negative")
        if self.birth > self.last_use:
            raise ValueError("birth must not exceed last_use")

    @property
    def end(self) -> int:
        """返回区间结束偏移（不含）。"""
        return self.offset + self.size

    def overlaps(self, other: "MemoryEntry") -> bool:
        """判断生命周期与字节区间是否同时重叠。"""
        lifetime_overlap = (
            self.birth <= other.last_use
            and other.birth <= self.last_use
        )
        byte_overlap = (
            self.offset < other.end and other.offset < self.end
        )
        return lifetime_overlap and byte_overlap

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 原生表示。"""
        return {
            "name": self.name,
            "offset": self.offset,
            "size": self.size,
            "birth": self.birth,
            "last_use": self.last_use,
            "reusable": self.reusable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """从 JSON 字典构造显存条目。"""
        return cls(
            name=str(data["name"]),
            offset=int(data["offset"]),
            size=int(data["size"]),
            birth=int(data["birth"]),
            last_use=int(data["last_use"]),
            reusable=bool(data["reusable"]),
        )


@dataclass(frozen=True)
class MemoryPlan:
    """Activation 显存计划。

    功能：
      汇总对齐、arena 高水位、理论峰值、常量区字节与逐张量条目。

    边界：
      只描述逻辑偏移，不分配真实显存。
    """

    alignment: int
    arena_bytes: int
    peak_bytes: int
    constants_bytes: int
    entries: tuple[MemoryEntry, ...]

    def __post_init__(self) -> None:
        """校验数值非负。"""
        for name, value in (
            ("alignment", self.alignment),
            ("arena_bytes", self.arena_bytes),
            ("peak_bytes", self.peak_bytes),
            ("constants_bytes", self.constants_bytes),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.alignment == 0:
            raise ValueError("alignment must be positive")

    def entry(self, name: str) -> MemoryEntry:
        """按张量名返回显存条目。

        异常：
          KeyError：名称不存在。
        """
        for item in self.entries:
            if item.name == name:
                return item
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 原生表示。"""
        return {
            "alignment": self.alignment,
            "arena_bytes": self.arena_bytes,
            "peak_bytes": self.peak_bytes,
            "constants_bytes": self.constants_bytes,
            "entries": [item.to_dict() for item in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryPlan":
        """从 JSON 字典构造显存计划。"""
        return cls(
            alignment=int(data["alignment"]),
            arena_bytes=int(data["arena_bytes"]),
            peak_bytes=int(data["peak_bytes"]),
            constants_bytes=int(data["constants_bytes"]),
            entries=tuple(
                MemoryEntry.from_dict(item)
                for item in data.get("entries", [])
            ),
        )


@dataclass
class ModelGraph:
    """模型图元数据。

    功能：
      保存图名、Adapter、输入/常量名称、张量表、算子表、输出绑定与
      模型元信息，作为 Optimizer 与 Serializer 的输入。

    边界：
      不是可执行图；执行仍由 Graph 承担。
    """

    name: str
    adapter: str
    inputs: tuple[str, ...]
    constants: tuple[str, ...]
    outputs: tuple[IOBinding, ...]
    tensors: dict[str, TensorMeta]
    ops: tuple[OpMeta, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """规范化字段并执行自校验。"""
        if not self.name:
            raise ValueError("graph name must not be empty")
        if not self.adapter:
            raise ValueError("adapter must not be empty")
        self.inputs = tuple(str(item) for item in self.inputs)
        self.constants = tuple(str(item) for item in self.constants)
        self.outputs = tuple(self.outputs)
        self.ops = tuple(self.ops)
        normalized = normalize_json_value(
            self.metadata, context="metadata")
        if not isinstance(normalized, dict):
            raise ValueError("metadata must be a dict")
        self.metadata = normalized
        self.validate()

    def tensor(self, name: str) -> TensorMeta:
        """按名称返回张量元数据。

        异常：
          KeyError：名称不存在。
        """
        return self.tensors[name]

    @property
    def activation_tensors(self) -> list[TensorMeta]:
        """按登记顺序返回全部 activation 张量。"""
        return [
            tensor for tensor in self.tensors.values()
            if tensor.kind == "activation"
        ]

    @property
    def constant_bytes(self) -> int:
        """返回全部常量字节数之和。"""
        return sum(
            tensor.byte_size
            for tensor in self.tensors.values()
            if tensor.kind == "constant"
        )

    def producer_map(self) -> dict[str, OpMeta]:
        """返回 activation 张量名到生产 op 的映射。

        异常：
          BuildError：同一张量有多个生产 op。
        """
        producers: dict[str, OpMeta] = {}
        for op in self.ops:
            for output in op.outputs:
                if output in producers:
                    raise BuildError(
                        "model_ir",
                        "tensor has multiple producers",
                        target=output,
                    )
                producers[output] = op
        return producers

    def validate(self) -> None:
        """校验引用完整性、命名唯一性与无环。

        返回：
          无；校验通过时直接返回。

        异常：
          BuildError：输入/输出缺失、引用不存在、重复名称、成环或
            存在无法从输出到达的 op。
        """
        if not self.inputs:
            raise BuildError(
                "model_ir", "graph must have at least one input")
        if not self.outputs:
            raise BuildError(
                "model_ir", "graph must have at least one output")
        if not self.ops:
            raise BuildError(
                "model_ir", "graph must have at least one op")

        for name in self.inputs:
            tensor = self.tensors.get(name)
            if tensor is None:
                raise BuildError(
                    "model_ir", "input tensor not found", target=name)
            if tensor.kind != "input":
                raise BuildError(
                    "model_ir",
                    f"input tensor kind must be input, got {tensor.kind}",
                    target=name)
        for name in self.constants:
            tensor = self.tensors.get(name)
            if tensor is None:
                raise BuildError(
                    "model_ir",
                    "constant tensor not found",
                    target=name,
                )
            if tensor.kind != "constant":
                raise BuildError(
                    "model_ir",
                    "constant tensor kind must be constant, "
                    f"got {tensor.kind}",
                    target=name,
                )

        if len(set(self.inputs)) != len(self.inputs):
            raise BuildError(
                "model_ir", "duplicate input tensor name")
        if len(set(self.constants)) != len(self.constants):
            raise BuildError(
                "model_ir", "duplicate constant tensor name")

        op_names: set[str] = set()
        output_names: set[str] = set()
        for op in self.ops:
            if op.name in op_names:
                raise BuildError(
                    "model_ir", "duplicate op name", target=op.name)
            op_names.add(op.name)
            if not op.inputs:
                raise BuildError(
                    "model_ir",
                    "op must have at least one input",
                    target=op.name,
                )
            for name in op.inputs + op.outputs:
                if name not in self.tensors:
                    raise BuildError(
                        "model_ir",
                        f"tensor not found: {name}",
                        target=op.name,
                    )
            for name in op.outputs:
                tensor = self.tensors[name]
                if tensor.kind != "activation":
                    raise BuildError(
                        "model_ir",
                        "op output tensor kind must be activation, "
                        f"got {tensor.kind}",
                        target=name,
                    )

        producers = self.producer_map()
        for name, tensor in self.tensors.items():
            if tensor.kind == "input" and name in producers:
                raise BuildError(
                    "model_ir",
                    "input tensor must not be produced by op",
                    target=name,
                )
            if tensor.kind == "constant" and name in producers:
                raise BuildError(
                    "model_ir",
                    "constant tensor must not be produced by op",
                    target=name,
                )
        for name in self.tensors:
            if name in producers:
                continue
            tensor = self.tensors[name]
            if tensor.kind == "activation":
                raise BuildError(
                    "model_ir",
                    "activation tensor has no producer",
                    target=name,
                )

        for binding in self.outputs:
            if binding.name in output_names:
                raise BuildError(
                    "model_ir",
                    "duplicate output name",
                    target=binding.name,
                )
            output_names.add(binding.name)
            if binding.tensor not in self.tensors:
                raise BuildError(
                    "model_ir",
                    "output tensor not found",
                    target=binding.tensor,
                )
            if binding.tensor not in producers:
                raise BuildError(
                    "model_ir",
                    "output tensor has no producer",
                    target=binding.tensor,
                )

        reachable = self._reachable_op_names(producers)
        if len(reachable) != len(self.ops):
            missing = sorted(op.name for op in self.ops
                             if op.name not in reachable)
            raise BuildError(
                "model_ir",
                "ops are not reachable from graph outputs: "
                + ",".join(missing[:5]),
            )

    def _reachable_op_names(
        self, producers: dict[str, OpMeta]) -> set[str]:
        """从输出绑定出发 DFS，返回可达 op 名称集合。

        参数：
          producers：activation 张量名到生产 op 的映射。

        返回：
          可达 op 名称集合。

        异常：
          BuildError：检测到依赖环或 activation 缺少生产者。
        """
        producer_by_name = {
            op.name: op for op in self.ops}
        state: dict[str, int] = {}
        reachable: set[str] = set()

        def visit(op_name: str) -> None:
            current = state.get(op_name, 0)
            if current == 1:
                raise BuildError(
                    "model_ir",
                    "dependency cycle detected",
                    target=op_name,
                )
            if current == 2:
                return
            state[op_name] = 1
            op = producer_by_name[op_name]
            for tensor_name in op.inputs:
                tensor = self.tensors[tensor_name]
                if tensor.kind in ("constant", "input"):
                    continue
                producer = producers.get(tensor_name)
                if producer is None:
                    raise BuildError(
                        "model_ir",
                        "activation tensor has no producer",
                        target=tensor_name,
                    )
                visit(producer.name)
            state[op_name] = 2
            reachable.add(op_name)

        for binding in self.outputs:
            producer = producers[binding.tensor]
            visit(producer.name)
        return reachable

    def topological_order(self) -> list[OpMeta]:
        """按 Graph 的 post-DFS 规则返回拓扑序。

        返回：
          op 列表；生产者一定排在消费者之前。

        异常：
          BuildError：依赖环、activation 缺少生产者或输出无生产者。
        """
        producers = self.producer_map()
        state: dict[str, int] = {}
        ordered: list[OpMeta] = []

        def visit(op: OpMeta) -> None:
            current = state.get(op.name, 0)
            if current == 1:
                raise BuildError(
                    "model_ir",
                    "dependency cycle detected",
                    target=op.name,
                )
            if current == 2:
                return
            state[op.name] = 1
            for tensor_name in op.inputs:
                tensor = self.tensors[tensor_name]
                if tensor.kind in ("constant", "input"):
                    continue
                producer = producers.get(tensor_name)
                if producer is None:
                    raise BuildError(
                        "model_ir",
                        "activation tensor has no producer",
                        target=tensor_name,
                    )
                visit(producer)
            state[op.name] = 2
            ordered.append(op)

        for binding in self.outputs:
            producer = producers.get(binding.tensor)
            if producer is None:
                raise BuildError(
                    "model_ir",
                    "output tensor has no producer",
                    target=binding.tensor,
                )
            visit(producer)
        return ordered

    def clone(self) -> "ModelGraph":
        """返回深拷贝。"""
        return ModelGraph.from_dict(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 原生表示。"""
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "adapter": self.adapter,
            "inputs": list(self.inputs),
            "constants": list(self.constants),
            "outputs": [
                binding.to_dict() for binding in self.outputs],
            "tensors": [
                tensor.to_dict() for tensor in self.tensors.values()],
            "ops": [op.to_dict() for op in self.ops],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelGraph":
        """从 JSON 字典构造模型图。

        异常：
          KeyError/ValueError/BuildError：字段缺失或引用非法。
        """
        schema_version = int(data.get("schema_version", 0))
        if schema_version != SCHEMA_VERSION:
            raise BuildError(
                "model_ir",
                f"unsupported schema version: {schema_version}",
            )
        tensors: dict[str, TensorMeta] = {}
        for item in data.get("tensors", []):
            tensor = TensorMeta.from_dict(item)
            if tensor.name in tensors:
                raise BuildError(
                    "model_ir",
                    "duplicate tensor name",
                    target=tensor.name,
                )
            tensors[tensor.name] = tensor
        return cls(
            name=str(data["name"]),
            adapter=str(data["adapter"]),
            inputs=tuple(str(item) for item in data["inputs"]),
            constants=tuple(
                str(item) for item in data["constants"]),
            outputs=tuple(
                IOBinding.from_dict(item)
                for item in data["outputs"]
            ),
            tensors=tensors,
            ops=tuple(
                OpMeta.from_dict(item) for item in data["ops"]),
            metadata=dict(data.get("metadata", {})),
        )

    def save_json(self, path: Path) -> None:
        """把模型图写入 JSON 文件。

        参数：
          path：输出路径；父目录必须存在。

        异常：
          OSError：写入失败。
        """
        write_json(path, self.to_dict())

    @classmethod
    def load_json(cls, path: Path) -> "ModelGraph":
        """从 JSON 文件读取模型图。

        异常：
          OSError：读取失败；
          BuildError：格式或引用非法。
        """
        return cls.from_dict(read_json(path))

    def summary(self) -> dict[str, Any]:
        """返回计数与字节数摘要。"""
        precision_histogram: dict[str, int] = {}
        for tensor in self.tensors.values():
            precision_histogram[tensor.precision] = (
                precision_histogram.get(tensor.precision, 0) + 1)
        return {
            "name": self.name,
            "adapter": self.adapter,
            "inputs": len(self.inputs),
            "constants": len(self.constants),
            "ops": len(self.ops),
            "outputs": len(self.outputs),
            "activation_tensors": len(self.activation_tensors),
            "total_tensors": len(self.tensors),
            "constant_bytes": self.constant_bytes,
            "precision_histogram": dict(
                sorted(precision_histogram.items())),
        }
