# SUPA Python 对外 API 接口设计

## 1. 文档定位

本文只定义 Python 端对外 API，用于指导 `BirenModelEngine` 接口层代码编写。

SUPA runtime 的整体架构、模型加载、Graph 构建、pybind、C++ runtime、metadata 组织、
KV cache、并行切分和底层执行细节，由其他模块设计文档细化；本文不展开这些内部实现。
主设计文档为：

```text
../../.doc/high_performance_runtime_detailed_design.md
```

## 2. 对外入口

Python 端对外只提供一个高层入口类：

```python
class BirenModelEngine:
    ...
```

用户通过该类完成模型加载、runtime 初始化、推理和资源释放。`BirenModelEngine` 内部组合 Python 侧 `ModelLoader`、API 层保存的 typed config / 生命周期状态、已初始化的 graph 字典，以及通过 binding 暴露到 Python 端的 C++ `Engine`。`load_model()` 返回 graph 字典，调用方需将返回值显式传给 `init()`；初始化成功后 API 保存该字典供 Python Forward 查询 main graph。各 API 方法先执行生命周期检查和参数入口检查，再委托内部对象完成实际工作。

## 3. 内部组成与调用关系

`BirenModelEngine` 内部至少包含：

| 成员 | 来源 | 作用 |
|---|---|---|
| `model_config` | `load_model()` 读取模型目录得到 | 保存模型结构配置。 |
| `strategy_config` | `load_model()` 入参 normalize 得到 | 保存 typed 执行策略配置，用于模型加载阶段的切图 / 融合选择，以及 runtime 初始化和执行路径选择；字段可随策略能力扩展。 |
| `runtime_config` | `init()` 入参 normalize 得到 | 保存 typed 运行时资源配置，包括 batch/token/sequence 上限、buffer 上限和设备 id 等。 |
| `model_loader` | Python `ModelLoader` | 负责加载权重、构建 graph，并返回 graph 字典和 typed `ModelConfig`。 |
| `engine` | API 对象创建时创建的 C++ `Engine` binding | 负责 runtime 初始化、推理执行和资源释放。 |
| `state` | API 层状态机 | 记录 `CREATED`、`MODEL_LOADED`、`INITIALIZED` 等生命周期状态。 |

方法委托关系：

| API 方法 | 内部调用 | 调用前检查 |
|---|---|---|
| `load_model()` | `model_loader.load_model()` | 检查当前状态、`model_path` 和 `strategy_config`。 |
| `init()` | `engine.init()` | 检查 graph 字典、runtime 配置和当前状态，并向 C++ 传入三个 typed 子 config。 |
| `forward()` | `engine.forward()` | 检查输入 tensor、输出 tensor、metadata、投机解码输出参数和当前状态，并以语义参数调用 binding。 |
| `uninit()` | `engine.uninit()` | 检查当前状态，并保持未初始化和重复调用行为稳定成功。 |

## 4. API 列表

`BirenModelEngine` 对外提供四个方法：

```python
class BirenModelEngine:
    def __init__(self) -> None:
        ...

    def load_model(self, model_path: str | Path, strategy_config: dict[str, Any]) -> dict[str, Any]:
        ...

    def init(self, graphs: dict[str, Any], runtime_config: dict[str, Any]) -> None:
        ...

    def forward(
        self,
        input_tensor: torch.Tensor | Sequence[torch.Tensor],
        output_tensor: torch.Tensor | Sequence[torch.Tensor],
        metadata: dict[str, Any],
        draft_output_tensor: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> None:
        ...

    def uninit(self) -> None:
        ...
```

## 5. `__init__`

### 接口

```python
def __init__(self) -> None:
    ...
```

### 行为

- 创建 `ModelLoader`。
- 创建 C++ `Engine` binding 对象。
- 创建 API 层对象的初始状态。
- 不读取模型结构配置。
- 不读取模型权重。
- 不构建 graph。
- 不初始化 C++ runtime 资源。

## 6. `load_model`

### 接口

```python
def load_model(self, model_path: str | Path, strategy_config: dict[str, Any]) -> dict[str, Any]:
    ...
```

### 参数

- `model_path`：HuggingFace 格式模型目录。
- `strategy_config`：执行策略配置字典，由用户直接传入，用于描述会影响模型加载、图构建、runtime 初始化和执行路径选择的策略参数；当前可包含并行策略、量化策略、目标设备、投机解码、减层加载等，后续可扩展。

### 返回值

返回 graph 字典：

```python
{
    "main": main_graph,
}
```

如果启用投机采样，可额外包含：

```python
{
    "main": main_graph,
    "draft": draft_graph,
}
```

### 行为

- 检查当前生命周期状态和 `model_path`。
- 检查 `strategy_config` 字典，并转换为 typed `StrategyConfig` 保存。
- 调用 `ModelLoader.load_model(model_path, typed_strategy_config)` 读取模型结构配置、完成权重读取和 graph 构建，并接收 `(graph_dict, model_config)`。
- 保存返回的 `model_config`。
- 返回 graph 字典；`init()` 成功后由 `BirenModelEngine` 保存其引用，供 Python Forward 查询 main graph。
- 不执行 runtime 初始化。

### 约束

- `load_model()` 必须在 `init()` 前调用，并将返回的 graph 字典传给 `init()`。
- runtime 已初始化后，不允许再次调用 `load_model()`。

## 7. `init`

### 接口

```python
def init(self, graphs: dict[str, Any], runtime_config: dict[str, Any]) -> None:
    ...
```

### 参数

- `graphs`：`load_model()` 返回的 graph 字典，至少包含 `"main"`。
- `runtime_config`：运行时配置字典，由用户直接传入，必须包含 `max_batch_size`、`max_num_token`、
  `max_seq_len`、`max_buffer_size`、`device_id`；`device_id` 为单个非负 int。

### 行为

- 检查当前生命周期状态、`graphs` 和 `runtime_config`。
- 检查并转换 `runtime_config` 字典为 typed `RuntimeConfig`。
- 将已保存的 `model_config`、`strategy_config` 和本次 typed `runtime_config` 作为三个 typed 子 config 传入 C++。
- 调用 C++ `Engine.init(graphs, model_config, strategy_config, runtime_config)` 初始化底层 runtime。
- C++ `Engine.init()` 成功返回后，Python API 可释放 `Parameter._torch_value` 持有的 CPU 权重引用；失败时不释放，保留给诊断或重试。
- 初始化成功后进入可推理状态。
- 不重新加载权重。
- 不重新构建 graph。

### 约束

- 调用 `init()` 前必须已经调用 `load_model()`。
- `graphs` 不能为空，且必须包含 `"main"`。
- 已初始化状态下重复调用 `init()` 应报错。

## 8. `forward`

### 接口

```python
def forward(
    self,
    input_tensor: torch.Tensor | Sequence[torch.Tensor],
    output_tensor: torch.Tensor | Sequence[torch.Tensor],
    metadata: dict[str, Any],
    draft_output_tensor: torch.Tensor | Sequence[torch.Tensor] | None = None,
) -> None:
    ...
```

### 参数

- `input_tensor`：main graph 对外输入 tensor 或 tensor 序列。单 tensor 作为长度为 1 的序列处理；序列按 `Graph.GetAllInputOp()` 的边界顺序绑定，且数量必须完全一致。
- `output_tensor`：main graph 对外输出 tensor 或 tensor 序列，由调用方预先分配。单 tensor 作为长度为 1 的序列处理；序列按 `Graph.GetAllOutputOp()` 的边界顺序绑定，且数量必须完全一致。
- `metadata`：本次推理所需的运行时参数字典，例如 batch、sequence、KV cache 等相关信息。
- `draft_output_tensor`：投机解码输出 tensor 或 tensor 序列。未启用投机解码时必须为 `None`；启用时必须非空，并按 draft graph 的 `GetAllOutputOp()` 顺序绑定，数量必须完全一致。

### 行为

- 检查当前生命周期状态、输入 tensor、输出 tensor、metadata 和投机解码输出参数。
- tensor 或 tensor 序列都在 API 层校验；序列不能为空，且每个元素必须是 `torch.Tensor`。
- 根据 `strategy_config.enable_speculative_decoding` 校验 `draft_output_tensor` 是否允许或必填。
- 调用 C++ binding `Engine.forward(input_tensor, output_tensor, metadata, draft_output_tensor)` 执行推理。
- binding 将单 tensor 规范化为单元素 `Storage` 列表，将 tensor 序列按原顺序转换为 `Storage` 列表；C++ Engine 在绑定前检查列表数量与 graph 边界数量一致。
- 默认执行 `"main"` graph。
- 投机解码启用时，main graph 的内部输出 `output_hidden_state` 以及 draft graph 的 `input_hidden_state` / `input_ids` 由 C++ Engine 内部准备，不进入 Python API 参数。
- 输出写入 `output_tensor`；投机解码输出写入 `draft_output_tensor`。
- 返回 `None`。

### 约束

- `forward()` 不传 `graph_name`。
- 调用 `forward()` 前必须已经成功调用 `init()`。
- `forward()` 不读取 checkpoint。
- `forward()` 不修改 graph 结构。
- `forward()` 不申请新的模型权重资源。
- 多输入/多输出采用 graph 边界列表顺序对应，不接受 name-to-tensor dict。

## 9. `uninit`

### 接口

```python
def uninit(self) -> None:
    ...
```

### 行为

- 调用 C++ `Engine.uninit()` 释放底层 runtime 初始化阶段申请的资源。
- 保留已加载模型配置、执行策略信息和 graph 字典引用；graph 对象的实际生命周期由 Python/C++ shared ownership 共同覆盖。
- 调用后允许再次执行 `init()`。

### 约束

- `uninit()` 表示反初始化，不得误调用 `init()`。
- 未初始化状态下调用 `uninit()` 直接按稳定成功处理。
- 重复调用 `uninit()` 不作为用户错误。

## 10. 生命周期

推荐状态流转如下：

```text
CREATED
  |
  | load_model(model_path, strategy_config)
  v
MODEL_LOADED
  |
  | init(graphs, runtime_config)
  v
INITIALIZED
  |
  | forward(input_tensor, output_tensor, metadata, draft_output_tensor=None)
  v
INITIALIZED
  |
  | uninit()
  v
MODEL_LOADED
```

错误用法：

| 场景 | 行为 |
|---|---|
| 未 `load_model()` 调用 `init()` | 报错 |
| `init()` 传入空 graph 字典 | 报错 |
| `init()` 传入的 graph 字典缺少 `"main"` | 报错 |
| 未 `init()` 调用 `forward()` | 报错 |
| 已 `init()` 后再次调用 `load_model()` | 报错 |
| 已 `init()` 后重复调用 `init()` | 报错 |

## 11. 示例

```python
from tensorrt_llm._supa.python.api import BirenModelEngine

engine = BirenModelEngine()

graphs = engine.load_model(
    "/path/to/hf-model",
    {
        "tp_size": 1,
        "pp_size": 1,
        "ep_size": 1,
        "cp_size": 1,
        "weight_precision": "fp16",
        "activation_precision": "fp16",
        "expert_precision": "fp16",
        "kv_precision": "fp16",
        "device_type": "br288",
    },
)

engine.init(
    graphs,
    {
        "max_batch_size": 8,
        "max_num_token": 4096,
        "max_seq_len": 8192,
        "max_buffer_size": 8 * 1024 * 1024 * 1024,
        "device_id": 0,
    },
)

engine.forward(input_tensor, output_tensor, metadata)

engine.uninit()
```

## 12. API 层开发要求

- 文件建议放在 `tensorrt_llm/_supa/python/api.py`。
- 对外类名统一为 `BirenModelEngine`。
- 对外只暴露本文列出的四个核心方法。
- API 层内部创建并持有 `ModelLoader` 和 C++ `Engine` binding 对象。
- API 层负责生命周期检查和参数入口检查。
- API 层不展开模型加载、Graph 构建、metadata 解析和 C++ runtime 内部逻辑。
- 上述内部逻辑通过对应模块接口调用，具体设计以各自模块文档为准。

## 14. Python Forward 执行后端

`StrategyConfig` 提供 `use_cpp: bool = True`。`BirenModelEngine.init()` 保存已初始化的 graph 字典，以便 Forward 查询 main graph；`forward()` 在完成原有输入、输出、metadata 和生命周期校验后按以下规则分派：

```text
use_cpp=True   -> engine.forward(...)
use_cpp=False  -> executor.execute_graph(engine, graphs["main"], ...)
```

Python executor 在执行前调用 `engine.prepare_python_gpu_context(graph, input_tensor, output_tensor, context)`；该接口在 C++ 侧完成 boundary storage 绑定、RuntimeContext 转换/缓存和 GraphEnv metadata 准备。随后逐个遍历 `graph.get_orderly_op_list()`，每个 op 先调用无 context 参数的 `engine.infer_shape(op)`，通过 `PYTHON_OP_REGISTRY` 按 `op_name` 查找 Python 实现；当前注册 `GATHER`/embedding 和 `RMS_NORM`，未注册 op 调用 `engine.forward_op(op)` 执行当前 C++ OpEnv。executor 不修改 graph 拓扑、不申请 workspace、不拥有或释放底层 pointer。

注册的 Python 算子可按名称调用 `op.get_share_data(name)` 和
`op.get_temp_data(name)`，获得与 C++ Storage 共享内存的 Torch Tensor。share data
返回本轮实际写入的逻辑长度并按只读资源使用；当前 runtime share 数组不接受
零长度输入。tempdata 必须由对应 C++ OpEnv 在
Init 阶段声明完整 shape/precision/layout，并由 Manager 规划和 realize。Python
不得声明、扩容、替换或跨 op 保存这些资源；未知名称或未 realize pointer 必须报错。

执行期 Python 算子统一放在 `python/runtime_ops/`，不得放入构图期
`python/ops/`。目录按 `standalone/` 和 `fused_ops/` 分类，不设置 `kernels/`
子目录；每个算子独立成文件，根目录 `registry.py` 维护
`PYTHON_OP_REGISTRY`。当前 `embedding` 和 `rms_norm` 位于 `standalone/`，注册为
`GATHER -> embedding` 和
`RMS_NORM -> rms_norm`。RMSNorm 支持 input 加可选 weight，从
`op.attrs["eps"]` 读取 epsilon，并使用 FP32 中间计算后写回 graph-owned
destination tensor。executor 将 `StrategyConfig.use_fused_op` 和 runtime stream
作为每个 registry runner 的第二、第三个参数传入；RMSNorm 在 `false` 时使用当前 Torch
散算子组合，在 `true` 时选择 SUPA 融合实现。融合实现接入前，`true` 分支必须
抛出明确的 `NotImplementedError`，不得静默退回散算子。Embedding 接收同一
参数以保持 registry 签名一致，但当前没有融合/散算子分支。

Standalone Python runtime 算子的模块边界、内存约束和扩展流程见
`runtime_ops/standalone/.doc/standalone_runtime_ops_design.md`。

Python Torch 算子与 `engine.forward_op(op)` 必须提交到同一 SUPA stream。
SupaModelEngine 的 runtime config 传入 Torch SUPA stream 对象；
`BirenModelEngine.init()` 保留该对象，并在调用 C++ init 前把配置副本中的 stream
转换为 `stream.supa_stream` 句柄。`forward()` 将原 stream 对象传入 executor，
registry runner 使用 `torch.supa.stream(stream)` context 提交 Torch 计算。调用方
无需预先切换当前 stream，executor 也不插入逐 op 全局同步。

Python 路径当前执行 main graph。投机解码打开时由于 draft graph 内部输入准备尚未纳入 Python executor，`forward()` 必须抛出明确的 `NotImplementedError`；`use_cpp=True` 的投机路径保持现有行为。