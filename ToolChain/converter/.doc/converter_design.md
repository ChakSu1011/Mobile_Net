# Converter 模块设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v1.0 |
| 文档状态 | 已实现，随 Build 阶段代码同步 |
| 编写日期 | 2026-11-13 |
| 所属阶段 | Build（离线工具链） |
| 模块目录 | `ToolChain/converter/` |
| 上游依赖 | `Graph`（图对象模型）、`Models/Qwen3`（Qwen3 Adapter） |
| 下游模块 | `ToolChain/optimizer`、`ToolChain/serializer` |
| 权威图设计 | [Graph/.doc/Graph_design.md](../../../Graph/.doc/Graph_design.md) |

## 1. 背景与目标

### 1.1 背景

推理引擎在工具链端的流程为“模型源码 → Python 端静态图 → 精度/Shape 推导 →
显存管理 → 序列化”。其中静态图表示已由 `Graph` 重新定义：图、算子、
张量与常量全部由 C++ 实现，Python 侧只通过 pybind11 暴露最小构图接口。

`Graph` 的 Python 绑定刻意只暴露 `Graph`、`Tensor` 与四个枚举，
`Tensor` 仅提供 `name`、`is_constant`、`reusable`、`set_reusable`，
不提供 shape、precision、storage 访问器，也不提供 op 遍历接口。
因此 Converter 不能把优化与序列化所需元数据只留在 Graph 对象里，必须
在构图的同时旁路记录一份 `ModelGraph` 元数据。

### 1.2 目标

1. 以 `Graph` 为唯一图表示，把模型源码转换为可执行的静态图。
2. 首期实现 Qwen3-0.6B Dense Decoder-only Adapter，复用
   `Models/Qwen3/qwen3_graph_builder.py` 的既有层结构。
3. 构图时旁路记录 `ModelGraph`，完整描述输入、常量、激活张量、算子属性、
   权重来源与图输出，供 Optimizer 与 Serializer 消费。
4. 提供 `RecordingGraph`，对 `Graph` 的构图 API 做透明代理与记录，
   不改变图语义，不维护第二套可执行图。
5. 支持权重目录探测：只读取 safetensors 头信息，登记权重是否可用，
   不加载真实权重内容。
6. 提供命令行转换入口、单元测试与可复现的转换报告。
7. 读取并校验模型目录下的运行时输入配置
   `input/inference_config.json`，只补充官方 `config.json` 与
   `generation_config.json` 没有的引擎参数，并合并为有效运行参数写入
   `ModelGraph.metadata`。

### 1.3 非目标

- 不实现精度推导、形状推导与显存规划（属于 Optimizer）。
- 不实现模型包序列化与反序列化（属于 Serializer）。
- 不执行数值推理、不加载权重内容、不校验数值正确性。
- 不实现 Tokenization、采样执行或对话模板；输入配置只记录默认参数。
- 不实现 Qwen3 之外的模型 Adapter；Adapter 注册机制预留。
- 不支持 MoE、VL、KV Cache、padding mask 等扩展结构。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收要点 |
| --- | --- | --- |
| FR-CONV-01 | `RecordingGraph` 必须完整代理 `create_input`、`create_constant`、`create_op`、`create_output`、`sort_op`、`dump_ir`，调用方无感知 | 使用 RecordingGraph 与直接使用 Graph 构建同一模型，`dump_ir()` 完全一致 |
| FR-CONV-02 | Converter 输出必须同时包含可执行的 `Graph.Graph` 与 `ModelGraph` 元数据 | 两个对象均非空；`ModelGraph` 通过自校验 |
| FR-CONV-03 | Qwen3 Adapter 必须按 `Models/Qwen3/qwen3_graph_builder.py` 的结构构图，不得复制或改写层结构 | 图计数与既有 Qwen3 测试一致：2 inputs、310 constants、311 个计算 op、1 output；`dump_ir()` 的 `ops` 段为 314，因为额外包含 2 个 Input op 与 1 个 Output op |
| FR-CONV-04 | 每个 constant 必须记录图内名称、shape、dtype、byte_size 与 Hugging Face 权重名 | 覆盖 embedding、28 层 11 个权重、final norm；tied LM Head 记录别名关系 |
| FR-CONV-05 | 输入张量必须记录符号形状 `[B, S]` 与 `int64` 精度，并作为 `ModelGraph` 的图输入 | `input_ids`、`position_ids` 均为 `[B, S]`、`int64` |
| FR-CONV-06 | 支持可选 safetensors 权重目录探测，登记权重可用性与来源文件 | 有真实权重时 `weight_available=true`；缺失时为 false，不报错 |
| FR-CONV-07 | `ModelGraph` 必须支持规范化 JSON 保存/加载往返 | 保存后加载，`to_dict()` 逐字段一致 |
| FR-CONV-08 | 提供 CLI：输入 config 与可选权重目录，输出 `model_graph.json` 与转换摘要 | 命令退出码 0；输出文件存在且可被 Optimizer 加载 |
| FR-CONV-09 | 对非法模型结构、重复名称、未知 op 类型、非法属性值必须给出带上下文的 `BuildError` | 错误信息包含阶段与对象名；不产生半成品文件 |
| FR-CONV-10 | 读取并校验 `Models/<模型>/input/inference_config.json`，只声明官方 config 没有的引擎参数；官方 `config.json` 与 `generation_config.json` 已有参数直接读取，不得重复声明 | 有效配置包含官方 context/vocab/采样/eos 与引擎 max token 参数；重复或未知字段报错；缺失文件时保持兼容 |

### 2.2 非功能需求

| 编号 | 类别 | 要求 |
| --- | --- | --- |
| NFR-CONV-01 | 正确性 | 代理层不得改变 Graph 的命名、连接与拓扑排序语义 |
| NFR-CONV-02 | 可重复性 | 同一配置与权重目录产生完全一致的 `ModelGraph` JSON |
| NFR-CONV-03 | 性能 | Qwen3-0.6B 转换在 10 s 内完成（不含权重读取） |
| NFR-CONV-04 | 内存 | 只记录元数据；不复制权重字节；峰值内存与图规模线性相关 |
| NFR-CONV-05 | 规范 | Python 代码遵循 Google Style；模块/类/函数含中文注释 |
| NFR-CONV-06 | 可测试性 | 核心逻辑不依赖 torch/transformers；单元测试可在纯 CPU 环境运行 |
| NFR-CONV-07 | 单一事实源 | 官方 config 已有参数不得在新配置重复声明；合并结果确定、可审查 |

### 2.3 约束条件

- Python 3.10+；图对象必须来自 `Graph/python/mobilenet_runtime_graph.py`。
- 原生扩展位于 `Graph/build/python/_graph.pyd`（或 `.so`/`.dylib`）。
- `Graph` 不允许删除/改写已创建的 op 与 tensor；Converter 只做增量构图。
- `Graph` 的常量是依赖叶子，不进入 `sort_op()` 执行序列。
- `Tensor` 句柄由 Graph 持有；Converter 不得在 Graph 析构后继续使用句柄。

### 2.4 验收标准

| 编号 | 验收项 | 通过条件 |
| --- | --- | --- |
| AC-CONV-01 | 图一致性 | RecordingGraph 与原生 Graph 的 `dump_ir()` 逐字节一致 |
| AC-CONV-02 | Qwen3 计数 | 2 inputs、310 constants、311 个计算 op、1 output（IR `ops` 段为 314，含 3 个 I/O op） |
| AC-CONV-03 | 元数据完整性 | 每个 op 的输入/输出/attrs 均已记录；每个 tensor 有 kind/precision/layout |
| AC-CONV-04 | 权重映射 | 310 个 constant 均有 HF 权重名与形状；tied 别名有记录 |
| AC-CONV-05 | 往返 | `ModelGraph` JSON 保存/加载后 `to_dict()` 相等 |
| AC-CONV-06 | 错误路径 | 非法配置、重复名称、坏属性均抛出 `BuildError` 或 `ValueError` |
| AC-CONV-07 | 输入配置 | 有效配置合并官方 `config.json`/`generation_config.json`；仅声明引擎参数；非法字段、越界与类型错误均被拒绝 |

## 3. 概要设计

### 3.1 总体架构

```text
config.json ─────────────────┐
generation_config.json ──────┤
input/inference_config.json ─┤
                             ▼
                   Qwen3Adapter（合并与校验）
                          │ 创建
                          ▼
   Graph.Graph ◀── RecordingGraph ──▶ ModelGraph（旁路记录）
                          │                        │
                          ▼                        ▼
                   dump_ir()/sort_op()      model_graph.json
                          │                        │
                          └──────────┬─────────────┘
                                     ▼
                              ConversionResult
                       （graph + model_graph + inference_config）
```

Converter 不引入第二套可执行图。`RecordingGraph` 的每个方法先转发给
`Graph.Graph`，成功后再记录元数据；记录失败时抛出异常，避免出现
“图已创建但元数据缺失”的不一致状态。

### 3.2 模块划分

| 文件 | 职责 |
| --- | --- |
| `ToolChain/converter/__init__.py` | 导出 `convert_qwen3`、`ConversionResult`、`RecordingGraph` |
| `ToolChain/converter/recording_graph.py` | 构图代理与元数据记录 |
| `ToolChain/converter/qwen3_adapter.py` | Qwen3 配置加载、builder 调用、权重映射 |
| `ToolChain/converter/inference_config.py` | 运行时输入配置加载、官方 config 合并与校验 |
| `ToolChain/converter/converter.py` | Adapter 注册、转换编排、报告生成 |
| `ToolChain/converter/__main__.py` | 命令行入口 |
| `ToolChain/common/model_ir.py` | `ModelGraph`、`TensorMeta`、`OpMeta`、`IOBinding` |

### 3.3 关键流程

1. 加载并校验 `Qwen3Config`；读取官方 `generation_config.json` 与
   `input/inference_config.json`，合并为有效运行参数。
2. 创建 `Graph.Graph` 与 `RecordingGraph`。
3. 通过注入的 `graph_factory` 让 `Qwen3GraphBuilder` 使用 RecordingGraph。
4. Builder 依次登记常量、输入、计算 op、输出，并调用 `sort_op()`。
5. Adapter 补充权重来源、tied 别名与模型配置元数据。
6. `RecordingGraph.finish()` 生成 `ModelGraph` 并自校验。
7. 可选写 `model_graph.json` 与转换摘要。

### 3.4 技术选型与取舍

| 决策点 | 选择 | 理由 | 代价 |
| --- | --- | --- | --- |
| 图表示 | Graph 单一表示 | 与权威设计一致，避免双表示漂移 | Python 可见元数据少，需旁路记录 |
| 元数据载体 | `ModelGraph` sidecar | 不改 Graph C++ 接口，Optimizer/Serializer 可独立演进 | 需保证记录与图同步 |
| 构图方式 | 复用现有 Qwen3 builder + `graph_factory` 注入 | 不复制层结构，测试覆盖已存在 | Builder 需增加一个可选参数 |
| 权重探测 | safetensors 头信息 | 不加载 1.2 GB 权重即可生成包 | 无法验证数值内容 |
| 序列化中间格式 | 规范化 JSON | 可审查、可 diff、稳定 | 文件体积大于二进制 |

## 4. 详细设计

### 4.1 数据结构

公共数据结构定义在 `ToolChain/common/model_ir.py`：

- `Dim`：维度，`int` 表示静态值，`str` 表示符号（Qwen3 使用 `B`、`S`）。
- `TensorMeta`：`name`、`kind`（input/constant/activation/output）、
  `shape`、`precision`、`layout`、`byte_size`、`pitch`、`data_ptr`、
  `reusable`、`weight_source`、`weight_available`、`quantization`。
- `OpMeta`：`name`、`op_type`（Graph 字符串名）、`inputs`、`outputs`、
  `attrs`、`origin`。
- `IOBinding`：`name`（输出 op 名）、`tensor`（被输出张量名）。
- `ModelGraph`：`name`、`adapter`、`inputs`、`constants`、`outputs`、
  `tensors`（name→TensorMeta）、`ops`（创建顺序）、`metadata`。
- `InferenceConfig`：有效运行参数，包含官方来源字段（context_length、
  vocab_size、do_sample、temperature、top_k、top_p、bos/eos/pad、
  stop_token_ids）与引擎补充字段（max_num_tokens、max_batch_size、
  repetition_penalty、seed）。

### 4.2 接口定义

```python
class RecordingGraph:
    def __init__(self, graph: Any, adapter: str) -> None: ...
    def create_input(
        self, name: str, precision: Any = None, layout: Any = None) -> Any: ...
    def create_constant(
        self, name: str, dims, pitch, precision, layout,
        byte_size: int, data_ptr: int) -> Any: ...
    def create_op(
        self, name: str, src_tensors, out_cnt: int,
        op_type: Any, attr=None) -> list[Any]: ...
    def create_output(self, name: str, tensor: Any) -> None: ...
    def sort_op(self) -> None: ...
    def dump_ir(self) -> str: ...
    def finish(self) -> ModelGraph: ...


def convert_qwen3(
    config_path: Path,
    *,
    weights_path: Path | None = None,
    inference_config_path: Path | None = None,
) -> ConversionResult: ...


def load_inference_config(
    model_dir: Path,
    *,
    config_path: Path,
    vocab_size: int,
    max_position_embeddings: int,
    inference_config_path: Path | None = None,
) -> InferenceConfig | None: ...
```

`ConversionResult` 为冻结数据类：

```python
@dataclass(frozen=True)
class ConversionResult:
    graph: Any
    model_graph: ModelGraph
    adapter: str
    inference_config: InferenceConfig | None
```

### 4.3 算法与流程

#### 4.3.1 输出张量命名

与 `Graph::CreateOp` 保持一致：

- `out_cnt == 1`：`<op_name>.out`；
- `out_cnt > 1`：`<op_name>.out.<index>`。

RecordingGraph 在调用原生接口前生成期望名称，调用后逐项比对原生返回的
`Tensor.name`；不一致时抛出 `BuildError`，防止元数据与图脱节。

#### 4.3.2 op 类型映射

使用 `OpType.name` 的 `k` 前缀转 snake_case 映射，例如
`kLinear → linear`、`kMegaAttention → mega_attention`、
`kLinearSwiGlu → linear_swi_glu`。映射表在模块内显式声明，未知类型抛错。

#### 4.3.3 属性规范化

属性值只允许 `bool/int/float/str` 及同构数组；浮点必须是有限值。
`Storage`、`Tensor` 等非 JSON 值一律拒绝，避免序列化阶段才发现问题。
属性键排序由 `ModelGraph.to_dict()` 统一处理。

#### 4.3.4 权重映射

Qwen3 图内常量名到 Transformers 参数名的映射：

| 图内常量 | Transformers 参数 |
| --- | --- |
| `embedding.weight` | `model.embed_tokens.weight` |
| `layer.{i}.attn_norm.weight` | `model.layers.{i}.input_layernorm.weight` |
| `layer.{i}.q_proj.weight` | `model.layers.{i}.self_attn.q_proj.weight` |
| `layer.{i}.k_proj.weight` | `model.layers.{i}.self_attn.k_proj.weight` |
| `layer.{i}.v_proj.weight` | `model.layers.{i}.self_attn.v_proj.weight` |
| `layer.{i}.o_proj.weight` | `model.layers.{i}.self_attn.o_proj.weight` |
| `layer.{i}.q_norm.weight` | `model.layers.{i}.self_attn.q_norm.weight` |
| `layer.{i}.k_norm.weight` | `model.layers.{i}.self_attn.k_norm.weight` |
| `layer.{i}.mlp_norm.weight` | `model.layers.{i}.post_attention_layernorm.weight` |
| `layer.{i}.gate_proj.weight` | `model.layers.{i}.mlp.gate_proj.weight` |
| `layer.{i}.up_proj.weight` | `model.layers.{i}.mlp.up_proj.weight` |
| `layer.{i}.down_proj.weight` | `model.layers.{i}.mlp.down_proj.weight` |
| `final_norm.weight` | `model.norm.weight` |
| `lm_head.weight`（untied） | `lm_head.weight` |

`tie_word_embeddings=true` 时图中没有 `lm_head.weight` 常量，Adapter 在
`metadata.tied_weights` 中记录 `{"lm_head.weight": "embedding.weight"}`。

#### 4.3.5 权重探测

`weights_path` 可以是单文件或目录：

1. 目录：扫描 `*.safetensors`，按文件名排序；
2. 文件：直接使用；
3. 使用 `ToolChain/common/safetensors.py` 直接解析 8 字节小端头长度与 JSON 头
   （`read_index`），登记张量名、dtype、shape、`data_offsets` 与所属分片；
   支持 F64/F32/F16/BF16/FP8/I64/I32/I16/I8/U8/BOOL，不依赖 torch/numpy；
4. 对每个图常量，若对应 HF 名存在且 dtype、shape 匹配，记录
   `weight_available=true`；不匹配时写入 `weight_issues` 并保持不可用；
5. `weights_path` 未提供或路径不存在时视为未提供权重，全部标记不可用且不报错；
   路径存在但文件损坏或目录内无 `*.safetensors` 时抛出 `BuildError`。

不读取 tensor 数据，只读取头部元信息；序列化阶段才按 `data_offsets`
以流式方式读取权重字节。

#### 4.3.6 元数据自校验

`ModelGraph.validate()` 检查：

- 名称非空且全局唯一（输入、常量、激活、输出张量）；
- 每个 op 的输入、输出张量均存在；
- 每个 op 至少一个输入、一个输出；
- 输出绑定引用存在的张量；
- 至少一个输入与一个输出；
- 从输出出发的 DFS 可覆盖所有计算 op，且无环；
- 常量不出现在 op 的输出中。

#### 4.3.7 运行时输入配置

模型目录下的 `input/inference_config.json` 面向使用引擎运行模型的开发者，
只声明官方 `config.json` 与 `generation_config.json` 没有的引擎参数。
官方已有参数直接读取，不得在新配置中重复声明，避免两处配置漂移。

新配置文件是扁平 JSON 对象，字段固定、全部必填、不接受未知字段：

| 字段 | 类型 | 约束 | 含义 |
| --- | --- | --- | --- |
| `max_num_tokens` | int | > 0 且 <= `max_position_embeddings` | 一次推理 Step 内所有请求累加的 token 预算，绑定符号维 `S` |
| `max_batch_size` | int | > 0 | 规划 batch 上界；Qwen3 首期为 1 |
| `repetition_penalty` | number | > 0 | 重复惩罚 |
| `seed` | int | >= 0 | 随机种子；0 表示由引擎选择非固定种子 |

单请求生成上限不在本文件中配置：运行时按
`context_length - prompt_length` 推导；`max_num_tokens` 是 Step 级预算，
不是单请求生成长度。`max_new_tokens` 属于派生字段，写入即报错。

官方来源字段与读取顺序：

| 有效字段 | 官方来源 | 缺省行为 |
| --- | --- | --- |
| `context_length` | `config.json.max_position_embeddings` | 缺失时使用 `Qwen3Config` 默认值 |
| `vocab_size` | `config.json.vocab_size` | 缺失时使用 `Qwen3Config` 默认值 |
| `bos_token_id` | `generation_config.json.bos_token_id` | 回退 `config.json.bos_token_id`；仍缺失为 null |
| `pad_token_id` | `generation_config.json.pad_token_id` | 回退 `config.json.pad_token_id`；仍缺失为 null |
| `eos_token_id` | `generation_config.json.eos_token_id` | 回退 `config.json.eos_token_id`；支持 int 或 int[] |
| `do_sample` | `generation_config.json.do_sample` | 缺失或 null 时为 false |
| `temperature` | `generation_config.json.temperature` | 缺失或 null 时为 1.0 |
| `top_k` | `generation_config.json.top_k` | 缺失或 null 时为 0（关闭） |
| `top_p` | `generation_config.json.top_p` | 缺失或 null 时为 1.0（关闭） |
| `stop_token_ids` | 由 `eos_token_id` 规范化得到 | 允许为空，此时仅靠上下文窗口上限停止 |

加载与校验规则：

1. 默认路径为 `<config.json 所在目录>/input/inference_config.json`；
   文件不存在时返回 None，`ModelGraph.metadata["inference_config"]=null`，
   旧模型目录仍可转换。
2. 显式传入路径时，相对路径按模型目录解析；文件不存在或 JSON 非法时抛
   `BuildError`，不得静默忽略。
3. 新配置文件字段缺失、未知字段、类型错误（含 bool 冒充 int）、越界或
   `seed` 为负时抛 `BuildError`；`max_new_tokens` 属运行时派生字段，
   写入即报错；所有错误信息包含文件路径与字段名。
4. 官方 `generation_config.json` 存在但字段类型或取值非法时抛
   `BuildError`；文件缺失时按上表缺省行为处理。
5. 合并后的有效配置写入 `ModelGraph.metadata["inference_config"]`，随
   `model.mgf` manifest 的 `graph.metadata` 一起交付；转换报告同时输出。
6. 该配置不参与构图，不改变 op、tensor 或权重映射。

构建默认值联动：

- 流水线未显式传入优化配置时，用有效配置的 `max_batch_size`、
  `max_num_tokens`、`context_length` 初始化 `OptimizeConfig`，使 Step
  token 预算与图的形状规划一致。
- CLI 显式给出的优化参数优先，只覆盖显式字段，其余字段仍取有效配置。
- 采样字段、Token ID 与运行时派生的单请求生成上限只记录，不参与 build。

### 4.4 异常与错误处理

| 场景 | 异常 | 信息 |
| --- | --- | --- |
| 生成器返回的 tensor 名称与预期不符 | `BuildError` | 阶段 `converter`、op 名、期望/实际名称 |
| 未知 OpType | `BuildError` | 枚举名 |
| 属性值非 JSON 安全 | `BuildError` | op 名、属性名、类型 |
| 配置非法 | `ValueError` | 由 `Qwen3Config.validate()` 抛出 |
| 输入配置缺失、字段未知/缺失、类型或范围非法 | `BuildError` | 文件路径与字段名 |
| 官方 `generation_config.json` 非法 | `BuildError` | 文件路径与字段名 |
| 图构造失败 | `RuntimeError` | 由 Graph 抛出，原样传播 |
| 输出路径不可写 | `OSError` | 文件路径 |

错误信息统一包含阶段名与对象名，禁止只输出“转换失败”。

### 4.5 性能与内存

- 记录层只保存名字与小型元数据，不复制 Storage。
- Qwen3 图规模 623 个 tensor、311 个计算 op（IR 中另有 3 个 I/O op）；记录开销为 O(V+E)。
- 权重探测只读 safetensors 头部；不触碰权重数据区。
- 输入配置只读取少量 JSON 字段，合并结果规模固定，不随模型规模增长。
- 转换完成后 `ModelGraph` 与 Graph 同时存活；调用方负责释放。

## 5. 实施步骤

| 阶段 | 任务 | 依赖 | 产出 |
| --- | --- | --- | --- |
| P1 | 实现 `ToolChain/common` 数据类与 JSON 工具 | 无 | `model_ir.py` 等 |
| P2 | 实现 `RecordingGraph` 与代理测试 | P1 | `recording_graph.py` |
| P3 | 给 `Qwen3GraphBuilder` 增加 `graph_factory` 注入 | 无 | Builder 修改 |
| P4 | 实现 Qwen3 Adapter 与权重映射 | P2、P3 | `qwen3_adapter.py` |
| P5 | 实现转换编排与 CLI | P4 | `converter.py`、`__main__.py` |
| P6 | 编写单元测试并修复问题 | P5 | `test_converter.py` |
| P7 | 输出转换报告与测试结果 | P6 | `tests/test_results.md` |
| P8 | 实现输入配置加载与 build 默认值联动 | P5 | `inference_config.py`、pipeline 集成 |

风险与应对：

- **Builder 改动影响既有测试**：`graph_factory` 默认 `None`，行为不变；
  运行 `Models/Qwen3/test_qwen3_graph.py` 回归。
- **Graph 枚举名称变化**：用显式映射表并在未知类型时报错。
- **safetensors 版本差异**：只依赖文件格式规范（8 字节头长度 + JSON 头 +
  数据区），不依赖 Hugging Face 运行时库；遇到未知 dtype 显式报错。
- **官方 config 字段演进**：只读取本设计列出的字段，缺失时按缺省行为处理；
  未知官方字段忽略，不因厂商新增字段中断构建。
- **输入配置与官方配置漂移**：新配置禁止重复声明官方字段；未知字段直接
  报错，避免开发者误以为覆盖生效。

## 6. 测试与验证设计

### 6.1 测试策略

- 单元测试：RecordingGraph 代理一致性、ModelGraph 校验、权重映射、错误路径。
- 集成测试：Qwen3-0.6B 转换，校验图计数与 `dump_ir()`。
- 回归测试：`Models/Qwen3/test_qwen3_graph.py` 全部用例。

### 6.2 用例设计

| 编号 | 用例 | 输入 | 通过标准 |
| --- | --- | --- | --- |
| UT-CONV-001 | 代理一致性 | 小型手工图 | RecordingGraph 与原生 Graph 的 `dump_ir()` 相同 |
| UT-CONV-002 | 输入记录 | `create_input` | kind/precision/layout 正确 |
| UT-CONV-003 | 常量记录 | `create_constant` | dims/pitch/byte_size/data_ptr 正确 |
| UT-CONV-004 | 多输出命名 | `out_cnt=2` | 记录 `name.out.0`、`name.out.1` |
| UT-CONV-005 | 非法属性 | 传入 `object()` | 抛 `BuildError` |
| UT-CONV-006 | 输出绑定 | `create_output` | `ModelGraph.outputs` 正确 |
| UT-CONV-007 | Qwen3 计数 | Qwen3 配置 | 2/310/311/1（IR `ops` 段 314） |
| UT-CONV-008 | 权重映射 | Qwen3 配置 | 310 个常量均有 HF 名与形状 |
| UT-CONV-009 | tied/untied | 两种配置 | tied 记别名；untied 有 `lm_head.weight` |
| UT-CONV-010 | JSON 往返 | 转换结果 | `to_dict()` 相等 |
| UT-CONV-011 | 缺失权重 | 不存在的目录 | 全部 `weight_available=false`，不报错 |
| UT-CONV-012 | 自校验失败 | 构造坏 ModelGraph | `BuildError` 且信息含对象名 |
| UT-CONV-013 | 输入配置合并 | Qwen3 配置与 generation_config | 有效配置含官方采样/eos 与引擎 max token 字段 |
| UT-CONV-014 | 输入配置缺失 | 临时模型目录无 `input/` | `inference_config=None`，转换成功 |
| UT-CONV-015 | 输入配置未知字段 | 新配置含 `temperature` | `BuildError`，提示重复声明 |
| UT-CONV-016 | 输入配置取值越界 | `max_num_tokens` 超上下文 | `BuildError` 且信息含字段名 |
| UT-CONV-017 | 官方 generation_config 非法 | `top_p=2.0` | `BuildError` |
| UT-CONV-018 | 未知字段 | 输入配置含 `extra` | `BuildError` |
| UT-CONV-019 | 官方字段为 null | `temperature=null` 等 | 使用缺省值，转换成功 |
| UT-CONV-020 | 派生字段被配置 | 输入配置含 `max_new_tokens` | `BuildError`，提示由 `context_length - prompt_length` 推导 |
| UT-CONV-021 | build 默认值联动 | 输入配置 max token | `OptimizeConfig` 采用配置值 |

### 6.3 测试环境

- Windows 11 / Linux，Python 3.10+；
- Graph 原生扩展已构建；
- 不需要 GPU、torch 或 transformers；
- 命令：`PYTHONUTF8=1 python -m unittest ToolChain.converter.test_converter -v`。

### 6.4 通过标准

- 全部用例通过，无跳过（可选 safetensors 用例除外）；
- Qwen3 `dump_ir()` 与既有 Qwen3 测试输出一致；
- 转换 JSON 往返一致；
- 测试报告记录实际输出、未覆盖项与风险。
