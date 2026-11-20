# Runtime Loader 模块设计（loader / manifest / ir_dump）

## 1. 背景与目标

### 1.1 背景

Engine 在 `Load` 阶段需要把离线工具链产出的模型包转换成可执行运行时：

- `model.mgf`：112 字节固定头 + canonical JSON manifest；
- `weights.bin`：按 256 字节对齐的常量权重；
- manifest 中的 `graph` 描述构图顺序、属性、IR 哈希与 activation 显存计划。

Runtime 的 Graph/Backend/OpEnvFactory/ResourceStack 契约已冻结；Loader 必须
在不修改这些公共头的前提下实现 R-ARCH-04 与附录 B 的 G1/G2 契约。

### 1.2 目标

1. 通用加载：不写死 Qwen3，所有名称、顺序、形状、属性来自 manifest。
2. 完整性：头部、manifest SHA-256、IR SHA-256、权重逐项 SHA-256 全校验。
3. 资源安全：失败时 result 不修改，已创建资源经 ResourceStack 逆序释放。
4. 可验证：C++ `DumpIr` 与 Python `dump_ir` 逐字节一致（G1）。
5. 可测试：提供 `loader_smoke`，覆盖成功、metadata-only 与主要失败路径。

### 1.3 非目标

- 不实现 Scheduler/Executor/KVCache/Destroy（其它终端负责）。
- 不执行 Forward、不做数值精度比较。
- 不支持多线程加载、异步上传或模型包写入。
- 不实现 OCL/Vulkan 后端；只依赖 `Backend` 抽象与 BackendRegistry。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收标准 |
| --- | --- | --- |
| R-LD-01 | 读 112B 头（`<8sIIQQ32sQQ32s`），校验 magic/version/flags/边界/SHA | 损坏头返回 kInvalidModelFile |
| R-LD-02 | 解析 manifest JSON，构建张量/算子/权重/显存计划/统计视图 | Qwen3 manifest 全量通过 |
| R-LD-03 | 按 manifest 顺序重建 Graph：constants → inputs → ops → outputs → SortOp | 输出名与 manifest 完全一致 |
| R-LD-04 | `sample` 走 `CreateSample`，其余走 `CreateOp` | 唯一 sample op、`<name>.out` |
| R-LD-05 | 绑定 data_ptr 之前计算 C++ `DumpIr` SHA-256 并与 `graph.ir_sha256` 比对 | 不一致返回 kInvalidModelFile |
| R-LD-06 | 按 `LoadConfig.runtime.backend` 经 BackendRegistry 创建并 Init 后端 | 未知后端返回 kUnsupportedBackend |
| R-LD-07 | 能力校验：算子/精度/H2D/D2H/对齐/单缓冲上限 | 缺失返回对应错误码 |
| R-LD-08 | 权重按 entries 流式读取、可逐项 SHA-256、`ConvertParam` 上传 | 权重字节数与常量 storage 一致 |
| R-LD-09 | 常量 Storage 绑定 `Buffer*`，offset=0，device_type=kDevice | Mock Kernel 可正确取址 |
| R-LD-10 | 按 memory_plan 分配单个 activation arena 并绑定每个 activation | 条目覆盖全部 activation |
| R-LD-11 | 输入分配设备 Buffer；输出源张量复用 activation arena | I/O op 初始化通过 |
| R-LD-12 | 创建 GraphEnv，逐 Input/Output/Orderly op 建 OpEnv 并 SetOpEnv | 全部 op 挂载成功 |
| R-LD-13 | `GraphEnv::SharedStorages()` 统一分配 workspace 并回填 data_ptr | 无 workspace 时为空操作 |
| R-LD-14 | 解析 ModelConfig，dtype 使用 `optimization.config.target_precision` | Qwen3 得到 float16 而非 bf16 |
| R-LD-15 | 返回 Graph/Backend/Model/weights_bytes/arena_bytes/io_bytes | 统计与 manifest 一致 |
| R-LD-16 | 失败时 result 不修改，资源登记 ResourceStack 逆序释放 | 负向用例 rollback 成功 |

### 2.2 非功能需求

| 编号 | 需求 |
| --- | --- |
| R-LD-NF-01 | C++17、Google 风格、中文注释、类型别名 U8/I32/U64/Usize/C8 |
| R-LD-NF-02 | 不修改 Engine/Scheduler/KVCache/Executor/Destroy/Graph/Backend 公共头 |
| R-LD-NF-03 | 单线程加载；不引入第三方依赖 |
| R-LD-NF-04 | 1.19 GB 权重全量校验 + 上传在 10 s 量级完成（本机实测 ~8.7 s） |
| R-LD-NF-05 | 失败必须可回滚，不留下半初始化状态 |

### 2.3 约束条件

- 固件格式由 `ToolChain/serializer/format.py` 定义，Loader 只读不写。
- Graph 的 `SortOp` 会把 Input op 也纳入拓扑序（input tensor 有 producer），
  因此 `GetOrderlyOPList()` 含 Input/Output op，OpEnv 创建需去重。
- Kernel OpEnv 在 `Init` 时要求所有 src/dst storage 已绑定 data_ptr，
  因此缓冲绑定必须早于 OpEnv 创建。
- Mock Backend 的 `ConvertParam` 只支持 FP16 目标与 BF16/FP32/FP16 源。

### 2.4 验收标准

- AC-01：Qwen3-0.6B 包全量加载成功，IR/weights 哈希全部通过。
- AC-02：`loader_smoke` 打印 310 常量、315 拓扑 op、312 activation、
  1,192,099,840 权重字节、78,317,568 arena 字节、4,096 I/O 字节。
- AC-03：损坏包按预期错误码拒绝，且可安全回滚。
- AC-04：`-Wall -Wextra` 零警告。
- 测试记录见 `Runtime/loader/tests/test_results.md`。

## 3. 概要设计

### 3.1 模块划分

| 文件 | 职责 | 关键依赖 |
| --- | --- | --- |
| `manifest.h/.cc` | 112B 头解析、manifest 视图与校验、权重表、memory plan、ModelConfig、名称转换 | common/json、common/file_util、Graph/attr |
| `ir_dump.h/.cc` | C++ `DumpIr`、IR SHA-256、行数统计 | Graph 查询接口、common/sha256 |
| `loader.cc` | `CreateDefaultLoader` 与完整加载编排 | manifest、ir_dump、BackendRegistry、OpEnvFactory、ResourceStack |
| `tests/loader_smoke.cc` | 真实包冒烟与绑定/统计检查 | loader、manifest、file_util |

### 3.2 加载流程

```text
ILoader::Load(config, stack, result)
  1  ReadModelPackageHeader(model.mgf)           → 校验 magic/version/flags/边界
  2  ReadFileRange(manifest) + SHA-256           → 与 header.manifest_sha256 比较
  3  ParseJson → ParseManifest                   → 引用/统计/权重/计划校验
  4  ParseModelConfig                            → ModelConfig（float16）
  5  weights.bin 存在性/大小校验                 → metadata-only 例外
  6  RebuildGraph + SortOp                       → 命名逐项校验
  7  DumpIr + SHA-256                            → 与 graph.ir_sha256 比较（绑指针前）
  8  BuildTensorIndex + GraphCoverage            → 只使用 Graph 公共查询接口
  9  预填 input/activation Storage 元数据        → 不改 data_ptr
 10  BackendRegistry::Create → Init → Capability → 能力/对齐/预算校验
 11  权重流式读取 + 可选 SHA + ConvertParam     → 常量绑定 Buffer*
 12  metadata-only 占位缓冲（仅在需要时）        → 禁止执行
 13  输入设备 Buffer 绑定                        → io_bytes
 14  activation arena 分配与逐条绑定             → arena_bytes
 15  GraphEnv + 逐 op OpEnv（Input/Orderly/Output 去重）
 16  SharedStorages workspace 分配回填
 17  移交 Graph/Backend/Model 与统计
```

### 3.3 资源所有权

| 资源 | 创建者 | 持有者 | 释放路径 |
| --- | --- | --- | --- |
| Graph | Loader | LoadResult → Engine | Engine 析构/Graph 析构清 OpEnv |
| Backend | Loader | LoadResult → Executor | Executor::Deinit；失败时堆栈闭包 Deinit+delete |
| 权重/arena/IO/workspace Buffer | Loader | ResourceStack 闭包（shared_ptr<unique_ptr<Buffer>>） | `Clear()` → `Backend::DestroyBuffer` |
| OpEnv/Kernel | OpEnvFactory | Op（unique_ptr） | `Op::ClearOpEnv` 或 Graph 析构 |
| GraphEnv | Loader | Graph（unique_ptr） | Graph 析构 |

失败回滚时，Loader 局部 `unique_ptr<Graph>` 先析构（清 OpEnv），随后
Engine 调用 Destroyer 清 ResourceStack（缓冲 → 后端 Deinit），二者顺序与
正常 Destroy 一致。

### 3.4 错误码映射

| 场景 | 错误码 |
| --- | --- |
| 文件缺失/短读 | kIoError |
| magic/version/flags/边界/manifest SHA/IR SHA/JSON/字段/引用/权重尺寸 | kInvalidModelFile |
| 后端未注册 | kUnsupportedBackend |
| Backend::Init 失败 | 透传（kBackendInitError/kDeviceError 等） |
| 能力缺少算子 | kUnsupportedOp |
| 能力缺少精度 | kUnsupportedDtype |
| 缺少 H2D/D2H 或对齐非法 | kUnsupportedBackend |
| 单缓冲超限 | kBufferError |
| 预算不足 | kDeviceOom |
| 参数为空 | kInvalidArgument |

## 4. 详细设计

### 4.1 固定头

字段顺序（小端）：

| 偏移 | 类型 | 含义 |
| --- | --- | --- |
| 0 | 8s | magic `"MNETMGF\0"` |
| 8 | I | version（必须 1） |
| 12 | I | flags（只允许 bit0=weights） |
| 16 | Q | manifest_offset（≥112） |
| 24 | Q | manifest_length（>0 且 offset+length ≤ 文件大小） |
| 32 | 32s | manifest SHA-256 |
| 64 | Q | weights_offset（有 weights 时 ≥112） |
| 72 | Q | weights_length |
| 80 | 32s | weights SHA-256 |

校验：flags 与 weights_length 一致性、file_size 上界、weights flag 与长度
互推。头部解析不依赖结构体打包，逐字段小端读取，避免对齐/填充差异。

### 4.2 manifest 视图与校验

- 顶层：`format == "MobileNet.ModelPackage"`、`format_version == 1`；
  `model/graph/optimization/weights/statistics` 必需。
- 张量：静态整数维度（symbol 维度拒绝）；rank ≤ 8；pitch 为空或与 shape
  等长；精度/布局名称必须在稳定名称表内；名称唯一。
- op：名称唯一、`op_type` 可映射、不得为 input/output/unknown、I/O 非空、
  输出必须是 activation 且唯一 producer、输入/输出引用存在的张量。
- 引用：Input/Constant 不得被 op 生产；activation 必须有 producer；
  输出绑定必须指向有 producer 的 activation；名称不得跨 inputs/constants。
- 权重：条目名唯一、available 条目 offset 对齐/在 total_bytes 内、
  byte_size > 0、sha256 为 64 位十六进制（可空）；每个常量恰好一个条目；
  `weight_available` 与 `available` 一致；条目 shape 与张量 shape 一致；
  `byte_size == elements × dtype 字节数`。
- 显存计划：alignment > 0 且为 2 的幂（Loader 再校验）；条目名唯一、
  offset+size ≤ arena_bytes、offset 对齐、birth ≤ last_use、size 与张量
  byte_size 一致、每个 activation 恰好一个条目。
- 统计：张量/op/常量/输出/activation 计数与常量字节数必须与实际一致；
  `optimization.config.alignment` 与 `memory_plan.alignment` 一致。
- `weights.file` 只允许简单文件名（禁止 `/`、`\`、`.`、`..`），避免路径穿越。

`TensorMeta` 中 `pitch` 为空表示与 dims 相同（与 ToolChain `MakeStorage`
一致），不能误判为长度不一致。

### 4.3 图重建与 IR 校验

重建顺序严格复刻 Python `rebuild_graph_from_model`：

1. 常量：`Storage{shape=dims, pitch=dims 或 meta.pitch, precision,
   layout, device=kHost, byte_size, data_ptr=nullptr, reusable=true}`，
   再 `CreateConstant`（内部把 reusable 置 false）。
2. 输入：`CreateInput(name, precision, layout)`（只写 op attr）。
3. op：`CreateOp(name, src, outputs.size(), type, attrs)`；
   `sample` 使用 `CreateSample(name, src, attrs)`。
4. 每个 op 创建后逐项比较 `DstTensors()` 名称与 manifest `outputs`。
5. `CreateOutput(binding.name, binding.tensor)`。
6. `SortOp`。

随后立即：

```cpp
ir = DumpIr(graph);
sha = Sha256::Hex(ir);
if (sha != manifest.graph.ir_sha256) return kInvalidModelFile;
if (ir_line_count > 0 && CountIrLines(ir) != ir_line_count) error;
```

该步骤必须在填充 input/activation Storage 与绑定 data_ptr 之前，因为
`StorageToString` 输出 `device/data/offset/bytes`，绑定后文本会变化。

### 4.4 张量索引

禁止依赖构建期私有 map 做运行期查找。`SortOp` 后统一从 Graph 公共接口
建立 `name → Tensor*`：

- `GetAllConstantTensor()`；
- `GetAllInputOp()` 的 `DstTensors()`；
- `GetOrderlyOPList()` 的 `DstTensors()`。

再用 manifest 做双向覆盖校验：manifest 每个非 output 张量必须在索引中且
kind 匹配；索引中每个张量必须在 manifest 中。常量顺序、输入顺序也必须
与 manifest 一致（影响 IR 契约）。

### 4.5 能力校验语义

- 目标 dtype（ModelConfig.dtype）必须出现在 `supported_dtypes`。
- `h2d` 与 `d2h` 必须同时为 true（Input/Output IoOpEnv 依赖）。
- `GetOrderlyOPList()` 中除 Input/Output 外的每个 op 必须在
  `supported_ops` 中。
- 精度：常量与 activation 中的浮点精度必须被支持；整数/布尔张量只用于
  索引、采样与 I/O，不参与浮点计算能力判断（避免 int64 I/O 误报）。
- 对齐：`max(memory_plan.alignment, min_alignment)`，两者都必须为 2 的幂。
- 单缓冲上限：权重目标字节、arena、输入、workspace 均检查
  `max_buffer_bytes`（0 表示不限制）。
- 预算：`memory_budget_bytes > 0` 时检查 weights+arena+io 不超过预算。

### 4.6 权重上传

- `weights.bin` 存在时要求大小同时等于头部 `weights_length` 与 manifest
  `total_bytes`；`verify_weights_sha256` 打开时先流式哈希整个文件与头部
  `weights_sha256` 比较。
- 每个 available 条目：
  1. `FileReader::ReadAt(offset, byte_size)`（内部 8 MB 分块）；
  2. 可选逐项 SHA-256 与 `entry.sha256` 比较；
  3. `ParamDesc{source_dtype, target_dtype=model.dtype}` → `ConvertParam`；
  4. 校验 `buffer->Bytes()` 等于目标字节数且等于常量 storage.byte_size；
  5. `Storage.data_ptr = Buffer 对象地址`（不是 `DevicePtr()`），
     `offset=0`，`device_type=kDevice`；
  6. 释放闭包捕获 `shared_ptr<unique_ptr<Buffer>>` 与后端状态，
     调用 `Backend::DestroyBuffer`。

Mock/CPU Kernel 通过 `static_cast<Buffer*>(view.data)` 取回 Buffer 对象，
因此绑定对象地址是契约要求。

### 4.7 activation arena 与 I/O

- `memory_plan.arena_bytes > 0` 时分配一个 `BufferUsage::kActivation` 缓冲；
  每条计划将 activation 的 `data_ptr` 指向 arena Buffer 对象、`offset` 为
  条目偏移、`byte_size` 为条目大小。
- 输入张量按 manifest byte_size 分配 `BufferUsage::kIo`，绑定 offset=0。
- 输出绑定（Output op 的源张量，如 `sample.out`）是 activation，已由
  arena 绑定；`IoOpEnv::Init` 只要求其 data_ptr 非空。
- 所有输入/activation Storage 的 shape/pitch/precision/layout/byte_size
  从 manifest 预填，保证 `KernelOpEnv::Init` 的 `CheckTensor` 通过。

### 4.8 GraphEnv 与 OpEnv

1. `graph->SetGraphEnv(make_unique<GraphEnv>())`。
2. `CreateDefaultOpEnvFactory()`（ExecBackend 实现）。
3. 按 `GetAllInputOp` → `GetOrderlyOPList` → `GetAllOutputOp` 顺序创建，
   用 `std::set<Op*>` 去重（Input/Output 也在拓扑序中）。
4. 每个 OpEnv `Init` 成功后立即 `Op::SetOpEnv`。
5. 遍历 `GraphEnv::SharedStorages()`：byte_size>0 且 data_ptr 为空的条目
   分配 `BufferUsage::kWorkspace` 缓冲并回填 `data_ptr`/offset/device。
   Qwen3 当前没有 workspace op，该路径为空操作。

### 4.9 ModelConfig

| 字段 | 来源 |
| --- | --- |
| num_layers | `metadata.model_config.num_hidden_layers` |
| num_heads / num_kv_heads / head_dim | `num_attention_heads` / `num_key_value_heads` / `head_dim` |
| hidden_size / intermediate_size / vocab_size | 同名整数 |
| rms_norm_eps / rope_theta | 同名数字（缺省 0） |
| context_length | `inference_config.context_length` → `optimization.config.context_length` → `max_position_embeddings` |
| max_num_tokens | `inference_config.max_num_tokens` → `optimization.config.max_num_tokens` → 0 |
| max_batch_size | `inference_config.max_batch_size` → `optimization.config.batch_size` → 1 |
| pad/bos/stop ids | `inference_config` |
| dtype | `optimization.config.target_precision`（Qwen3 为 float16，不取官方 bf16） |

结构字段必须为正；context_length 必须为正；max_num_tokens 不得为负。

### 4.10 metadata-only

- 头部无 weights 或 `weights.bin` 缺失时，只有
  `RuntimeConfig.allow_metadata_only == true` 才允许继续。
- 未绑定的常量统一绑定一个 64 字节占位缓冲，`weights_bytes` 记为 0。
- 该模式仅用于加载期校验与冒烟；占位缓冲远小于真实权重，一旦执行推理会
  越界读取，Loader 在文档与 smoke 输出中明确标注。

### 4.11 失败与回滚

- 所有资源（后端、权重、arena、IO、workspace 缓冲）在创建时立即 Push 到
  ResourceStack；后端闭包最先 Push，LIFO 时最后执行。
- 失败返回前不写 `result`；局部 Graph 析构清 OpEnv；Engine rollback 通过
  Destroyer 清栈：缓冲先释放，后端 Deinit 最后执行。
- 成功时后端所有权移入 `LoadResult.backend`，闭包内 owned 为空、只保留
  raw 指针供缓冲释放使用；正常 Destroy 由 Executor::Deinit 释放后端。

## 5. 实施步骤

| 阶段 | 任务 | 依赖 | 风险 |
| --- | --- | --- | --- |
| S1 | sha256/json/file_util | 无 | 数字类型/偏移错误 |
| S2 | manifest.h/.cc 头部与视图 | S1 | manifest 字段假设过强 |
| S3 | ir_dump.h/.cc 与 Python 对齐 | Graph 库 | 格式漂移 |
| S4 | loader.cc 重建/后端/权重/缓冲/OpEnv | S2、S3、Backend | 绑定顺序错误 |
| S5 | loader_smoke + 负向包 | S4 | 大文件耗时 |
| S6 | 设计文档、skill、测试结果 | S1–S5 | — |

## 6. 测试与验证设计

### 6.1 测试策略

- 真实包正向：全量校验加载 + metadata-only 加载。
- 负向：坏 magic、坏 JSON、坏 IR 哈希、权重尺寸、OpEnv 失败回滚、
  未知后端、缺失目录。
- 交叉验证：Python `dump_ir` 哈希与 manifest、C++ `DumpIr` 哈希三者一致。
- 编译：全部新文件 `-Wall -Wextra` 零警告，whole-archive 链接 Mock 后端。

### 6.2 用例与追溯

| 用例 | 输入 | 通过标准 | 追溯 |
| --- | --- | --- | --- |
| T1 | Qwen3 完整包 | 加载成功、统计匹配、release OK | R-LD-01..15 |
| T2 | 仅 model.mgf（metadata-only） | 加载成功、weights_bytes=0 | R-LD-08/16 |
| T3 | 坏 magic | kInvalidModelFile | R-LD-01 |
| T4 | 尾部 `}`→`]` | JSON 错误含 offset | R-LD-02 |
| T5 | ir_sha256 全 0 | kInvalidModelFile | R-LD-05 |
| T6 | 100 字节 weights.bin | kInvalidModelFile | R-LD-08 |
| T7 | gather axis=1（IR 重算） | OpEnv 失败且 rollback OK | R-LD-12/16 |
| T8 | 未注册后端 | kUnsupportedBackend | R-LD-06 |
| T9 | 目录不存在 | kIoError | R-LD-01 |

### 6.3 未覆盖项与风险

- 未执行 `Forward`，权重数值/布局正确性由后端与 e2e 测试覆盖。
- workspace 分配路径在本模型为空操作，未做实测。
- BF16/FP32→FP16 权重转换路径未由该包触发（包内已是 FP16）。
- 非 Mock 后端（OCL）未验证；能力语义以 Mock 声明为准。
- 正常 Destroy 依赖 Destroyer 先清 OpEnv、再清 ResourceStack、最后
  `Executor::Deinit`；顺序错误会导致悬空指针（设计文档已冻结该顺序）。
- 跨模块生命周期边界：Engine 在 `IExecutor::Init` 失败时可能先释放
  Backend，再以 `backend=null` 回滚清 ResourceStack；Loader 的缓冲闭包
  保存 Backend 裸指针，此时存在悬空风险。建议 Executor::Init 失败时保留
  Backend 到 Destroyer 清栈之后，或 Destroyer 在该路径直接析构 Buffer。

### 6.4 测试环境与记录

- Windows 11、zig clang 22（`x86_64-windows-gnu`）、C++17、-O2。
- 实际命令、完整输出与结论见 `Runtime/loader/tests/test_results.md`。
