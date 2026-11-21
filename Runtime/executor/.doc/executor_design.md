# Executor 模块设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.2 |
| 文档状态 | 实现完成；v0.2 增加阶段一 Paged Attention 元数据装配与上传 |
| 上游设计 | `Runtime/.doc/Runtime_design.md` v0.6 |
| 模块路径 | `Runtime/executor/` |

## 1. 背景与目标

### 1.1 背景

Runtime 的单步执行链路为
`Engine → IExecutor → OpEnv::Forward → Kernel → Backend`。接口层
`Runtime/executor/executor.h` 已冻结 `IExecutor` 与
`CreateDefaultExecutor()`，但 `Runtime/executor/` 下没有实现文件，Engine 的
默认加载路径无法完成一步推理。

本模块补齐默认实现（内部类 `ExecBackend`），把 Scheduler 产出的
`ExecutionPlan`、`KvCacheView` 转换为：

- Graph Input op 的主机 token/position 缓冲；
- 每步 `StepContext`；
- 按拓扑序的 `OpEnv::Forward` 调用；
- 从 `sample.out` 回读的 `StepResult`。

### 1.2 目标

| 编号 | 目标 |
| --- | --- |
| GOAL-01 | 实现 `IExecutor` 的 `Init/Execute/Deinit` 与 `CreateDefaultExecutor()` |
| GOAL-02 | Init 接管 Backend，并按 Graph 的 Input/Output op 预分配主机 I/O 缓冲 |
| GOAL-03 | Execute 把 plan 打包为 token/position 输入，经 IoBinder 绑定主机 Storage |
| GOAL-04 | Execute 注入 `StepContext`，按 `GetOrderlyOPList()` 驱动全部 OpEnv |
| GOAL-05 | 同步后回读 sample 输出，按 `output_slot` 填充 `StepResult` |
| GOAL-06 | 错误路径返回明确错误码，Deinit 幂等并释放 Backend |
| GOAL-07 | 不写死 Qwen3 或其他模型，形状/字节数全部来自 Graph Storage |
| GOAL-08 | KV 模式下装配 `AttentionMetadataView`（7 个数组 + stride），每步一次 H2D 到 Executor 私有元数据 Buffer |

### 1.3 非目标

- 不实现调度、KV 分配/释放、抢占、停止判定或 `Response` 构建；
- 不创建/销毁 OpEnv、Kernel 或 Buffer；这些由 Loader/Engine/Manager 负责；
- 不实现 H2D/D2H 本身；传输由 `IoOpEnv` 调用 `Backend::Upload/Download`；
- 不实现 paged attention 数学、块分配或抢占；只装配执行元数据；
- 不实现采样算法；`sample` 由后端 Kernel 完成，本模块只回读结果；
- 不覆盖多模型专有输入（例如多模态、LoRA、投机解码 draft token）。

### 1.4 范围与约束

- 代码只位于 `Runtime/executor/`；
- 不修改 Engine/Loader/Scheduler/KVCache/Backend/Graph 的公共头文件；
- Host 端类型统一使用 `Utils/cpp/types.h` 别名；
- 遵循 Google C++ Style 与 `memory.md` 的中文注释要求。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| REQ-EX-01 | `Init` 接管 `std::unique_ptr<Backend>`，保存 Graph/IoBinder 借用指针；失败时释放已接管的 Backend | TC-EX-01/02 |
| REQ-EX-02 | `Init` 按 Input/Output op 的 tensor Storage 预分配主机缓冲，复制 shape/precision/layout/byte_size | TC-EX-03 |
| REQ-EX-03 | `Execute` 约定 `GetAllInputOp()` 第 1 个为 token_ids，第 2 个为 position_ids；只有 1 个时仅填 token | TC-EX-04/10 |
| REQ-EX-04 | token_ids/positions 按 `PlanEntry.packed_offset` packed 顺序写入，未覆盖位置补 0 | TC-EX-04/05 |
| REQ-EX-05 | 校验 packed token 总数不超过主机缓冲元素数 | TC-EX-08 |
| REQ-EX-06 | 每个 Input op 构造 host Storage（`kHost`、`data_ptr`=主机缓冲、shape/precision/layout/byte_size 复制设备），调用 `IoBinder::SetInput` | TC-EX-04 |
| REQ-EX-07 | 每个 Output op 构造 host Storage 并调用 `IoBinder::SetOutput` | TC-EX-04 |
| REQ-EX-08 | 构造 `StepContext`：backend、plan、kv_view、`attention_meta`、batch、packed_tokens；`seq_lens` 由 `context_length+token_count` 推导，不再单独传裸指针 | TC-EX-04/05/14 |
| REQ-EX-09 | 按 `GetOrderlyOPList()` 依次 `SetStepContext(env,&ctx)`、`Forward()`、`SetStepContext(env,nullptr)` | TC-EX-04/05 |
| REQ-EX-10 | 全部 op 后执行 `Backend::Synchronize`；从 `GetSampleOp()->DstTensors()[0]` 的 Storage 读取 sample | TC-EX-04 |
| REQ-EX-11 | sample Storage 的 `data_ptr` 为 `Buffer*`，通过 `Backend::DownloadAt(host, DevicePtr(), offset, bytes)` 回读到 `vector<I64>`，大小 `byte_size/8`；不得对设备句柄做指针加法 | TC-EX-04 |
| REQ-EX-12 | 按 `output_slot` 从 sample 取 token；`output_slot<0` 跳过；填 `Token{id,0,0,true}`、`ok=true` | TC-EX-04/05/06 |
| REQ-EX-13 | 缺失 OpEnv、I/O 绑定失败、尺寸不匹配返回明确错误码 | TC-EX-07/08/09/11 |
| REQ-EX-14 | `Deinit` 清空缓冲，调用 `Backend::Deinit` 并释放；幂等；未 Init 调用返回成功 | TC-EX-02/12 |
| REQ-EX-15 | 不写死模型/形状；所有尺寸来自 Graph Storage | 代码评审 |
| REQ-EX-16 | 由 `PlanEntry` 装配 `seq_lens/query_lens/context_lens/request_indices/positions/slots/block_table`；`seq_lens=context_length+token_count` | TC-EX-14 |
| REQ-EX-17 | 元数据写入 Executor 私有 Buffer，采用高水位增长；steady-state 不逐步分配 | TC-EX-14/15 |
| REQ-EX-18 | 每步只做一次元数据 H2D；各数组按 16 字节对齐，`TensorView.offset` 正确 | TC-EX-14 |
| REQ-EX-19 | 混合 batch：同一 plan 内 prefill chunk 与 decode 混排；`request_indices` 逐 token 正确 | TC-EX-15 |
| REQ-EX-20 | 校验 `slots.size()==token_count`、块表覆盖 `ceil(seq_lens/block_size)`、packed 区间不重叠 | TC-EX-16 |

### 2.2 非功能需求

| 编号 | 需求 |
| --- | --- |
| NFR-EX-01 | C++17，Google Style，新增代码通过 `-Wall -Wextra` 零告警 |
| NFR-EX-02 | 所有模块/类/函数具备中文注释，覆盖功能边界、输入输出与主要逻辑 |
| NFR-EX-03 | 正常运行路径不抛异常，全部可预期错误通过 `Status` 返回 |
| NFR-EX-04 | 主机 I/O 缓冲在 Init 一次分配、每步复用，避免逐步分配大块内存 |
| NFR-EX-05 | 仅允许单推理线程调用，内部不加锁、不做线程安全承诺 |
| NFR-EX-06 | 不依赖后端专有头文件，公共头不出现设备类型 |

### 2.3 约束条件

- Graph Storage 的 `data_ptr` 在设备侧是 `Buffer*`，不是裸设备地址；
  Buffer 基址为 `static_cast<Buffer*>(data_ptr)->DevicePtr()`，`offset` 是缓冲内
  字节偏移。带偏移的传输必须调用 `Backend::UploadAt/DownloadAt`；`cl_mem`
  等句柄型设备指针不能通过 `DevicePtr() + offset` 表达偏移。
- `IoOpEnv::Forward` 从 op 的 `attr["storage"]` 读取主机 Storage，并且要求
  `device_type=kHost`、`byte_size` 与精度和设备视图一致。
- `KernelOpEnv::Forward` 要求 `StepContext` 非空；未设置时返回 `kStateError`。
- Engine 在 Load 中把 Backend 移交给 Executor，Executor 必须在 Destroy
  阶段晚于所有 OpEnv 释放 Backend。
- 元数据 Buffer 由 Executor 通过 `Backend::CreateBuffer` 创建并在 `Deinit`
  中先于 `Backend::Deinit` 释放；不属于 Graph arena 或 `ResourceStack`。
- 元数据数组布局固定为 `seq_lens | query_lens | context_lens |
  request_indices | positions | slots | block_table`，每段 16 字节对齐；
  段内元素类型与 `AttentionMetadataView` 一致。
- `recompute_attention=true` 时 `attention_meta` 为空，Executor 不做 KV
  元数据上传；两种模式不能混用。

### 2.4 验收标准

1. `Runtime/executor/executor.cc` 与测试在 zig clang C++17 下编译、
   链接零告警零错误；
2. `executor_test` 全部用例 PASS，退出码 0；
3. 成功路径中 token/position 缓冲内容、绑定 Storage 字段、sample 回读、
   `StepResult` 与错误码均被断言；
4. 四件套（设计文档、代码、测试结果、skill）齐备。

## 3. 概要设计

### 3.1 总体架构

```text
Engine (IoBinder 实现)
   │  Init(std::move(backend), graph, this)
   │  Execute(plan, kv_view, &step_result)
   ▼
IExecutor / ExecBackend
   │  持有 Backend；缓存 HostIoSlot；构造 StepContext
   ▼
OpEnv (Graph 持有)
   ├─ IoOpEnv       : attr["storage"] → Backend::Upload/Download
   └─ KernelOpEnv   : StepContext + TensorView → Kernel::Execute
                      │
                      ▼
                    Backend (Mock/OCL/CUDA/Vulkan)
```

### 3.2 模块划分

| 单元 | 职责 |
| --- | --- |
| `HostIoSlot` | 单个 Input/Output op 的名称、主机 Storage 模板、预分配缓冲 |
| `ResolveByteSize` | 由 `byte_size` 或 shape×precision 推导有效字节数 |
| `BuildHostIoSlot` | 校验边界 tensor 并构造主机槽 |
| `CheckPackedInputSlot` | 校验 token/position 输入为 int64 且字节数可整除 8 |
| `FillPackedI64` | 清零并按 packed 偏移写入 I64 列 |
| `ValidatePlan` | 校验 entry 数量、token_ids/positions 长度、packed 范围与总数 |
| `MakeHostStorage` | 由主机槽生成传给 IoBinder 的 Storage |
| `ExecBackend` | `IExecutor` 默认实现 |

### 3.3 关键流程

**Init**

```text
校验参数与重复 Init
→ 校验 Input/Output 列表非空、Sample op 存在且恰有 1 个输出
→ 逐 Input op 构造主机槽（复制 Storage + 分配缓冲）
→ 逐 Output op 构造主机槽
→ 提交 backend_/graph_/io_binder_/槽（全部成功后才提交）
```

**Execute**

```text
校验 result / 初始化状态
→ ValidatePlan（entries 非空、长度一致、total 一致、packed 范围合法）
→ 校验前两个输入为 int64
→ 打包 token/position（清零 + packed 写入）
→ SetInput/SetOutput（host Storage）
→ KV 模式：装配元数据并一次 Upload；构造 StepContext 与 attention_meta
→ for op in GetOrderlyOPList():
     SetStepContext(env,&ctx) → Forward() → SetStepContext(env,nullptr)
→ Backend::Synchronize
→ DownloadAt sample.out（Buffer* + offset）到 vector<I64>
→ 按 output_slot 填充 StepResult
```

**Deinit**

```text
Backend::Deinit（若存在）→ 释放 Backend → 清空 graph_/io_binder_/槽
```

### 3.4 技术选型与取舍

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 主机缓冲分配时机 | Init 一次分配，Execute 复用 | 减少每步分配；尺寸由静态图 Storage 确定 |
| token 数据搬运 | `std::memcpy` 逐元素 | 避免 `vector<U8>` 转 `I64*` 的对齐未定义行为 |
| 失败处理 | 返回 `Status`，`StepResult` 保持空 | 与 Runtime 其他模块一致；Engine 会按错误码失败全部请求 |
| sample 回读 | 显式 `Download` 到 `vector<I64>` | sample.out 不一定是 Output op 的源张量，不能依赖输出绑定 |
| 上下文生命周期 | `Execute` 栈上构造，逐 op 注入后立即清空 | 符合 `StepContext` 单步有效约定 |
| 模型耦合 | 无 | 输入顺序是唯一约定，其余全部来自 Graph |

## 4. 详细设计

### 4.1 对外接口

本模块不新增公共头，`executor.h` 保持不变：

```cpp
class IExecutor {
 public:
  virtual Status Init(std::unique_ptr<Backend> backend,
                      runtime_graph::Graph* graph, IoBinder* io_binder) = 0;
  virtual Backend* backend() = 0;
  virtual Status Execute(const ExecutionPlan& plan,
                         const KvCacheView& kv_view,
                         StepResult* result) = 0;
  virtual Status Deinit() = 0;
};
std::unique_ptr<IExecutor> CreateDefaultExecutor();
```

### 4.2 数据结构

`HostIoSlot`（内部）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | `std::string` | op 名称，作为 SetInput/SetOutput 的键 |
| `storage` | `runtime_graph::Storage` | 主机模板：shape/precision/layout/byte_size 来自设备；`device_type=kHost`、`offset=0`、`data_ptr=nullptr`、`reusable=false` |
| `buffer` | `std::vector<U8>` | 大小等于 `storage.byte_size`，Init 时清零 |

输入槽顺序严格等于 `Graph::GetAllInputOp()`；输出槽顺序等于
`Graph::GetAllOutputOp()`。

`AttentionMetadataHost`（内部，主机暂存）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `seq_lens/query_lens/context_lens` | `std::vector<I32>` | 长度 batch |
| `request_indices/positions` | `std::vector<I32>` | 长度 token_count |
| `slots` | `std::vector<I64>` | 长度 token_count |
| `block_table` | `std::vector<I32>` | 长度 batch × stride，尾部补 0 |
| `block_table_stride` | `I32` | 本步最大块数 |

`MetadataCapacity`（内部）：记录已分配的 batch/token/stride 上限与
`std::unique_ptr<Buffer>`；当本步需求超过上限时重新 `CreateBuffer` 并更新，
否则复用。

### 4.3 Init 详细流程

1. 若 `backend_` 非空，返回 `kStateError`（拒绝重复 Init）。
2. `backend/graph/io_binder` 任一为空 → `kInvalidArgument`。
3. Input 列表为空 → `kUnsupportedOp`；Output 列表为空 →
   `kUnsupportedOp`；Sample op 为空 → `kUnsupportedOp`。
4. Sample op 的 `DstTensors()` 不是恰好 1 个 → `kInvalidArgument`。
5. 对每个 Input op：
   - op 为空或 `DstTensors().size()!=1` → `kInvalidArgument`；
   - 读取 tensor Storage；`ResolveByteSize`：
     - `byte_size>0` 直接使用；
     - 否则 `shape.GetShapeSize() * PrecisionByteSize(precision)`；
     - shape/precision 非法 → `kInvalidArgument`/`kUnsupportedDtype`；
   - 分配并清零 `buffer`。
6. 对每个 Output op 以 `SrcTensors()` 执行同样的步骤 5。
7. 全部成功后一次性提交成员；失败时函数参数 `backend` 析构，自动释放后端。

边界条件：

- `storage.byte_size` 与 `shape×precision` 不一致时以 `byte_size` 为准
  （与 `IoOpEnv::Forward` 的字节数比较一致）；
- 缓冲大小为 0 视为非法，生产端必须提供正字节数或可推导形状。

### 4.4 Execute 详细流程

#### 4.4.1 前置校验

- 未 Init（`backend_/graph_/io_binder_` 为空）→ `kStateError`；
- `result == nullptr` → `kInvalidArgument`；进入时清空 `result->requests`；
- `plan.entries` 为空 → `kInvalidArgument`；
- `plan.total_tokens < 0` → `kInvalidArgument`；
- 每 entry：
  - `token_count <= 0` → `kInvalidArgument`；
  - `packed_offset < 0` → `kInvalidArgument`；
  - `token_ids.size() != token_count` 或 `positions.size() != token_count`
    → `kInvalidArgument`；
  - 累计和溢出 `I32` → `kInvalidArgument`；
- `plan.total_tokens != sum(token_count)` → `kInvalidArgument`；
- `packed_offset + token_count > sum` → `kInvalidArgument`；
- 第 1 个输入精度必须为 `kInt64`，第 2 个（存在时）同样；否则
  `kUnsupportedDtype`；缓冲 `byte_size % 8 != 0` → `kInvalidArgument`。
- KV 模式（`kv_view.block_size > 0`）额外校验：
  - `entry.slots.size() == token_count`；
  - `entry.block_table.size() >= ceil((context_length+token_count)/block_size)`；
  - `positions[i]` 单调递增且等于 `context_length+i`（prefill chunk）或
    `context_length`（decode，token_count=1）；
  - 各 entry 的 `[packed_offset, packed_offset+token_count)` 互不重叠；
  - 元数据需求不超过 `max_num_tokens`、`batch` 与块表 stride 上限。

#### 4.4.2 打包 token/position

```text
memset(buffer, 0, buffer.size())
for entry in plan.entries:
  for i in [0, entry.token_count):
    buffer[entry.packed_offset + i] = entry.token_ids[i]   // 或 positions[i]
```

- 使用 `memcpy` 按 8 字节写入，避免对齐问题；
- 第 2 个输入不存在时只填 token；
- 第 3 个及之后的输入不会被 plan 填充，缓冲保持全 0（保留扩展空间）。

#### 4.4.3 绑定主机 Storage

对每个输入槽：

```cpp
Storage host = slot.storage;          // shape/precision/layout/byte_size 复制
host.device_type = DeviceType::kHost;
host.offset = 0;
host.data_ptr = slot.buffer.data();
host.reusable = false;
io_binder_->SetInput(slot.name, host);
```

输出槽同理调用 `SetOutput`。任一步失败 → `kInvalidArgument`（由
IoBinder 返回）并加 `IoBinder::SetInput(name)` 前缀。

#### 4.4.4 StepContext 与 Attention 元数据

**元数据装配顺序**：

1. 计算 `batch = plan.entries.size()`、`token_count = Σ token_count`；
2. 逐 entry 填 `seq_lens[r] = context_length + token_count`、
   `query_lens[r] = token_count`、`context_lens[r] = context_length`；
3. 按 packed offset 展平 `request_indices`、`positions`、`slots`；
4. 计算 `stride = max(ceil(seq_lens[r]/block_size))`，按 `[r, b]` 填块表，
   尾部补 0；
5. 计算各数组在元数据缓冲中的 16 字节对齐 offset，写入主机暂存区；
6. 若需求超过 `MetadataCapacity`，先 `Backend::CreateBuffer` 扩容；
7. 一次 `Backend::UploadAt(metadata_buffer->DevicePtr(), 0, staging.data(),
   staging.size(), nullptr)`；
8. 用同一 Buffer* 与各 offset 填 `AttentionMetadataView`。

**StepContext**：

| 字段 | 取值 |
| --- | --- |
| `backend` | `backend_.get()` |
| `stream` | `nullptr`（默认流） |
| `plan` | `&plan` |
| `kv_view` | `&kv_view` |
| `attention_meta` | 7 个 `TensorView` + `block_table_stride`；`recompute_attention` 时为空 |
| `batch` | `plan.entries.size()` |
| `packed_tokens` | 所有 entry 的 `token_count` 之和 |
| `workspace` | 默认空 TensorView |

`plan/kv_view/attention_meta` 的生命周期覆盖整个 op 遍历；元数据 Buffer
在 `Deinit` 释放。删除旧字段 `seq_lens` 与 `is_prefill`。

#### 4.4.5 逐 op 执行

```text
for op in graph_->GetOrderlyOPList():
  op 为空                        → kStateError
  env = op->GetOpEnv() 为空      → kStateError（op 名称入消息）
  SetStepContext(env, &ctx) 失败 → 透传错误码
  Forward() 失败                 → 透传错误码并加 op 前缀
  SetStepContext(env, nullptr)   → 无论 Forward 成败都执行
```

遍历结束后要求实际访问到 `GetSampleOp()`；否则返回 `kStateError`
（sample 未从输出可达，回读会读到未初始化数据）。

#### 4.4.6 同步与 sample 回读

1. `Backend::Synchronize(nullptr)`；失败透传（通常 `kDeviceError`）。
2. 取 `graph_->GetSampleOp()->DstTensors()[0]->GetStorage()`：
   - 精度不是 `kInt64` → `kUnsupportedDtype`；
   - `data_ptr == nullptr` → `kInvalidArgument`；
   - `byte_size == 0` 或不能被 8 整除 → `kInvalidArgument`；
   - `Buffer*` 的 `DevicePtr()` 为空 → `kInternal`。
3. `vector<I64> sample_tokens(byte_size/8, 0)`；
4. `DownloadAt(sample_tokens.data(), DevicePtr(), Storage.offset, byte_size,
   nullptr)` 失败 → 透传错误码。不得先对 `DevicePtr()` 做 `+offset`。

#### 4.4.7 StepResult

对每个 entry：

- `output_slot < 0`：跳过，不产生 `RequestStepResult`（本步无 token）；
- `output_slot >= sample_tokens.size()`：`kInvalidArgument`；
- 否则追加：

```text
RequestStepResult{
  request_id = entry.request_id,
  token = Token{sample_tokens[output_slot], 0.0f, 0.0f, true},
  ok = true,
  error = kOk
}
```

顺序与 `plan.entries` 一致（消费方仍应按 `request_id` 匹配）。

### 4.5 Deinit

- `backend_ == nullptr` 时直接清空借用指针，返回 `kOk`；
- 否则调用 `Backend::Deinit()`：
  - 成功 → 释放 Backend，返回 `kOk`；
  - 失败 → 加 `Backend::Deinit` 前缀返回错误，但仍释放 Backend；
- 清空 `graph_/io_binder_/input_slots_/output_slots_`；
- 析构函数调用 `Deinit()` 并忽略返回值。

### 4.6 错误码汇总

| 场景 | 错误码 |
| --- | --- |
| 重复 Init / 未 Init 调用 Execute / OpEnv 缺失 / sample 不可达 / sample DevicePtr 空 | `kStateError` 或 `kInternal` |
| 空指针、空 plan、total 不一致、尺寸越界、output_slot 越界、sample Storage 非法 | `kInvalidArgument` |
| 图缺少 Input/Output/Sample op | `kUnsupportedOp` |
| token/position/sample 精度不是 int64 | `kUnsupportedDtype` |
| Forward/Synchronize/Download 失败 | 透传原错误码（`kKernelError`/`kDeviceError`/`kBufferError` 等） |
| slots 长度不符、块表覆盖不足、packed 区间重叠、元数据容量溢出 | `kInvalidArgument` |
| 元数据 Buffer 创建/上传失败 | `kBufferError`/`kDeviceError` |

### 4.7 内存与所有权

| 资源 | 所有者 | 生命周期 |
| --- | --- | --- |
| `Backend` | ExecBackend | Init 接管；Deinit/析构释放 |
| 主机 I/O 缓冲 | ExecBackend | Init 分配；Deinit 清空 |
| 元数据 Buffer | ExecBackend | 首次 Execute 分配/高水位扩容；Deinit 先于 Backend::Deinit 释放 |
| Graph / IoBinder | Engine | 借用，不释放 |
| OpEnv / Kernel / Buffer | Graph / Runtime | 借用；Engine 在 Executor Deinit 前清理 |
| `plan/kv_view/StepContext/attention_meta/sample_tokens` | Execute 栈 | 单步有效 |
| 元数据主机暂存 | Execute 栈或成员复用 | 上传完成后即可失效 |

### 4.8 性能与并发

- Init 一次性分配全部主机 I/O 缓冲；Execute 仅分配/复用元数据暂存与
  `sample_tokens` 小 vector；
- 元数据每步一次 H2D，大小约 `batch×12 + token_count×16 + batch×stride×4`
  字节；稳态不分配 Buffer；
- sample 数据被 D2H 两次（Output op 一次、Executor 回读一次）；这是通用性
  与正确性的取舍，后续可在样本 tensor 同时是图输出时复用输出缓冲；
- 单推理线程模型，无锁；Engine 保证 SetInput/SetOutput 在同一线程调用。

### 4.9 已知限制

1. ~~Engine 对每个 `PlanEntry` 都要求存在 `RequestStepResult`~~；已修复：
   Engine 跳过 `output_slot<0` 的 entry。
2. token/position 输入必须是 `int64`；其他整数精度不支持。
3. 仅约定前两个输入为 token/position；更多输入不参与 plan 填充。
4. `Init` 使用 `storage.byte_size` 作为缓冲大小，若 Loader 提供错误的
   字节数，错误在 `IoOpEnv::Forward` 处暴露为绑定失败。
5. sample 输出必须是顶层 `[..., vocab]` int64；不解析 `axis` 等采样属性。
6. 元数据 Buffer 首次超过容量时会重新分配；峰值容量在测试中记录，尚未做
   上限预分配。
7. KV 模式要求 `kv_view.block_size > 0`；recompute 模式与 KV 模式不能在同一
   Engine 生命周期内混用。

## 5. 实施步骤

| 阶段 | 任务 | 依赖 | 产出 |
| --- | --- | --- | --- |
| S1 | 阅读接口与存储约定，确定 token/position 顺序与 sample 读取方式 | executor.h、Graph headers | 设计要点 |
| S2 | 实现 `executor.cc`：HostIoSlot、打包、绑定、上下文、遍历、回读、Deinit | S1 | 可编译实现 |
| S3 | 编写 `executor_test.cc`：MockBackend + Gather→Linear→Sample 小图 | S2 | 测试程序 |
| S4 | 元数据装配：7 数组、块表 stride、高水位 Buffer、一次 H2D | S2 | 元数据代码与单测 |
| S5 | 混合 batch 与 chunked prefill 用例；与 Mock paged 联调 | S4 | 测试结果 |
| S4 | 编译、运行、修复失败用例并回归 | S3 | 通过的测试输出 |
| S5 | 编写设计文档、测试结果与 skill | S4 | 四件套 |

里程碑：S2 编译零告警；S4 全部用例 PASS。

风险与应对：

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 测试图权重设计错误导致 argmax 不可区分 | 误判实现正确性 | 用可解析的小整数权重并在首次失败后修正（见测试结果） |
| 测试 IoBinder 未写 `attr["storage"]` | IoOpEnv 绑定失败 | 测试 Binder 模拟 Engine，校验后 SetAttr |
| sample 非图输出张量 | 不能复用输出缓冲 | 显式按 Storage 回读 |
| Engine 无 token 步不兼容 | chunked prefill 集成失败 | 记录为限制，交由 Engine/Scheduler 后续处理 |

## 6. 测试与验证设计

### 6.1 测试策略

- 使用 `MockBackend`（主机 CPU 参考 Kernel）与 C++ Graph API 构造
  `Input → Gather → Linear → Sample → Output` 小图；
- 固定小整数权重：`embedding[i]=[i,1]`、`W[v]=[2v,-v²]`，使
  `logits[v]=-(v-i)²+i²`，argmax 恒等于输入 token id；
- 用 `TestIoBinder` 模拟 Engine：校验并写入 `attr["storage"]`；
- 通过断言检查 Host 缓冲内容、绑定字段、StepResult 与错误码。

### 6.2 用例设计

| 用例 | 覆盖需求 | 步骤 | 通过标准 |
| --- | --- | --- | --- |
| TC-EX-01 | REQ-EX-01 | `CreateDefaultExecutor`，检查 backend() | 非空、backend 为空 |
| TC-EX-02 | REQ-EX-01/14 | 未 Init Deinit；失败 Init 后检查；重复 Deinit | 均 kOk、backend 释放 |
| TC-EX-03 | REQ-EX-02 | 构造两输入/单输入图并 Init | Init kOk |
| TC-EX-04 | REQ-EX-03/06/07/08/09/10/11/12 | Prefill 4 token，output_slot=3 | 绑定成功、缓冲内容正确、token=7 |
| TC-EX-05 | REQ-EX-04/08/12 | 两个 Decode entry | 缓冲 [2,6,0,0]/[4,5,0,0]，结果 2 与 6 |
| TC-EX-06 | REQ-EX-12 | output_slot=-1 | StepResult 为空 |
| TC-EX-07 | REQ-EX-13 | 清空 linear OpEnv 后执行 | kStateError |
| TC-EX-08 | REQ-EX-05/13 | 5 token 超过 4 容量 | kInvalidArgument |
| TC-EX-09 | REQ-EX-13 | total_tokens 不一致、output_slot 越界、空 plan、空 result | kInvalidArgument |
| TC-EX-10 | REQ-EX-03 | 单输入图两个 Decode | 仅 1 次 SetInput，token=[1,4,0,0] |
| TC-EX-11 | REQ-EX-01/13 | 空图无 Input、无 Sample、空参数 Init | kUnsupportedOp / kInvalidArgument |
| TC-EX-12 | REQ-EX-14 | Deinit 两次 | 均 kOk、backend() 为空 |
| TC-EX-13 | REQ-EX-13 | 注入 IoBinder 绑定失败 | kInvalidArgument 且 StepResult 为空 |
| TC-EX-14 | REQ-EX-16/17/18 | 单请求 prefill chunk（context=16, tokens=16） | seq/query/context/positions/slots/块表逐项正确；一次 H2D |
| TC-EX-15 | REQ-EX-19 | 2 prefill chunk + 2 decode 混排 | request_indices 逐 token 正确；batch/stride 正确 |
| TC-EX-16 | REQ-EX-20 | slots 长度不符、块表覆盖不足、packed 区间重叠 | kInvalidArgument；无部分执行 |

### 6.3 测试环境与命令

```bash
cd d:/Mobile_Net
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  Runtime/executor/executor.cc Runtime/executor/tests/executor_test.cc \
  Runtime/backend/build/libruntime_backend.a Graph/build/libgraph.a \
  -o Runtime/executor/build/executor_test.exe
./Runtime/executor/build/executor_test.exe
```

### 6.4 通过标准

- 编译与链接无错误、无告警；
- `executor_test.exe` 输出 `ALL TESTS PASSED`，退出码 0；
- 实际输出记录在 `Runtime/executor/tests/test_results.md`。

### 6.5 未覆盖项

- 真实设备后端、异步流与多流；
- Engine/Scheduler/KVCache 端到端联调（由 Runtime e2e 覆盖）；
- 非 int64 token/position 的拒绝路径（代码已实现，测试未覆盖）；
- 第 3 个及以后输入的绑定内容；
- `storage.byte_size==0` 时按 shape 推导的路径；
- 元数据 Buffer 的峰值容量/长期增长（阶段一记录实测，不做上限预分配）。

## 附录 A：与设计文档的对应关系

- `Runtime/.doc/Runtime_design.md` 4.2.4 明确 Executor 是 Backend 唯一所有者；
  本实现遵循 `Init` 失败释放、`Deinit` 幂等。
- 4.3.4 的 `Backend::Download` 语义为“主机指针 + 设备指针 + 字节数 +
  流”；带 arena 偏移的 sample 回读使用新增的 `DownloadAt`（设备指针保持
  `Buffer::DevicePtr()`，偏移单独传入），避免对 `cl_mem` 句柄做指针加法。
- 3.5 的调用链要求 Executor 是唯一遍历 `GetOrderlyOPList()` 并调用
  `Forward()` 的对象；本实现满足。

## 10. 2026-11-21 补充：sample_rows 缓冲

- `Execute` 从 `PlanEntry.output_slot>=0` 构建 `std::vector<I32> sample_rows`；
- 通过 Executor 私有高水位 Buffer 一次 H2D 上传，填入
  `StepContext.sample_rows_view`；
- `Deinit` 在 `Backend::Deinit` 前释放该缓冲；
- partial prefill（无 output_slot）时行数为 0，LM Head/Sample 跳过计算；
- 单算子测试未启用行选择，Kernel 按全部有效行处理。
