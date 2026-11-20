# Executor 模块设计文档

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

### 1.3 非目标

- 不实现调度、KV 分配/释放、抢占、停止判定或 `Response` 构建；
- 不创建/销毁 OpEnv、Kernel 或 Buffer；这些由 Loader/Engine/Manager 负责；
- 不实现 H2D/D2H 本身；传输由 `IoOpEnv` 调用 `Backend::Upload/Download`；
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
| REQ-EX-08 | 构造 `StepContext`：backend、plan、kv_view、seq_lens（每 entry 的 token_count）、batch、packed_tokens、is_prefill | TC-EX-04/05 |
| REQ-EX-09 | 按 `GetOrderlyOPList()` 依次 `SetStepContext(env,&ctx)`、`Forward()`、`SetStepContext(env,nullptr)` | TC-EX-04/05 |
| REQ-EX-10 | 全部 op 后执行 `Backend::Synchronize`；从 `GetSampleOp()->DstTensors()[0]` 的 Storage 读取 sample | TC-EX-04 |
| REQ-EX-11 | sample Storage 的 `data_ptr` 为 `Buffer*`，加 `offset` 后 `Download` 到 `vector<I64>`，大小 `byte_size/8` | TC-EX-04 |
| REQ-EX-12 | 按 `output_slot` 从 sample 取 token；`output_slot<0` 跳过；填 `Token{id,0,0,true}`、`ok=true` | TC-EX-04/05/06 |
| REQ-EX-13 | 缺失 OpEnv、I/O 绑定失败、尺寸不匹配返回明确错误码 | TC-EX-07/08/09/11 |
| REQ-EX-14 | `Deinit` 清空缓冲，调用 `Backend::Deinit` 并释放；幂等；未 Init 调用返回成功 | TC-EX-02/12 |
| REQ-EX-15 | 不写死模型/形状；所有尺寸来自 Graph Storage | 代码评审 |

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
  有效地址 = `static_cast<Buffer*>(data_ptr)->DevicePtr() + offset`。
- `IoOpEnv::Forward` 从 op 的 `attr["storage"]` 读取主机 Storage，并且要求
  `device_type=kHost`、`byte_size` 与精度和设备视图一致。
- `KernelOpEnv::Forward` 要求 `StepContext` 非空；未设置时返回 `kStateError`。
- Engine 在 Load 中把 Backend 移交给 Executor，Executor 必须在 Destroy
  阶段晚于所有 OpEnv 释放 Backend。

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
→ 构造 StepContext 与 seq_lens
→ for op in GetOrderlyOPList():
     SetStepContext(env,&ctx) → Forward() → SetStepContext(env,nullptr)
→ Backend::Synchronize
→ Download sample.out（Buffer* + offset）到 vector<I64>
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

#### 4.4.4 StepContext

| 字段 | 取值 |
| --- | --- |
| `backend` | `backend_.get()` |
| `stream` | `nullptr`（默认流） |
| `plan` | `&plan` |
| `kv_view` | `&kv_view` |
| `seq_lens` | 栈上 `vector<I32>`，逐 entry 填 `token_count` |
| `batch` | `plan.entries.size()` |
| `packed_tokens` | 所有 entry 的 `token_count` 之和 |
| `is_prefill` | `plan.entries[0].phase == RequestPhase::kPrefill` |
| `workspace` | 默认空 TensorView |

`seq_lens` 与 `plan/kv_view` 的生命周期覆盖整个 op 遍历。

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
4. `Download(sample_tokens.data(), DevicePtr()+offset, byte_size, nullptr)`
   失败 → 透传错误码。

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

### 4.7 内存与所有权

| 资源 | 所有者 | 生命周期 |
| --- | --- | --- |
| `Backend` | ExecBackend | Init 接管；Deinit/析构释放 |
| 主机 I/O 缓冲 | ExecBackend | Init 分配；Deinit 清空 |
| Graph / IoBinder | Engine | 借用，不释放 |
| OpEnv / Kernel / Buffer | Graph / Runtime | 借用；Engine 在 Executor Deinit 前清理 |
| `plan/kv_view/StepContext/seq_lens/sample_tokens` | Execute 栈 | 单步有效 |

### 4.8 性能与并发

- Init 一次性分配全部主机 I/O 缓冲；Execute 仅分配 `seq_lens` 与
  `sample_tokens` 两个小 vector，后者可后续改为成员复用；
- 每个 step 重复绑定 I/O，但 `IoBinder` 只写 attr，开销可忽略；
- sample 数据被 D2H 两次（Output op 一次、Executor 回读一次）；这是通用性
  与正确性的取舍，后续可在样本 tensor 同时是图输出时复用输出缓冲；
- 单推理线程模型，无锁；Engine 保证 SetInput/SetOutput 在同一线程调用。

### 4.9 已知限制

1. Engine 当前对每个 `PlanEntry` 都要求存在 `RequestStepResult`；当
   `output_slot<0` 时本模块按约定跳过，Engine 会把该请求判为
   `kInternal`。这需要 Engine/Scheduler 侧支持“无 token 步”，本模块不改
   公共模块。
2. token/position 输入必须是 `int64`；其他整数精度不支持。
3. 仅约定前两个输入为 token/position；更多输入不参与 plan 填充。
4. `Init` 使用 `storage.byte_size` 作为缓冲大小，若 Loader 提供错误的
   字节数，错误在 `IoOpEnv::Forward` 处暴露为绑定失败。
5. sample 输出必须是顶层 `[..., vocab]` int64；不解析 `axis` 等采样属性。

## 5. 实施步骤

| 阶段 | 任务 | 依赖 | 产出 |
| --- | --- | --- | --- |
| S1 | 阅读接口与存储约定，确定 token/position 顺序与 sample 读取方式 | executor.h、Graph headers | 设计要点 |
| S2 | 实现 `executor.cc`：HostIoSlot、打包、绑定、上下文、遍历、回读、Deinit | S1 | 可编译实现 |
| S3 | 编写 `executor_test.cc`：MockBackend + Gather→Linear→Sample 小图 | S2 | 测试程序 |
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
- Engine/Scheduler/KVCache 端到端联调（默认 Loader 尚未提供模型包）；
- 非 int64 token/position 的拒绝路径（代码已实现，测试未覆盖）；
- 第 3 个及以后输入的绑定内容；
- `storage.byte_size==0` 时按 shape 推导的路径；
- 多 batch 形状 > `[1,4]`、动态形状与真实 Qwen3 图。

## 附录 A：与设计文档的对应关系

- `Runtime/.doc/Runtime_design.md` 4.2.4 明确 Executor 是 Backend 唯一所有者；
  本实现遵循 `Init` 失败释放、`Deinit` 幂等。
- 4.3.4 的 `Backend::Download` 语义为“主机指针 + 设备指针 + 字节数 +
  流”，本实现用 `Buffer*` + `offset` 组装设备指针。
- 3.5 的调用链要求 Executor 是唯一遍历 `GetOrderlyOPList()` 并调用
  `Forward()` 的对象；本实现满足。
