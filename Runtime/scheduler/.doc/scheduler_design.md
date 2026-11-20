# Scheduler 模块设计文档

| 项目 | 内容 |
| --- | --- |
| 模块路径 | `Runtime/scheduler/` |
| 权威接口 | `Runtime/scheduler/scheduler.h`（`IScheduler`） |
| 上游设计 | `Runtime/.doc/Runtime_design.md` 4.2.2；`MobileNet_design.md` 4.3.2 |
| 实现文件 | `Runtime/scheduler/scheduler.cc` |
| 测试 | `Runtime/scheduler/tests/scheduler_test.cc`、`Runtime/scheduler/tests/test_results.md` |
| 状态 | 阶段一/二默认实现；不含抢占与投机解码 |

## 1. 背景与目标

### 1.1 背景

Runtime 以 Continuous Batching 方式执行推理：Engine 每个 Step 调用
`IScheduler::Schedule` 获取 `ExecutionPlan`，Executor 执行后由 Engine 调用
`IScheduler::Update` 回写 token。Scheduler 必须在不感知设备与 KV 物理布局的
前提下，把多请求组织成单步执行计划，并维护每个请求的 `Sequence` 与状态。

后端支持两种执行形态：

- **KV 模式**（`recompute_attention=false`）：Prompt 走 chunked prefill，
  已进入解码的请求每步只喂 1 个 token；KV 通过 `KvAllocator` 申请。
- **完整历史重算模式**（`recompute_attention=true`）：每步把请求的完整历史
  重新送入图，不申请 KV；适用于无 KV 的后端与调试。

### 1.2 目标

1. 实现通用（不绑定 Qwen3 或任何具体模型）的请求合并、调度与停止判定。
2. 两种执行形态都满足 `max_num_tokens` 单步预算，且计划字段自洽
   （`total_tokens == Σ token_count`）。
3. 正确维护 `RequestStatus`：`kWaiting`、`kChunkedPrefill`、`kRunning`、
   `kStopped`、`kIllegal`。
4. `Update/Stop/Fail` 生成完整 `Response`，终态统一经 `KvAllocator::Release`
   释放 KV，并支持重复调用幂等。
5. 所有错误通过 `Status` 返回，空指针与非法规格有明确错误码。

### 1.3 非目标

- 不实现抢占/换出；`preempted_requests` 恒为 0。
- 不实现阶段三投机解码、Beam、采样；只按 token id 判停。
- 不调用设备 API、不操作 Kernel、不解析文本。
- 不拥有 KV 物理块；块表与 slot 由 `KvAllocator` 提供。
- 不负责停止推理线程；线程生命周期由 Engine 管理。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| FR-SCH-01 | `Merge` 校验 prompt 非空、token id 合法、长度严格小于 `context_length`；`request_id=0` 时自动分配 | TC-SCH-01/03 |
| FR-SCH-02 | `Merge` 初始化 `Sequence`：history=prompt、prompt_count、vocab_size（来自 `ModelConfig`）、generated_count=0、status=`kWaiting`，并保存 `SampleParams` | TC-SCH-01 |
| FR-SCH-03 | 重算模式 `Schedule` 生成完整历史计划：token_ids=history、positions=0..len-1、token_count=len、output_slot=len-1、context_length=len-1；generated_count=0 时 phase=Prefill，否则 Decode | TC-SCH-01 |
| FR-SCH-04 | 重算模式不调用 `KvAllocator::Reserve`，slots/block_table 为空 | TC-SCH-01 |
| FR-SCH-05 | KV 模式 Waiting 请求按 `min(prompt 剩余, max_prefill_chunk, 预算)` 切 chunk；调用 Reserve/GetBlockTable/GetSlots；未完成置 `kChunkedPrefill`，完成置 `kRunning` | TC-SCH-02 |
| FR-SCH-06 | KV 模式 Running 请求每步 decode 1 token，token 取自 `history[computed_tokens]`，位置与 KV slot 一致 | TC-SCH-02 |
| FR-SCH-07 | 单步总 token 不超过 `max_num_tokens`；Prefill 优先于 Decode | TC-SCH-01/02/04 |
| FR-SCH-08 | `Update` 追加 token 到 history/current_token/generated_count；按 min_tokens、stop_token_ids、ignore_eos、max_tokens、context_length 判停 | TC-SCH-01/02/05 |
| FR-SCH-09 | 终态 `Update/Stop/Fail` 调用 `KvAllocator::Release`（仅当曾 Reserve），并填充 `Response` | TC-SCH-02/03 |
| FR-SCH-10 | `Stop` 生成 `kStopped`；`Fail` 生成 `kIllegal`；已结束请求重复调用幂等 | TC-SCH-03 |
| FR-SCH-11 | `Snapshot` 返回 request_id/status/prompt/computed/generated；`Stats` 返回 waiting/running/preempted/free/total blocks；`HasRequests` 反映未结束请求 | TC-SCH-02 |
| FR-SCH-12 | `Merge/Stop/HasRequests` 可与推理线程并发；内部互斥锁保护 | 代码审查 + 设计约束 |
| FR-SCH-13 | `Response*`/`ExecutionPlan*`/`request_id*` 为空返回 `kInvalidArgument` | TC-SCH-03 |

### 2.2 非功能需求

- **NFR-SCH-01 确定性**：同一组请求与配置下，按 request_id 升序调度，计划稳定。
- **NFR-SCH-02 有界内存**：请求表只随请求数增长；单个 plan 的 token 不超过
  `max_num_tokens`。终态请求保留用于幂等，长期运行需由上层定期回收
  Engine 生命周期（见 4.6 限制）。
- **NFR-SCH-03 复杂度**：单步调度 O(请求数)；`Reserve/GetSlots` 由
  KVCache 的 O(新增块数) 实现。
- **NFR-SCH-04 可观测性**：所有错误消息包含方法名与原因；Stats 足以驱动
  `StepStats`。
- **NFR-SCH-05 无线程泄漏**：Scheduler 不创建线程，不持有设备资源。

### 2.3 约束条件

- `max_num_tokens > 0`；`context_length > 0`；`max_prefill_chunk=0` 时取
  `max_num_tokens`；`max_concurrent_requests=0` 时取 `max_num_tokens`。
- `Merge` 要求 `prompt_token_ids` 非空且 prompt 长度严格小于
  `context_length`（至少留 1 个生成位置）。
- 重算模式下 prompt 长度必须严格小于 `max_num_tokens`，否则无法在单步
  容纳“prompt + 至少 1 个生成 token”。
- 阶段一/二请求 `multimodal_inputs` 必须为空，否则 `kUnsupportedOp`。
- 终态请求的 `status` 不再改变；`Stop` 不会把 `kIllegal` 改成 `kStopped`。

### 2.4 验收标准

1. `scheduler_test` 全部断言通过，退出码 0（当前 83 条断言）。
2. 重算单请求流程覆盖 Prefill→Decode→Stop，计划字段逐项匹配。
3. KV 模式覆盖 partial chunk、final chunk、decode 与 Release。
4. Stop/Fail/重复调用/空指针/未知请求的错误码全部符合接口注释。
5. 编译命令在 `-Wall -Wextra` 下无警告。

## 3. 概要设计

### 3.1 总体架构

```text
Engine
  └── IScheduler (SchedulerImpl)
        ├── requests_: map<request_id, RequestState>
        │     ├── Sequence（history/current_token/generated_count/vocab_size）
        │     ├── RequestStatus
        │     ├── SampleParams
        │     └── computed_tokens / kv_reserved
        ├── SchedulerConfig（预算、上下文、停止 token、recompute）
        └── KvAllocator*（KV 模式块分配；非拥有）
```

### 3.2 模块划分

| 组成 | 职责 |
| --- | --- |
| `RequestState` | 请求级可变状态：Sequence、状态、采样参数、已计算 token 数、KV 标记 |
| `SchedulerImpl` | 公共接口实现与互斥锁；请求表与 ID/Step 计数器 |
| `ScheduleRecompute` | 重算模式计划生成 |
| `ScheduleWithKv` | chunked prefill + decode 计划生成 |
| `EffectiveMaxTokens` | 由 sample.max_tokens/context_length/重算预算推导生成上限 |
| `ShouldStop` | 停止条件判定 |
| `FillResponse` | 统一 Response 快照填充 |

### 3.3 关键流程

#### 3.3.1 Merge

```text
校验 request/response 指针与初始化状态
  → 校验 prompt/多模态/采样参数/token id
  → 校验并发上限（ActiveRequestCount < max_concurrent_requests）
  → 分配或校验 request_id
  → 构造 RequestState(status=kWaiting, computed=0)
  → requests_.emplace
```

#### 3.3.2 Schedule（重算模式）

```text
plan = {}; step_id = next_step_id_++
for request in requests_（按 id 升序，跳过终态）:
    if history 长度 > 剩余预算: 跳过
    entry = {history, positions 0..n-1, output_slot=n-1,
             context_length=n-1, phase=generated_count==0?Prefill:Decode}
    total += n; computed_tokens = n; status = kRunning
```

#### 3.3.3 Schedule（KV 模式）

```text
收集 waiting（kWaiting/kChunkedPrefill）与 running（kRunning）id
for waiting:
    chunk = min(prompt 剩余, max_prefill_chunk, 剩余预算)
    Reserve(chunk) → GetBlockTable → GetSlots(computed, chunk)
    entry.context_length = computed; entry.output_slot 仅在 chunk 完成 prompt 时有效
    computed += chunk; status = 完成?kRunning:kChunkedPrefill
for running:
    Reserve(1) → GetBlockTable → GetSlots(computed, 1)
    entry = {history[computed], positions={computed}, phase=Decode}
    computed += 1
```

### 3.4 技术选型与取舍

- **每步重新组批**：实现 Continuous Batching，避免维护复杂的 persistent batch。
- **请求表用 `std::map`**：天然按 request_id 排序，调度确定性高；请求规模为
  数十到数百，O(n) 遍历可接受。
- **全方法互斥**：`Merge/Stop/HasRequests` 与推理线程并发，锁粒度以请求表为
  单位；调度/更新本身在单线程完成，锁竞争低。
- **终态保留**：为了 `Update/Stop/Fail` 幂等与 Snapshot 可查询，终态请求
  不删除；内存由 Engine 生命周期兜底。
- **部分调度成功**：KV 分配失败时，若本步已有计划条目则返回 `kOk` 并执行
  已成功部分；若计划为空则返回首个 KV 错误。避免“已提交状态但计划被丢弃”。

## 4. 详细设计

### 4.1 数据结构

```cpp
struct RequestState {
  U64 request_id;
  Sequence sequence;
  RequestStatus status = RequestStatus::kWaiting;
  SampleParams sample_param;
  I32 computed_tokens = 0;   // Schedule 推进，Update 不修改
  Bool kv_reserved = false;  // 是否调用过 Reserve
};
```

`SchedulerImpl` 私有字段：

| 字段 | 含义 |
| --- | --- |
| `max_num_tokens_` | 单步 token 预算 |
| `max_prefill_chunk_` | 单 chunk 上限（已按 max_num_tokens 截断） |
| `max_concurrent_requests_` | Merge 准入上限 |
| `context_length_` | 上下文长度 |
| `vocab_size_` | token 校验上界；0 表示未知 |
| `recompute_attention_` | 执行形态开关 |
| `default_stop_ids_` | 模型默认停止 token（config 优先） |
| `kv_allocator_` | 非拥有 KvAllocator |
| `requests_` | 请求表 |
| `next_request_id_` / `next_step_id_` | 单调计数器 |

### 4.2 接口与状态机

```text
Merge ──► kWaiting
             │ Schedule（chunk 未完成）
             ▼
       kChunkedPrefill ──Schedule（chunk 完成）──► kRunning
             │                                      │
             │ Stop/Fail                            │ Update/Stop/Fail
             ▼                                      ▼
        kStopped/kIllegal  ◄──────────────── kStopped/kIllegal
```

- `kRunning` 的 `Update` 追加 token；若未停止则保持 `kRunning`。
- `kChunkedPrefill` 的 `Update` 忽略输出 token（partial chunk 无采样输出），
  保持 `kChunkedPrefill`。
- 终态请求的 `Update/Stop/Fail` 直接返回原状态快照。

### 4.3 算法与流程

#### 4.3.1 执行计划字段

| 字段 | 重算模式 | KV 模式 Prefill | KV 模式 Decode |
| --- | --- | --- | --- |
| `token_ids` | 完整 history | `history[computed, computed+chunk)` | `{history[computed]}` |
| `positions` | 0..len-1 | computed..computed+chunk-1 | computed |
| `token_count` | len | chunk | 1 |
| `context_length` | len-1 | computed | computed |
| `output_slot` | len-1 | chunk 完成 prompt 时=packed+chunk-1，否则 -1 | packed_offset |
| `phase` | generated=0?Prefill:Decode | Prefill | Decode |
| `is_chunk` | false | 未完成 prompt 为 true | false |
| `slots/block_table` | 空 | KvAllocator 输出 | KvAllocator 输出 |

#### 4.3.2 停止判定顺序

1. `max_tokens`：`sample.max_tokens>0` 时取该值，否则取
   `context_length - prompt_count`；重算模式再与 `max_num_tokens - prompt_count`
   取小。
2. `generated >= max_tokens` 或 `history.size() >= context_length` → 停止。
3. 重算模式 `history.size() >= max_num_tokens` → 停止（保障单步可执行）。
4. `generated < min_tokens` → 不因停止 token 停止。
5. 命中 `sample.stop_token_ids` → 停止。
6. `ignore_eos=false` 且命中 `default_stop_ids_` → 停止。

多 token 的 `Update` 按顺序追加，命中停止 token 后不再追加后续 token。

#### 4.3.3 错误与边界

| 场景 | 错误码 |
| --- | --- |
| 未 Init 调用 Schedule/Update/Stop/Fail/Merge | `kStateError` |
| `Response*`/`plan*`/`request_id*` 为空 | `kInvalidArgument` |
| prompt 为空/超上下文/非法 token/重复 id | `kInvalidArgument` |
| 多模态输入非空 | `kUnsupportedOp` |
| 并发已满 | `kDeviceOom` |
| 请求不存在 | `kRequestNotFound` |
| 计划为空且 KV 分配失败 | 首个 KV 错误（通常 `kDeviceOom`） |
| 重算 history 超预算/空 history | `kInternal` |

#### 4.3.4 线程与性能

- `std::mutex` 保护全部公共方法；`Schedule/Update/Fail/Stats` 正常只由推理线程
  调用，但实现不依赖该假设。
- 单步调度 O(请求数 + 新增块数)；`GetSlots` 每次分配输出向量。
- 计划中 token/positions/slots 均为值拷贝，Executor 可安全跨线程持有计划。

### 4.4 关键实现细节与限制

1. **Prefill 优先**：Waiting 先于 Running 消耗预算；短预算下可能推迟 Decode，
   与设计文档 4.3.2 的策略一致。
2. **无抢占**：Running 请求 Reserve(1) 失败时保留 `kRunning`，等下一步重试；
   `preempted_requests` 恒为 0。
3. **chunked prefill 输出**：Executor 对 partial chunk 可能返回 `token_id=-1`，
   `Update` 在 `kChunkedPrefill` 状态下忽略所有输出 token。
4. **终态保留**：请求表不自动清理，长时间服务需 Engine 重建或后续版本增加
   回收接口（当前无公共接口，故不实现）。
5. **预算保障**：重算模式在 `Update` 中按 `max_num_tokens` 截停，保证下一次
   `Schedule` 的 history 不会超预算。
6. **max_concurrent_requests**：在 Merge 处作为未结束请求上限；Token packing
   模式下它是请求数上限，不是单步条目上限。

## 5. 实施步骤

### 5.1 阶段划分

| 阶段 | 任务 | 产出 |
| --- | --- | --- |
| P1 | 阅读接口与公共数据契约，确定状态机与字段语义 | 本设计文档 4.1/4.2 |
| P2 | 实现 Merge/Init/Snapshot/Stats/HasRequests | `scheduler.cc` 基础部分 |
| P3 | 实现重算 Schedule 与 Update 停止判定 | 重算模式可运行 |
| P4 | 实现 KV 模式 chunked prefill/decode 与 Release | KV 模式可运行 |
| P5 | 编写单元测试、修复边界、记录结果 | `tests/` |

### 5.2 依赖与里程碑

- 依赖 `Runtime/common/{sequence,plan,kv_allocator,model_config,status}.h`；
  不修改公共头。
- 里程碑 1：重算单请求测试通过。
- 里程碑 2：KV chunked prefill/decode 测试通过。
- 里程碑 3：Stop/Fail/幂等/错误码测试通过。

### 5.3 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| KV 不足导致请求长期等待 | 请求不结束 | 当前保留 `kRunning`/`kWaiting`；后续实现抢占 |
| 终态请求累积 | 内存增长 | Engine 销毁时释放；后续增加请求回收接口 |
| partial chunk 无输出 | Update 误判 | 按 `kChunkedPrefill` 状态忽略输出 token |
| 重算 history 超单步预算 | 无法调度 | Merge/Update 双重截停 |

## 6. 测试与验证设计

### 6.1 测试策略

- 纯主机单元测试，Fake `KvAllocator` 记录 Reserve/Release 与容量。
- 以值断言验证 PlanEntry 每个字段、Response、Snapshot、Stats 与错误码。
- 不依赖设备/Backend/Graph，确保可在 CI 主机直接运行。

### 6.2 用例设计

| 用例 | 覆盖需求 | 内容 |
| --- | --- | --- |
| TC-SCH-01 | FR-SCH-01/02/03/04/08 | 重算单请求 Prefill→Update→Decode→Stop；断言计划字段与不调用 Reserve |
| TC-SCH-02 | FR-SCH-05/06/07/09/11 | KV chunked prefill（4+2）、partial chunk 无输出、decode、Release、Stats |
| TC-SCH-03 | FR-SCH-10/13 | Stop/Fail/未知请求/空指针/kOk 错误码/重复调用幂等 |
| TC-SCH-04 | FR-SCH-07 | 重算模式两个 3-token prompt、预算 5：每步只调度一个请求 |
| TC-SCH-05 | FR-SCH-08 | min_tokens 阻止提前停止、ignore_eos 忽略默认停止 token、max_tokens 截停 |
| TC-SCH-06 | 约束 | 并发上限 1 的 Merge 准入与 Stop 后重新准入 |

### 6.3 测试环境与命令

```bash
cd d:/Mobile_Net
mkdir -p Runtime/build
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  Runtime/scheduler/tests/scheduler_test.cc \
  Runtime/scheduler/scheduler.cc \
  Runtime/common/status.cc \
  -o Runtime/build/scheduler_test.exe
./Runtime/build/scheduler_test.exe
```

通过标准：退出码 0，`failures=0`。实际输出见
`Runtime/scheduler/tests/test_results.md`。

### 6.4 未覆盖项与残余风险

- 未测试多线程并发 Merge/Stop（实现有锁，但未做压力测试）。
- 未测试抢占、投机解码、Beam、采样温度/惩罚。
- 未测试超长 prompt（> I32 上限）与极端预算的整型边界。
- 终态请求回收策略不在本模块实现。
- **端到端 chunked prefill 契约缺口**：本模块对未完成 Prefill 的
  `PlanEntry` 设置 `output_slot=-1`（符合 `plan.h` 语义），Executor 会跳过
  该请求不返回 `RequestStepResult`；当前 `Engine::ExecuteStep` 把“缺少
  result”视为 `kInternal` 并调用 `Scheduler::Fail`。因此长 prompt 分块在
  真实 Engine 路径会被判失败（本模块单测不经过 Engine）。修复点在 Engine
  或 Executor：Engine 应跳过 `output_slot<0` 的条目，或 Executor 返回
  `token_id=-1` 的结果；Scheduler 的 `Update` 已能忽略后者。该修复不需要
  修改 `scheduler.h`。

## 附录 A：需求-设计-测试追溯

| 需求 | 设计章节 | 测试用例 |
| --- | --- | --- |
| FR-SCH-01 | 3.3.1 | TC-SCH-01/03 |
| FR-SCH-02 | 4.1 | TC-SCH-01 |
| FR-SCH-03 | 3.3.2/4.3.1 | TC-SCH-01 |
| FR-SCH-04 | 3.3.2 | TC-SCH-01 |
| FR-SCH-05 | 3.3.3 | TC-SCH-02 |
| FR-SCH-06 | 3.3.3 | TC-SCH-02 |
| FR-SCH-07 | 3.3.2/3.3.3 | TC-SCH-04 |
| FR-SCH-08 | 4.3.2 | TC-SCH-01/05 |
| FR-SCH-09 | 4.2/4.3.3 | TC-SCH-02/03 |
| FR-SCH-10 | 4.2/4.3.3 | TC-SCH-03 |
| FR-SCH-11 | 4.1/4.3.1 | TC-SCH-02 |
| FR-SCH-12 | 4.3.4 | 代码审查 |
| FR-SCH-13 | 4.3.3 | TC-SCH-03 |
