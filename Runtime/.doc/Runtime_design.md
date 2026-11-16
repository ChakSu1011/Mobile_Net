# Runtime 统一架构设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.3 |
| 文档状态 | 架构设计（接口冻结稿，Option A 已确认） |
| 编写日期 | 2026-11-16 |
| 变更摘要 | v0.3：Engine→Executor→Backend 所有权；Backend/KVCache/OpEnvFactory 不暴露给 Engine；Kernel 契约迁至 `Runtime/backend/`；Scheduler 统一释放 KV |
| 模块 | `Runtime/` |
| 权威位置 | `Runtime/.doc/Runtime_design.md` |
| 上游文档 | [`.doc/MobileNet_design.md`](../../.doc/MobileNet_design.md)、[`Graph/.doc/Graph_design.md`](../../Graph/.doc/Graph_design.md) |
| 子模块文档 | `Runtime/<module>/.doc/<module>_design.md`（不得与本文件冲突） |
| 强制规则 | memory.md：Google Style、中文注释、四件套、目录规范 |

**文档定位**：本文件是 Runtime 模块的权威架构文档，定义分层、Engine 组合方式、模块统一接口、调用关系、交互数据对象、所有权与生命周期。子模块文档只展开算法与实现细节；跨模块契约以本文件为准。

**本阶段交付范围**：本文件 + 全部接口头文件（仅声明，不实现）。`Runtime/` 下的 `.cc` 实现、单元测试、skill 在后续阶段按四件套补齐；本阶段不以“模块完成”宣称，只宣称“接口冻结”。

---

## 1. 背景与目标

### 1.1 背景

离线工具链（Converter → Optimizer → Serializer）已打通，Graph C++17 runtime IR 已实现（Input/Sample/Output、SortOp、GraphEnv/OpEnv）。首个模型包为 `Models/Qwen3/artifacts/qwen3_0_6b/`：图输入 2、常量 310、计算 op 312、activation 312、IR 633 行、FP16 权重约 1.19 GB。

Runtime 目前只有 `common/`、`backend/backend.h`（含 `backend/kernel.h`）头文件；`engine/`、`loader/`、`scheduler/`、`kvcache/`、`destroy/`、`executor/` 尚未定义接口。RuntimeCore 与 ExecBackend 将并行开发，必须先冻结接口与数据契约。

### 1.2 目标

1. 定义 Runtime 的分层、Engine 组合方式、模块边界与所有权。
2. 冻结 `Loader/Graph/Backend/Scheduler/Executor/KVCache/Destroy` 的统一接口与 Engine 的成员持有方式。
3. 定义加载、单步推理、请求生命周期、销毁四条主流程及逐层调用链。
4. 定义全部交互数据对象（配置、请求、计划、KV 视图、结果、统计、回滚上下文）。
5. 定义 Runtime 对外（阶段二 Service 层）的稳定 API。
6. 沉淀 vLLM v1 / SGLang 的可迁移技术，并明确 MobileNet 的取舍。
7. 交付可编译检查的接口头文件（仅声明），为后续实现与并行开发提供唯一契约。

### 1.3 非目标

- 不实现任何模块的 `.cc` 内部逻辑（本阶段只冻结接口）。
- 不定义 Kernel 数学实现、OCL 调优、Attention 算法（ExecBackend 文档）。
- 不定义模型转换、图优化、序列化格式（ToolChain 文档）。
- 不实现阶段二 Service 与阶段三功能（投机解码、异步调度、HiCache、多类型 KV），只冻结扩展点。
- 不做性能调优与真机基准。

---

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收要点 | 追溯 |
| --- | --- | --- | --- |
| R-ARCH-01 | Engine 以指针持有 `Loader/Graph/Backend/Scheduler/Executor/KVCache/Destroy`，通过统一接口调用；支持注入不同实现 | 成员类型为接口/指针；`SetModules` 可在 Load 前替换实现 | TC-ARCH-01 |
| R-ARCH-02 | 每个模块提供唯一统一接口：`ILoader/IScheduler/IKVCache/IExecutor/IDestroyer/Backend/Kernel`；Graph 使用既有类接口 | 接口头文件无实现；Engine 只依赖接口与公共数据 | TC-ARCH-01 |
| R-ARCH-03 | 公共数据对象集中在 `Runtime/common/`：错误码、配置、请求、计划、KV 视图、IO 绑定、资源栈 | 所有模块只使用 common 类型；无重复定义 | TC-ARCH-01 |
| R-ARCH-04 | Loader 完成包校验、图重建、IR 哈希、后端初始化、能力校验、权重上传、缓冲分配、GraphEnv/OpEnv 创建 | 正常包加载成功；坏包按错误码拒绝；失败逆序回滚 | TC-LOAD-01 |
| R-ARCH-05 | Engine 驱动单步：Schedule → Execute → Update → Release；单步 token ≤ `max_num_tokens` | 单请求/多请求结果正确；预算不超 | TC-EXEC-01、TC-SCHED-01 |
| R-ARCH-06 | 请求生命周期：Waiting/Running/Preempted/Finished/Cancelled/Failed；Merge/Update/Stop 语义固定 | 状态迁移合法；取消/超时生效 | TC-SCHED-02 |
| R-ARCH-07 | KVCache 块大小由后端 Attention tile 决定；块池不变量成立；容量启动期校验 | 不变量测试通过；OOM 可定位；启动期拒绝不可行配置 | TC-KV-01 |
| R-ARCH-08 | Executor 只消费 `ExecutionPlan`/`KvCacheView`，按 `GetOrderlyOPList()` 驱动 OpEnv | Executor 无调度策略、无后端专有头文件 | TC-EXEC-01 |
| R-ARCH-09 | Backend/Kernel 契约：能力、缓冲、权重、Kernel、传输、同步；Kernel 使用 `runtime_graph::OpType` | 契约测试通过；新增后端只加目录 | TC-COMPAT-01 |
| R-ARCH-10 | Engine 对外接口：Load/Start/AddRequest/CancelRequest/Update/RunOnce/RunUntilIdle/Destroy/SetTokenCallback/SetInput/SetOutput | 接口可重复调用；线程安全边界清晰；错误可诊断 | TC-ENG-01 |
| R-ARCH-11 | Destroy 幂等、逆序、收集首个错误；加载失败复用同一回滚路径 | 100 次 Load/Destroy 无泄漏；重复 Destroy 安全 | TC-ENG-01 |
| R-ARCH-12 | 可观测性：StepStats/EngineStats/SchedulerStats/KvCacheStats、请求快照 | 日志/统计可被脚本解析 | TC-ENG-01 |
| R-ARCH-13 | 阶段三扩展点预留：投机解码、异步调度、HiCache、多类型 KV | 字段/接口存在且不破坏现有实现 | TC-ARCH-02 |

### 2.2 非功能需求

| 编号 | 类别 | 要求 |
| --- | --- | --- |
| NFR-01 | 性能 | 初始目标：Decode ≥10 token/s、128 Token Prefill ≥100 token/s、首 Token ≤2s、4 并发 ≥1.5×（待真机校准） |
| NFR-02 | 内存 | 单会话峰值 ≤3 GB（含权重、KV、Activation、I/O） |
| NFR-03 | 可移植 | 新后端只加 `Runtime/backend/<name>/`，上层零改动 |
| NFR-04 | 可测试 | 主机侧可单测；无设备即可验证 Loader/Scheduler/KVCache/Engine 逻辑 |
| NFR-05 | 健壮性 | 非法包、OOM、取消、设备错误均返回错误码，不崩溃；1 小时压测 RSS 增长 ≤5% |
| NFR-06 | 工程 | Google C++ Style；模块/类/函数中文注释；四件套完整 |

### 2.3 约束条件

| 编号 | 约束 |
| --- | --- |
| C-01 | C++17；不链接 LibTorch；运行时不进 Python |
| C-02 | 单进程、单设备、单推理线程（默认） |
| C-03 | Graph 是唯一可执行 IR；不引入第二套图表示 |
| C-04 | 静态图，单步 S 维上限 = manifest 的 `max_num_tokens`；`context_length` 来自 manifest，仅作停止上限 |
| C-05 | KV block size 由后端 Attention tile 决定，不写死模型常量 |
| C-06 | 阶段一/二不实现投机解码、异步调度、HiCache、多类型 KV；只冻结扩展点 |
| C-07 | Backend 由 Engine 串行调用，不要求内部线程安全 |
| C-08 | 跨终端接口变更必须先改本文件与契约清单，双方确认后改代码 |
| C-09 | 本阶段只交付接口声明，不实现 `.cc`；头文件必须可被 `-fsyntax-only` 检查 |

### 2.4 验收标准

| 编号 | 通过条件 |
| --- | --- |
| AC-01 | 全部接口头文件可被 C++17 编译器以 `-fsyntax-only` 通过 |
| AC-02 | 接口清单与本文件一致；Engine 成员与调用关系可逐条追溯 |
| AC-03 | ExecBackend 与 ToolChain 契约项（附录 B）逐项确认 |
| AC-04 | 后续阶段：Qwen3 包加载成功，IR 哈希与 manifest 一致 |
| AC-05 | 后续阶段：单请求贪心 64 Token 与金标准一致率 ≥95% |
| AC-06 | 后续阶段：100 次 Load/Destroy 无泄漏；四个模块四件套齐全 |

---

## 3. 概要设计

### 3.1 分层架构

```text
┌──────────────────────────── Service（阶段二，本文只定义接口） ────────────────────────────┐
│ Tokenizer(CPU) → Session → PrefixCache → Engine                                          │
└───────────────────────────────────────┬──────────────────────────────────────────────────┘
                                        │ Engine 对外 API（token 级）
┌───────────────────────────────────────▼──────────────────────────────────────────────────┐
│ Engine（总接口/门面/生命周期/推理循环/IO 绑定）                                            │
│  成员指针：ILoader / Graph / IScheduler / IExecutor / IKVCache / IDestroyer               │
│  内部辅助：ResourceStack                                                                  │
└───────┬───────────────┬───────────────┬───────────────┬───────────────┬──────────────────┘
        │               │               │               │               │
   ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
   │ ILoader │    │IScheduler │   │ IKVCache  │   │ IExecutor │   │ IDestroyer│
   └────┬────┘    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └───────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
   Graph/Backend    KvAllocator     BlockPool      Backend（Executor 拥有）
   OpEnvFactory                                    Graph::GetOrderlyOPList
                                                   → Op::GetOpEnv
                                                   → OpEnv::Forward
                                                   → Kernel::Execute
                                                   → Backend 设备调用
```

### 3.2 Engine 组合与接口注入

Engine 是 Runtime 的唯一总接口；核心模块以 `std::unique_ptr<接口>` 成员持有，便于替换实现（例如不同版本的 Scheduler）。

```cpp
class Engine final : public IoBinder {
 private:
  EngineState state_ = EngineState::kCreated;

  std::unique_ptr<ILoader> loader_;                  // 可注入；加载期使用
  std::unique_ptr<runtime_graph::Graph> graph_;      // Loader 产出，Engine 持有
  std::unique_ptr<IScheduler> scheduler_;            // 可注入
  std::unique_ptr<IExecutor> executor_;              // 可注入；拥有 Backend
  std::unique_ptr<IKVCache> kv_cache_;               // Engine 内部创建并持有
  std::unique_ptr<IDestroyer> destroyer_;            // 可注入
  std::unique_ptr<ResourceStack> resource_stack_;    // 内部辅助
};
```

注意：**Backend 不是 Engine 的成员**；它由 Loader 创建、以 `unique_ptr` 移交给 Executor 持有。OpEnvFactory 也不是 Engine 成员，而是 Loader 的内部工具。

注入机制：

```cpp
// 定义于 Runtime/engine/engine_modules.h；单独成头，避免 engine.h 引入
// Loader/Backend 依赖。
struct EngineModules {
  std::unique_ptr<ILoader> loader;
  std::unique_ptr<IScheduler> scheduler;
  std::unique_ptr<IExecutor> executor;
  std::unique_ptr<IDestroyer> destroyer;
};

// 仅在 CREATED 状态可调用；未注入的模块由 Engine 创建默认实现。
Status Engine::SetModules(EngineModules modules);
```

规则：

1. `SetModules` 只在 `CREATED` 状态有效；`LOADED/RUNNING` 返回 `kStateError`。
2. Loader/Scheduler/Executor/Destroyer 可注入；未注入时由 Engine 创建默认实现。KVCache 由 Engine 内部创建，不对外注入。
3. **Loader 与 Destroyer 成对替换**：Loader 负责资源 Init（创建 Graph/Backend/OpEnv/缓冲），Destroyer 负责 UnInit（逆序释放）；自定义 Loader 应同时提供对应的 Destroyer，或确保所有资源都登记到 ResourceStack。
4. `Graph` 与 `Backend` 依赖模型包，不能预先注入：`ILoader::Load` 产出后，Graph 移交 Engine，Backend 移交 Executor。
5. `OpEnvFactory` 由 Loader 内部创建并用于逐 op 生成 OpEnv；Engine 不持有该工厂。
6. `ResourceStack` 由 Engine 持有并传给 Loader/KVCache/Destroyer，负责逆序回滚与释放。
7. Backend/KVCache/OpEnvFactory 不出现在 Engine 对外接口签名中。

### 3.3 模块划分与所有权

| 模块 | 目录 | 归属 | 统一接口 | 职责 | 不负责 |
| --- | --- | --- | --- | --- | --- |
| Common | `Runtime/common/` | RuntimeCore | 数据对象 | 错误码、配置、请求、计划、KV/IO/OpEnv 契约、资源栈、工具 | 设备、调度、KV 策略 |
| Loader | `Runtime/loader/` | RuntimeCore | `ILoader` | 包校验、图重建、IR 哈希、后端初始化、权重上传、缓冲与 OpEnv 创建 | 调度、KV 策略、设备 API 直调 |
| Scheduler | `Runtime/scheduler/` | RuntimeCore | `IScheduler` | 请求队列、IFB 变体组批、chunked prefill、准入、停止判定、抢占、统计 | 设备、Kernel、KV 内部布局 |
| KVCache | `Runtime/kvcache/` | RuntimeCore | `IKVCache : KvAllocator` | 块池、block table、slot、容量与不变量、阶段二/三扩展 | 抢占决策、图执行 |
| Engine | `Runtime/engine/` | RuntimeCore | `Engine` | 总接口、状态机、推理循环、线程、IO 绑定、模块装配 | 算法实现 |
| Destroy | `Runtime/destroy/` | RuntimeCore | `IDestroyer` | 逆序释放、幂等、回滚 | 资源创建 |
| Executor | `Runtime/executor/` | ExecBackend | `IExecutor` | 消费 plan、绑定 I/O、按拓扑序驱动 OpEnv、回读 token | 调度、KV 分配、后端专有分支 |
| Backend | `Runtime/backend/` | ExecBackend | `Backend` | 设备/上下文/缓冲/传输/权重转换/Kernel/同步/能力 | 调度、KV 策略、图语义 |
| Kernel | `Runtime/backend/kernel.h` | ExecBackend | `Kernel` | 单节点执行、`KernelContext` | 图执行、内存管理 |
| Graph | `Graph/` | ToolChain | `Graph`（类接口） | 图容器、构图、SortOp、查询、GraphEnv/OpEnv | 执行、显存分配、调度 |

依赖方向：Common ← 所有模块；Loader → Graph/Backend/OpEnvFactory；Scheduler → KvAllocator；Engine → ILoader/Graph/IScheduler/IExecutor/IKVCache/IDestroyer；Executor → Graph/Backend/Kernel；Backend/Kernel → Common。**Backend 由 Executor 持有**；禁止反向依赖。

### 3.4 关键流程

**加载流程**

```text
Engine.Load(config)
  → 确保模块存在（Loader/Scheduler/Executor/Destroyer 未注入则创建默认；KVCache 内部创建）
  → loader_->Load(config, resource_stack, &result)
      1 读头/校验 magic/version/manifest SHA
      2 JSON 解析 manifest
      3 统计交叉校验
      4 按 manifest 顺序重放构图 + SortOp
      5 DumpIr + SHA-256 比对（绑指针之前）
      6 BackendRegistry::Create → Init → Capability 校验
      7 权重流式上传（可选逐项 SHA-256）
      8 分配 arena / I/O 缓冲并绑定 storage
      9 GraphEnv + 逐 op OpEnv（Loader 内部 op_env_factory）→ Init 登记 workspace → 分配回填
  → graph_ = std::move(result.graph)
  → kv_cache_->Init(result.backend.get(), spec, kv_budget, resource_stack)  // 借用
  → scheduler_->Init(scheduler_config, kv_cache_.get(), result.model)
  → executor_->Init(std::move(result.backend), graph_.get(), this)          // 所有权移交
  → state = LOADED
任一步失败：destroyer_->Destroy(...) 逆序回滚，state = CREATED
```

回滚上下文：Executor 尚未 Init 时，`DestroyContext.backend` 取
`result.backend.get()`、`executor_initialized=false`，由 Destroyer 直接
`Backend::Deinit`；Executor 已 Init 时，`backend` 取 `executor_->backend()`、
`executor_initialized=true`，由 `Executor::Deinit` 释放 Backend。

**单步推理**

```text
Engine.RunOnce
  → scheduler_->Schedule(&plan)                 // 准入 + KV Reserve + chunk
  → kv_cache_->BuildView(&kv_view)
  → executor_->Execute(plan, kv_view, &step_result)
  → 逐请求 scheduler_->Update(id, tokens, status, &update_result)
       └─ 终态：Scheduler 通过 KvAllocator::Release 释放 KV
  → update_result.finished：TokenCallback
  → StepStats
```

**请求生命周期**

```text
WAITING ──Schedule──▶ RUNNING ──停止──▶ FINISHED
   ▲                     │
   │                     ├──Cancel/Timeout──▶ CANCELLED
   │                     ├──Error───────────▶ FAILED
   └──Preempt(recompute)─┘
```

**销毁流程**

```text
Engine.Destroy
  → 停推理线程 / join
  → destroyer_->Destroy(context)
       → Synchronize（经 Executor::backend()）
       → 逐 op ClearOpEnv()（Kernel 经 Backend 释放）
       → ResourceStack.Clear()（KV/arena/IO/权重缓冲经 Backend 释放）
       → Graph.ClearGraphEnv()
       → Executor.Deinit()（Backend.Deinit + 释放 Backend）
  → DESTROYED
```

### 3.5 逐层调用链（Engine → 算子）

```text
Engine::RunOnce
  → IScheduler::Schedule
  → IKVCache::BuildView
  → IExecutor::Execute
      → IoBinder::SetInput / SetOutput
      → Graph::GetOrderlyOPList()
      → Op::GetOpEnv()
      → OpEnv::Forward()
          → Kernel::Execute(KernelContext)
              → Backend 设备调用（由 Kernel 实现）
      → Backend::Synchronize
      → 读 sample.out
  → IScheduler::Update
      → 终态：KvAllocator::Release
  → TokenCallback
```

约束：

- Engine 不直接调用 OpEnv/Kernel，也不直接引用 Backend；
- Executor 是唯一遍历 `GetOrderlyOPList()` 并调用 `OpEnv::Forward()` 的对象；
- Backend 不知道 Graph/OpEnv，只提供设备能力与 Kernel 工厂；
- Kernel 不知道 Graph/Scheduler 存在；
- KV 的 Reserve/Release 都发生在 Scheduler 内（经 `KvAllocator`）。

### 3.6 线程模型

| 模式 | 行为 | 使用场景 |
| --- | --- | --- |
| 单线程（默认） | 调用方调 `RunOnce` 驱动 Step；`AddRequest/Cancel` 线程安全 | 单测、调试、确定性回归 |
| 推理线程（`threaded=true`） | Engine 启动唯一推理线程循环 Step；回调在该线程触发；`Destroy` join | 阶段二 Service |

规则：Backend 只被推理线程调用；`SetInput/SetOutput` 只允许推理线程；`TokenCallback` 不得阻塞或重入 Engine；`RunOnce` 只允许一个调用者。

### 3.7 错误处理模型

- 对外统一 `Status`；异常不穿越模块边界与 JNI/FFI。
- 单请求错误 → `FAILED`，不阻塞其他请求；设备级错误 → 本步全部失败 + Engine fatal 标记。
- 加载失败 → 逆序回滚 → 可重试（state 回 CREATED）。
- 错误消息带 `模块::阶段` 前缀 + request_id/op 名。

### 3.8 技术选型与关键决策

| 编号 | 决策 | 选择 | 理由 |
| --- | --- | --- | --- |
| D1 | 模块组合 | Engine 以 `unique_ptr<接口>` 持有全部模块 | 可替换实现、可 mock、边界清晰 |
| D2 | 调度范式 | 赤字调度（`num_computed_tokens` 真源）+ `is_chunk` 标记 | vLLM 验证；避免阶段状态与位置真源分裂 |
| D3 | 组批顺序 | decode 保底 + prefill 填充 | 静态图单步预算有限；避免长 prefill 饿死 decode |
| D4 | 上下文与 KV 容量 | 启动期校验；单请求上限 `min(context_length, kv_max_tokens)` | 避免运行期“永远排不上队” |
| D5 | 抢占 | v1 不做运行中 retract；准入+排队；recompute 仅兜底 | 手机上重算代价极高 |
| D6 | 前缀缓存（阶段二） | 块哈希链 + LRU（vLLM 式） | C++ 简单、与 BlockPool 天然匹配、无需树分裂/COW |
| D7 | 线程默认 | 单线程 `RunOnce` | 确定性测试优先 |
| D8 | KV block size | 由后端 Attention tile 决定（契约 C9） | 不绑定模型；由 Kernel 优化决定 |
| D9 | 预算 | `max_num_tokens` 驱动；`context_length` 仅停止上限 | 与 vLLM 一致；换模型只改配置 |
| D10 | 阶段三功能 | 投机解码/异步调度/HiCache/多类型 KV 只留扩展点 | 跟随模型与性能需求 |
| D11 | IR 哈希 | Loader 重建后自校验 | 防 manifest 与 C++ 构图语义漂移 |

---

## 4. 详细设计

### 4.1 公共数据对象（`Runtime/common/`）

#### 4.1.1 错误码 `status.h`

在现有基础上**只追加**（不重编号）：

```cpp
enum class ErrorCode {
  kOk = 0, kInvalidArgument, kStateError, kUnsupportedOp, kVersionMismatch,
  kBufferError, kDeviceError, kKernelError, kInternal, kUnsupportedDtype,
  kUnsupportedLayout,
  // 追加
  kInvalidModelFile, kIoError, kDeviceOom, kUnsupportedBackend,
  kBackendInitError, kRequestNotFound, kTimeout, kCancelled,
};
```

#### 4.1.2 配置 `runtime_config.h` / `model_config.h`

```cpp
struct RuntimeConfig {
  std::string backend = "ocl";
  int32_t device_id = 0;
  uint32_t queue_count = 1;
  uint64_t memory_budget_bytes = 0;      // 0 = 后端/设备默认
  uint64_t kv_cache_bytes = 0;           // 0 = 由预算自动推导
  int32_t max_num_tokens = 0;            // 0 = manifest；不得 > 图 S
  int32_t max_concurrent_requests = 0;   // 0 = min(max_num_tokens, 8)
  int32_t kv_block_size = 0;             // 0 = 后端 attention tile
  bool verify_weights_sha256 = true;
  bool allow_metadata_only = false;
  bool threaded = false;
  int32_t max_concurrent_batches = 1;    // 阶段三使用，v1 固定 1
};

struct ModelConfig {                     // 来自 manifest metadata
  int32_t num_layers = 0, num_heads = 0, num_kv_heads = 0, head_dim = 0;
  int64_t hidden_size = 0, intermediate_size = 0, vocab_size = 0;
  int64_t context_length = 0;            // 仅停止上限
  float rms_norm_eps = 0.0f;
  double rope_theta = 0.0;
  DataType dtype = DataType::kUnknown;
  int32_t max_num_tokens = 0, max_batch_size = 0;
  int64_t pad_token_id = 0, bos_token_id = 0;
  std::vector<int64_t> stop_token_ids;
};
```

#### 4.1.3 请求 `request.h`

```cpp
enum class RequestState { kWaiting, kRunning, kPreempted, kFinished,
                          kCancelled, kFailed };

struct SamplingParams {
  double temperature = 0.0;
  int32_t top_k = 0;
  double top_p = 0.0;
  double repetition_penalty = 1.0;
  uint64_t seed = 0;
  std::vector<int64_t> stop_ids;
};

struct Request {
  uint64_t id = 0;                       // 0 = Engine 分配
  std::vector<int64_t> prompt_ids;
  int32_t max_new_tokens = 0;            // 0 = 由上下文上限推导
  SamplingParams sampling;               // v1 只记录，图内固定 argmax
  int32_t priority = 0;
  uint64_t deadline_ns = 0;              // 0 = 不超时
};

struct RequestSnapshot {
  uint64_t request_id = 0;
  RequestState state = RequestState::kWaiting;
  int32_t prompt_tokens = 0;
  int32_t computed_tokens = 0;
  int32_t generated_tokens = 0;
};
```

#### 4.1.4 计划与结果 `plan.h`

```cpp
enum class RequestPhase { kPrefill, kDecode };

struct PlanEntry {
  uint64_t request_id = 0;
  RequestPhase phase = RequestPhase::kPrefill;
  int32_t packed_offset = 0;             // 在 packed token 中的起点
  int32_t token_count = 0;
  std::vector<int64_t> token_ids;
  std::vector<int64_t> positions;
  std::vector<int64_t> slots;            // KV 写入 slot
  std::vector<int32_t> block_table;
  int32_t context_length = 0;            // 本步之前的 prompt+generated
  int32_t output_slot = -1;              // 取 sample.out 的打包下标
  bool is_chunk = false;                 // prefill 未完成
  std::vector<int64_t> draft_token_ids;  // 阶段三预留
};

struct ExecutionPlan {
  uint64_t step_id = 0;
  int32_t total_tokens = 0;              // ≤ max_num_tokens
  std::vector<PlanEntry> entries;
};

struct RequestStepResult {
  uint64_t request_id = 0;
  int64_t token = 0;
  bool ok = true;
  ErrorCode error = ErrorCode::kOk;
};

struct StepResult { std::vector<RequestStepResult> requests; };

struct StepStats {
  uint64_t step_id = 0;
  int32_t total_tokens = 0, prefill_tokens = 0, decode_tokens = 0;
  int32_t running_requests = 0, waiting_requests = 0;
  int64_t schedule_ns = 0, execute_ns = 0;
  int64_t kv_used_blocks = 0, kv_free_blocks = 0;
  int32_t preempted_requests = 0;
};
```

不变量：`total_tokens == Σ entry.token_count ≤ max_num_tokens`；`entry.slots.size() == token_count`；`output_slot` 仅在产生 token 时有效。

#### 4.1.5 KV 契约 `kv_allocator.h` / `kv_view.h`

```cpp
class KvAllocator {
 public:
  virtual ~KvAllocator() = default;
  virtual Status Reserve(uint64_t request_id, int64_t new_tokens) = 0;
  virtual void Release(uint64_t request_id) = 0;
  virtual int64_t FreeBlocks() const = 0;
  virtual int32_t BlockSize() const = 0;
  virtual int64_t MaxTotalTokens() const = 0;
  virtual Status GetBlockTable(uint64_t request_id,
                               std::vector<int32_t>* table) const = 0;
  virtual Status GetSlots(uint64_t request_id, int64_t first_token,
                          int64_t count, std::vector<int64_t>* slots) const = 0;
};

enum class KvCacheType { kFullAttention, kSlidingWindow, kMla };

struct KvCacheSpec {
  KvCacheType type = KvCacheType::kFullAttention;
  int32_t block_size = 0;      // 后端 attention tile，非模型常量
  int32_t num_layers = 0, num_kv_heads = 0, head_dim = 0;
  DataType dtype = DataType::kFloat16;
  int32_t sliding_window = 0;  // full attention 为 0
};

struct KvLayerView { void* k_base = nullptr; void* v_base = nullptr;
                     int64_t block_stride = 0; };
struct KvCacheView {
  std::vector<KvLayerView> layers;
  int32_t block_size = 0;
  int64_t max_total_tokens = 0;
};

struct KvCacheStats {
  int32_t block_size = 0;
  int64_t total_blocks = 0, free_blocks = 0, max_total_tokens = 0;
};
```

#### 4.1.6 其他公共契约

```cpp
class IoBinder {
 public:
  virtual ~IoBinder() = default;
  virtual Status SetInput(const std::string& name,
                          const runtime_graph::Storage& host_storage) = 0;
  virtual Status SetOutput(const std::string& name,
                           const runtime_graph::Storage& host_storage) = 0;
};

class OpEnvFactory {
 public:
  virtual ~OpEnvFactory() = default;
  virtual Status Create(runtime_graph::Op* op, Backend* backend,
                        runtime_graph::GraphEnv* graph_env,
                        std::unique_ptr<runtime_graph::OpEnv>* out) = 0;
};
std::unique_ptr<OpEnvFactory> CreateDefaultOpEnvFactory();

class ResourceStack {
 public:
  using Release = std::function<Status()>;
  void Push(std::string label, Release release);
  Status Clear();          // LIFO、幂等、返回首个错误
  size_t size() const;
};
```

其中 `OpEnvFactory` 由 Loader 内部创建与使用，不作为 Engine 成员；`ResourceStack` 由 Engine 持有并传给 Loader/KVCache/Destroyer。

### 4.2 模块统一接口

#### 4.2.1 `ILoader`（`Runtime/loader/loader.h`）

```cpp
struct LoadConfig { std::string package_dir; RuntimeConfig runtime; };

struct LoadResult {
  std::unique_ptr<runtime_graph::Graph> graph;
  std::unique_ptr<Backend> backend;
  ModelConfig model;
  uint64_t weights_bytes = 0;
  uint64_t arena_bytes = 0;
  uint64_t io_bytes = 0;
};

class ILoader {
 public:
  virtual ~ILoader() = default;
  virtual Status Load(const LoadConfig& config, ResourceStack* resource_stack,
                      LoadResult* result) = 0;
};
std::unique_ptr<ILoader> CreateDefaultLoader();
```

说明：Loader 在内部创建 `OpEnvFactory`（`CreateDefaultOpEnvFactory`，ExecBackend 实现），不向 Engine 暴露；`LoadResult.backend` 由 Engine 移交给 Executor。

#### 4.2.2 `IScheduler`（`Runtime/scheduler/scheduler.h`）

```cpp
struct SchedulerConfig {
  int32_t max_num_tokens = 0;
  int32_t max_concurrent_requests = 0;
  int32_t max_prefill_chunk = 0;
  int64_t context_length = 0;
  std::vector<int64_t> stop_ids;
};

struct SchedulerStats {
  int32_t waiting_requests = 0, running_requests = 0, preempted_requests = 0;
  int64_t free_blocks = 0, total_blocks = 0;
};

struct UpdateResult {
  uint64_t request_id = 0;
  RequestState state = RequestState::kRunning;
  bool finished = false;
};

class IScheduler {
 public:
  virtual ~IScheduler() = default;
  virtual Status Init(const SchedulerConfig& config, KvAllocator* kv_allocator,
                      const ModelConfig& model) = 0;
  virtual Status Merge(const Request& request, uint64_t* request_id) = 0;
  virtual Status Schedule(ExecutionPlan* plan) = 0;
  virtual Status Update(uint64_t request_id,
                        const std::vector<int64_t>& tokens,
                        RequestState executor_status,
                        UpdateResult* result) = 0;
  virtual Status Stop(uint64_t request_id) = 0;
  virtual Status Fail(uint64_t request_id, ErrorCode code) = 0;
  virtual std::vector<RequestSnapshot> Snapshot() const = 0;
  virtual SchedulerStats Stats() const = 0;
  virtual bool HasRequests() const = 0;
};
std::unique_ptr<IScheduler> CreateDefaultScheduler();
```

说明：Scheduler 持有 `KvAllocator*`，在 `Update/Stop/Fail` 使请求进入终态时调用 `KvAllocator::Release`；Engine 不重复释放。实现必须保证 `Merge/Stop/HasRequests` 可被调用方线程与推理线程并发调用（内部加锁）；`Schedule/Update/Stats` 由推理线程调用。

#### 4.2.3 `IKVCache`（`Runtime/kvcache/kv_cache_manager.h`）

```cpp
class IKVCache : public KvAllocator {
 public:
  virtual Status Init(Backend* backend, const KvCacheSpec& spec,
                      uint64_t kv_budget_bytes,
                      ResourceStack* resource_stack) = 0;
  virtual Status BuildView(KvCacheView* view) const = 0;
  virtual KvCacheStats Stats() const = 0;
};
std::unique_ptr<IKVCache> CreateDefaultKVCache();
```

说明：`IKVCache::Init` 的 `Backend*` 为借用指针；Backend 生命周期由 Executor 保证。

#### 4.2.4 `IExecutor`（`Runtime/executor/executor.h`，ExecBackend 实现）

```cpp
class IExecutor {
 public:
  virtual ~IExecutor() = default;
  virtual Status Init(std::unique_ptr<Backend> backend,
                      runtime_graph::Graph* graph,
                      IoBinder* io_binder) = 0;
  virtual Backend* backend() = 0;   // 非拥有访问；未初始化返回 nullptr
  virtual Status Execute(const ExecutionPlan& plan,
                         const KvCacheView& kv_view,
                         StepResult* result) = 0;
  virtual Status Deinit() = 0;      // Backend::Deinit + 释放 Backend
};
std::unique_ptr<IExecutor> CreateDefaultExecutor();
```

说明：Executor 是 Backend 的唯一所有者；`Deinit` 负责 `Backend::Deinit` 并释放；OpEnv/KVCache 只借用 Backend。`Init` 失败时必须释放已接管的 Backend，不得把半初始化资源留给 Engine。

#### 4.2.5 `IDestroyer`（`Runtime/destroy/destroyer.h`）

```cpp
struct DestroyContext {
  IScheduler* scheduler = nullptr;
  IKVCache* kv_cache = nullptr;
  IExecutor* executor = nullptr;
  runtime_graph::Graph* graph = nullptr;
  Backend* backend = nullptr;   // 借用；Executor 拥有时由 executor->backend() 提供
  ResourceStack* resource_stack = nullptr;
  bool backend_initialized = false;
  bool executor_initialized = false;
};

class IDestroyer {
 public:
  virtual ~IDestroyer() = default;
  virtual Status Destroy(const DestroyContext& context) = 0;
};
std::unique_ptr<IDestroyer> CreateDefaultDestroyer();
```

### 4.3 Backend 与 Kernel 接口（ExecBackend 实现）

```cpp
class Buffer {
 public:
  virtual ~Buffer() = default;
  virtual void* DevicePtr() const = 0;
  virtual uint64_t Bytes() const = 0;
};

enum class BufferUsage { kPersistent, kActivation, kIo, kWorkspace, kKv };
enum class KvLayout { kNhd, kHnd };

struct BufferDesc {
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  BufferUsage usage = BufferUsage::kPersistent;
};

struct BackendConfig {
  int32_t device_id = 0;
  uint32_t queue_count = 1;
  std::map<std::string, std::string> options;
};

struct ParamDesc {
  std::string name;
  const void* host_ptr = nullptr;
  uint64_t bytes = 0;
  DataType source_dtype = DataType::kUnknown;
  DataType target_dtype = DataType::kUnknown;
};

struct BackendCapability {
  std::vector<DataType> supported_dtypes;
  std::vector<runtime_graph::OpType> supported_ops;
  bool h2d = false;
  bool d2h = false;
  uint64_t max_buffer_bytes = 0;
  uint64_t min_alignment = 0;
  int32_t attention_block_size = 0;   // KV 块大小来源
  KvLayout kv_layout = KvLayout::kNhd;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual Status Init(const BackendConfig& config) = 0;
  virtual Status Deinit() = 0;
  virtual Status Capability(BackendCapability* capability) const = 0;
  virtual Status CreateBuffer(const BufferDesc& desc,
                              std::unique_ptr<Buffer>* buffer) = 0;
  virtual Status DestroyBuffer(std::unique_ptr<Buffer> buffer) = 0;
  virtual Status ConvertParam(const ParamDesc& desc,
                              std::unique_ptr<Buffer>* buffer) = 0;
  virtual Status Upload(void* device_ptr, const void* host_ptr,
                        uint64_t bytes, Stream* stream) = 0;
  virtual Status Download(void* host_ptr, const void* device_ptr,
                          uint64_t bytes, Stream* stream) = 0;
  virtual Status CreateKernel(const KernelDesc& desc,
                              std::unique_ptr<Kernel>* kernel) = 0;
  virtual Status DestroyKernel(std::unique_ptr<Kernel> kernel) = 0;
  virtual Status Synchronize(Stream* stream) = 0;
};

struct KernelDesc {
  runtime_graph::OpType op_type = runtime_graph::OpType::kUnknown;
  DataType precision = DataType::kUnknown;
  std::string kernel_key;
};

class Kernel {
 public:
  virtual ~Kernel() = default;
  virtual Status Execute(const runtime_graph::KernelContext& context) = 0;
};
```

### 4.4 Graph 接口边界

Graph 由 ToolChain 维护，Runtime 只使用以下接口，不修改：

| 接口 | 用途 | 调用方 |
| --- | --- | --- |
| `CreateConstant/CreateInput/CreateOp/CreateSample/CreateOutput/SortOp` | 重建图 | Loader |
| `GetAllConstantTensor/GetAllInputOp/GetAllOutputOp/GetSampleOp/GetOrderlyOPList` | 查询 | Loader/Executor/Engine |
| `SetGraphEnv/GetGraphEnv/ClearGraphEnv` | GraphEnv 生命周期 | Loader/Destroy |
| `Op::SetOpEnv/ClearOpEnv/GetOpEnv` | OpEnv 挂载 | Loader/Executor/Destroy |
| `DumpIr` | IR 哈希 | Loader（需 ToolChain 暴露 C++ 接口） |

#### 4.4.1 OpEnv 与 Backend 的关系

```text
Graph
 └── Op（op_type/attr/src/dst）
      └── OpEnv（unique_ptr，Graph 侧）
           ├── Op*        （非拥有）
           ├── GraphEnv*  （非拥有）
           ├── Backend*   （非拥有）
           └── Kernel     （unique_ptr，由 Backend::CreateKernel 创建）

Backend（Executor 拥有）
 ├── 设备/上下文/队列
 ├── CreateBuffer/DestroyBuffer
 ├── Upload/Download
 └── CreateKernel/DestroyKernel ──► OclKernel/VulkanKernel/...
```

规则：

1. OpEnv 是 Graph IR 的节点执行环境，由 Op 通过 `unique_ptr` 持有；Backend 不知道 OpEnv 的存在。
2. OpEnv 由 Loader 通过 `OpEnvFactory` 创建，`Init(GraphEnv*)` 时经 `Backend::CreateKernel` 获得 Kernel。
3. Executor 按拓扑序调用 `OpEnv::Forward()`；OpEnv 组装 `KernelContext` 并调用 `Kernel::Execute()`；Kernel 使用 Backend 提供的设备资源。
4. Input/Output 的 I/O OpEnv 直接调用 `Backend::Upload/Download`；计算 OpEnv 只调用 Kernel。
5. 销毁顺序：先 `Op::ClearOpEnv()`（Kernel 经 Backend 释放），再释放缓冲，最后 `Backend::Deinit()`；Backend 必须比所有 OpEnv/Kernel 活得更久。
6. 一个通用 `KernelOpEnv` 可服务所有后端；后端差异收敛在 `Kernel` 实现，OpEnv 不直接调用厂商 API。

### 4.5 Engine 对外接口

```cpp
enum class EngineState { kCreated, kLoaded, kRunning, kDestroying, kDestroyed };

struct EngineConfig { std::string package_dir; RuntimeConfig runtime; };

struct EngineStats {
  uint64_t total_steps = 0, total_requests = 0, total_tokens = 0;
  int32_t running_requests = 0, waiting_requests = 0;
  int64_t kv_used_blocks = 0, kv_free_blocks = 0;
  int64_t last_step_ns = 0;
};

using TokenCallback =
    std::function<void(uint64_t request_id, int64_t token, bool finished)>;

class Engine final : public IoBinder {
 public:
  Engine();
  ~Engine() override;

  Status SetModules(EngineModules modules);
  Status Load(const EngineConfig& config);
  Status Start();
  Status AddRequest(const Request& request, uint64_t* request_id);
  Status CancelRequest(uint64_t request_id);
  Status Update(uint64_t request_id, const std::vector<int64_t>& tokens,
                RequestState status = RequestState::kRunning);
  Status RunOnce(StepStats* stats);
  Status RunUntilIdle(uint32_t timeout_ms);
  Status Destroy();
  void SetTokenCallback(TokenCallback callback);

  Status SetInput(const std::string& name,
                  const runtime_graph::Storage& host_storage) override;
  Status SetOutput(const std::string& name,
                   const runtime_graph::Storage& host_storage) override;

  EngineState state() const;
  EngineStats stats() const;
};
```

Engine 对外接口与模块统一接口的对应关系：

| Engine 对外接口 | 内部调用链 |
| --- | --- |
| `SetModules` | 替换 `loader_/scheduler_/executor_/destroyer_` 指针；Loader 与 Destroyer 成对 |
| `Load` | `loader_->Load` → `kv_cache_->Init` → `scheduler_->Init` → `executor_->Init(std::move(backend))` |
| `Start` | 状态迁移；threaded 时启动推理线程 |
| `AddRequest` | `scheduler_->Merge` |
| `CancelRequest` | `scheduler_->Stop` |
| `Update` | `scheduler_->Update` |
| `RunOnce` | `scheduler_->Schedule` → `kv_cache_->BuildView` → `executor_->Execute` → `scheduler_->Update`（终态由 Scheduler 释放 KV） |
| `RunUntilIdle` | 循环 `RunOnce` 直到 `scheduler_->HasRequests()` 为 false |
| `Destroy` | 停线程 → `destroyer_->Destroy`（Executor::Deinit 释放 Backend） |
| `SetInput/SetOutput` | 校验后写入 Input/Output op 的 `attr["storage"]` |
| `stats` | 返回本地累计统计；running/waiting/KV 为最近 Step 的快照 |

### 4.6 模块调用矩阵

| 调用方 → 被调方 | 接口 | 交互数据 | 说明 |
| --- | --- | --- | --- |
| Service → Engine | Load/Start/AddRequest/CancelRequest/Update/RunOnce/RunUntilIdle/Destroy/SetTokenCallback/Stats | Request、StepStats、EngineStats | 唯一对外入口 |
| Engine → ILoader | `Load` | LoadConfig → LoadResult | 加载期一次性 |
| Loader → Graph | Create*/SortOp/DumpIr/SetGraphEnv | Graph 对象、Storage | 重建与校验 |
| Loader → Backend | Init/Capability/CreateBuffer/ConvertParam/Upload；失败回滚时 Deinit | BufferDesc、ParamDesc | 零设备 API |
| Loader → OpEnvFactory | `Create` | Op/Backend/GraphEnv → OpEnv | Loader 内部，逐 op 创建 |
| Loader → ResourceStack | `Push` | 释放闭包 | 逆序回滚 |
| Engine → IScheduler | Init/Merge/Schedule/Update/Stop/Fail/Snapshot/Stats | Request、ExecutionPlan、UpdateResult | Schedule/Update/Stats 推理线程；Merge/Stop/HasRequests 需线程安全 |
| Scheduler → KvAllocator | Reserve/Release/GetBlockTable/GetSlots/FreeBlocks/BlockSize | 块号、slot | 失败零副作用 |
| Engine → IKVCache | Init/BuildView/Stats | KvCacheSpec、KvCacheView | 加载 / 每步 |
| Engine → IExecutor | Init(std::unique_ptr<Backend>)/Execute/Deinit | ExecutionPlan、KvCacheView → StepResult | 加载 / 每步 |
| Executor → Backend | Synchronize | — | 每步结束 |
| Executor → IoBinder | SetInput/SetOutput | host Storage | 每步 |
| Executor → Graph | GetOrderlyOPList/GetSampleOp/GetAllInputOp/GetAllOutputOp | Op/Tensor | 只读 |
| Executor → OpEnv | Forward | 无参（OpEnv 自读 storage） | 拓扑序 |
| OpEnv → Backend | CreateKernel/DestroyKernel；I/O OpEnv：Upload/Download | KernelDesc/Kernel、Storage | 加载 / H2D/D2H |
| OpEnv → Kernel | Execute | KernelContext | 单节点 |
| OpEnv → GraphEnv | GetSharedStorage | workspace | Init 登记、Forward 读 |
| Engine → IDestroyer | Destroy | DestroyContext | 结束/回滚 |
| Destroy → 各模块 | ClearOpEnv/DestroyBuffer/ClearGraphEnv/Deinit/stack.Clear | — | 逆序 |

### 4.7 所有权与生命周期矩阵

| 资源 | 所有者 | 创建者 | 使用者 | 释放者 |
| --- | --- | --- | --- | --- |
| Loader/Scheduler/Executor/KVCache/Destroyer | Engine | Engine 或注入 | — | Engine |
| Graph | Engine | Loader | Executor/OpEnv（借用） | Engine（Destroy） |
| GraphEnv | Graph | Loader | OpEnv | Graph.ClearGraphEnv |
| Op | Graph | Loader | Executor | Graph |
| OpEnv | Op（unique_ptr） | Loader（OpEnvFactory） | Executor | Op.ClearOpEnv |
| Kernel | OpEnv（unique_ptr） | Backend | OpEnv | Backend.DestroyKernel（OpEnv 析构） |
| Backend | **Executor** | Loader（移交） | Loader/KVCache/Executor/OpEnv | Executor::Deinit |
| 设备缓冲 | Engine（ResourceStack 句柄） | Loader/KVCache | OpEnv/Kernel | ResourceStack → Backend |
| 权重常驻缓冲 | Engine | Loader | Kernel | ResourceStack |
| 主机 I/O 缓冲 | Executor | Executor | Backend | Executor |
| Request/Plan | Scheduler | Scheduler | Engine/Executor（值/快照） | Scheduler |
| KV 块 | KVCache | KVCache | Executor（只读快照） | KVCache |
| ResourceStack | Engine | Engine | Loader/KVCache/Destroy | Destroy |

### 4.8 交互数据汇总

| 数据 | 生产者 | 消费者 | 传递方式 | 生命周期 |
| --- | --- | --- | --- | --- |
| `LoadConfig` | Engine | ILoader | 值 | Load 期间 |
| `LoadResult` | ILoader | Engine | 值（Graph 移交 Engine；Backend 再移交 Executor） | Load 后由 Engine/Executor 持有 |
| `Request` | Service/Engine | IScheduler | 值 | 请求生命周期 |
| `ExecutionPlan` | IScheduler | IExecutor/Engine | 值（不可变快照） | 单步 |
| `KvCacheView` | IKVCache | Engine → IExecutor | 值（裸指针视图） | 单步 |
| `StepResult` | IExecutor | Engine | 值 | 单步 |
| `UpdateResult` | IScheduler | Engine | 值 | 单次 Update |
| `DestroyContext` | Engine | IDestroyer | 值（裸指针；Backend 借用自 Executor） | Destroy 期间 |
| `StepStats` | Engine/Executor | Service/日志 | 值 | 单步 |
| `EngineStats` | Engine | Service | 值 | 累计/查询 |

### 4.9 错误码

| 错误码 | 典型场景 |
| --- | --- |
| `kInvalidArgument` | 参数非法、请求字段越界 |
| `kStateError` | 未 Load 就 Run、Destroy 后调用、SetModules 时机错误 |
| `kInvalidModelFile` | 包截断、SHA/IR 哈希不一致、格式不兼容 |
| `kIoError` | 文件读写失败 |
| `kUnsupportedBackend` | 后端未注册 |
| `kBackendInitError` | 设备/Context/Queue 初始化失败 |
| `kUnsupportedOp` | 后端缺少算子或 H2D/D2H |
| `kUnsupportedDtype` | 精度不支持 |
| `kBufferError` | 缓冲分配/传输失败 |
| `kDeviceOom` | KV/Activation/权重显存不足 |
| `kKernelError` | Kernel 编译/执行失败 |
| `kDeviceError` | 设备 API 失败、设备丢失 |
| `kRequestNotFound` | Update/Stop 非法 ID |
| `kTimeout` | 请求超时、设备等待超时 |
| `kCancelled` | 调用方取消 |
| `kInternal` | 不变量破坏、未分类异常 |

### 4.10 对外服务层使用序列

```text
engine.SetModules(...)        // 可选：注入自定义 Scheduler/Executor 等
engine.Load(config)           // CREATED → LOADED
engine.SetTokenCallback(cb)
engine.Start()                // LOADED → RUNNING（threaded 时启动推理线程）

engine.AddRequest(req, &id)   // 线程安全
// 单线程模式：while (...) engine.RunOnce(&stats);
// 线程模式：由 Engine 内部循环；调用方等待回调
engine.CancelRequest(id)      // 线程安全
engine.Update(id, tokens)     // 仅自定义执行路径使用；正常由 RunOnce 内部调用
engine.Destroy()              // 任意非 DESTROYED → DESTROYED
```

线程安全边界：

| 接口 | 线程安全 | 调用线程 |
| --- | --- | --- |
| `AddRequest/CancelRequest/Update/Stats` | 是 | 任意 |
| `Load/Start/Destroy/SetModules` | 否（需串行） | 主控线程 |
| `RunOnce/RunUntilIdle` | 否（单调用者） | 驱动线程 |
| `SetInput/SetOutput` | 否 | 推理线程 |
| `TokenCallback` | 回调在推理线程执行 | 不得阻塞/重入 |

### 4.11 阶段三扩展点（只预留，不实现）

| 功能 | 预留字段/接口 | 实现阶段 |
| --- | --- | --- |
| 投机解码 | `PlanEntry.draft_token_ids`、`Update(tokens[])`、stop 按 accepted 数 | 阶段三 |
| 异步调度 | `RuntimeConfig.max_concurrent_batches`、`plan.step_id`、KV `ReleaseAfter(step_id)` | 阶段三 |
| HiCache | `KvStorageTier`、block `tier/host_slot`、`EvictToHost/ReloadFromHost` | 阶段三 |
| 多类型 KV | `KvCacheSpec.type`、KvCache 多组、KvCacheView 多组 | 阶段三 |

### 4.12 目录与构建

```text
Runtime/
├── .doc/Runtime_design.md          # 本文件
├── CMakeLists.txt                  # 后续阶段创建，汇总子目录
├── common/                         # 数据对象与工具（本阶段：头文件）
├── loader/loader.h                 # ILoader
├── scheduler/scheduler.h           # IScheduler
├── kvcache/kv_cache_manager.h      # IKVCache
├── engine/engine.h                 # Engine
├── engine/engine_modules.h         # EngineModules（可注入模块集合）
├── destroy/destroyer.h             # IDestroyer
├── executor/executor.h             # IExecutor（ExecBackend）
├── backend/backend.h               # Backend/Buffer/BackendCapability（ExecBackend）
├── backend/kernel.h                # Kernel/KernelDesc 公共契约（ExecBackend）
├── backend/kernel_context.h        # KernelContext（ExecBackend）
└── backend/kernel_types.h          # Backend/Kernel 类型映射（ExecBackend）
```

构建：后续阶段由 `Runtime/CMakeLists.txt` 链接 Graph 静态库（`GRAPH_BUILD_PYTHON=OFF`）；各模块独立 target 与测试 target。本阶段只要求头文件可 `-fsyntax-only`。

---

## 5. 实施步骤

| 阶段 | 内容 | 交付 | 依赖 |
| --- | --- | --- | --- |
| I0 | 本文档 + 全部接口头文件 | 设计文档 + 头文件声明 | 无 |
| R0 | common 实现 + 单元测试 | common 四件套 | I0 |
| R1 | Loader 实现（FakeBackend） | loader 四件套 | R0；Graph DumpIr |
| R2 | Scheduler + KVCache 实现 | scheduler/kvcache 四件套 | R0；C9 |
| R3 | Engine + Destroy 实现 | engine/destroy 四件套 | R1/R2；IExecutor 冻结 |
| R4 | 接真实 Executor/OCL，Qwen3 单请求端到端 | 联合测试结果 | ExecBackend C1–C9 |
| R5 | 多请求连续批处理、取消/超时、性能与稳定性 | TC-SCHED/KV/ENG/PERF | R4 |

风险与应对：

| 风险 | 应对 |
| --- | --- |
| 接口冻结后仍需调整 | 变更必须先改本文件与附录 B，双方确认；头文件只做增量 |
| Graph 无 C++ DumpIr | 回退 Runtime 自带实现 + Qwen3 包哈希门槛测试 |
| Backend 接口延期 | FakeBackend 先打通 Loader/Engine 单测 |
| KV 容量与 context 冲突 | 启动期硬校验 + 可配置上限 |

---

## 6. 测试与验证设计

### 6.1 本阶段验证（接口）

| 用例 | 方法 | 通过标准 |
| --- | --- | --- |
| TC-ARCH-01 | 用 C++17 编译器对“包含全部头文件”的 TU 做编译检查 | 零错误、零警告（项目告警级别） |
| TC-ARCH-02 | 逐项核对本文件接口与头文件签名 | 完全一致；扩展点字段存在 |
| TC-ARCH-03 | 依赖方向扫描 | 无反向依赖；common 不依赖 Graph/Backend |
| TC-ARCH-04 | Engine 成员与调用矩阵核对 | Loader/Graph/Scheduler/Executor/KVCache/Destroyer 均为 `unique_ptr`；Backend 归 Executor；OpEnvFactory 归 Loader；调用链可追溯 |
| TC-ARCH-05 | EngineModules 注入对称性检查 | Loader/Scheduler/Executor/Destroyer 均在 `EngineModules` 中；Loader 与 Destroyer 成对可替换 |

验证记录见 [`Runtime/tests/test_results.md`](../tests/test_results.md)。

### 6.2 后续阶段验证（实现）

| 用例 | 覆盖 | 通过标准 |
| --- | --- | --- |
| TC-LOAD-01 | 正常包、截断、magic/版本/哈希错误、IR 不一致、权重缺失、分配失败 | 错误码正确；逆序回滚；句柄归零 |
| TC-SCHED-01/02 | 2/4/8 并发、长短混合、取消、超时、预算 | 无串扰；预算不超；取消及时 |
| TC-KV-01 | 随机分配/追加/释放 10,000 次、OOM、阶段二共享 | 无越界/重复释放；OOM 可恢复 |
| TC-EXEC-01 | 单步 plan 驱动、I/O 绑定、token 回读 | token 一致；无调度逻辑 |
| TC-ENG-01 | Load/Destroy ×100、非法状态、重复 Destroy、中途失败 | 无泄漏；幂等；错误码正确 |
| TC-ROBUST-01 | 设备错误、Forward 失败、绑定缺失 | 不崩溃；可定位；可回收 |
| TC-COMPAT-01 | mock 后端接入 | 只加目录 + 配置；上层零改动 |

环境：Windows 11 / Linux 主机，C++17，CMake 3.20+，FakeBackend；真机 Android + OCL 用于 R4/R5。测试结果写入各模块 `tests/test_results.md`。

---

## 附录 A：接口索引

| 头文件 | 接口/数据 |
| --- | --- |
| `Runtime/common/status.h` | `Status`、`ErrorCode` |
| `Runtime/common/data_types.h` | `DataType/Layout/DeviceType` |
| `Runtime/common/runtime_config.h` | `RuntimeConfig` |
| `Runtime/common/model_config.h` | `ModelConfig` |
| `Runtime/common/request.h` | `Request/SamplingParams/RequestState/RequestSnapshot` |
| `Runtime/common/plan.h` | `ExecutionPlan/PlanEntry/StepResult/StepStats` |
| `Runtime/common/kv_allocator.h` | `KvAllocator` |
| `Runtime/common/kv_view.h` | `KvCacheSpec/KvCacheView/KvCacheStats` |
| `Runtime/common/io_binder.h` | `IoBinder` |
| `Runtime/common/op_env_factory.h` | `OpEnvFactory` |
| `Runtime/common/resource_stack.h` | `ResourceStack` |
| `Runtime/loader/loader.h` | `ILoader/LoadConfig/LoadResult` |
| `Runtime/scheduler/scheduler.h` | `IScheduler/SchedulerConfig/SchedulerStats/UpdateResult` |
| `Runtime/kvcache/kv_cache_manager.h` | `IKVCache` |
| `Runtime/engine/engine.h` | `Engine/EngineConfig/EngineStats` |
| `Runtime/engine/engine_modules.h` | `EngineModules`（Loader/Scheduler/Executor/Destroyer 注入） |
| `Runtime/destroy/destroyer.h` | `IDestroyer/DestroyContext` |
| `Runtime/executor/executor.h` | `IExecutor` |
| `Runtime/backend/backend.h` | `Backend/Buffer/BackendCapability` |
| `Runtime/backend/kernel.h` | `Kernel/KernelDesc` |
| `Runtime/backend/kernel_context.h` | `KernelContext` |
| `Runtime/backend/kernel_types.h` | `DataTypeFromPrecision/PrecisionFromDataType/IsSupportedDataType` |

## 附录 B：跨终端契约

**与 ExecBackend（C1–C9）**

| 编号 | 契约 |
| --- | --- |
| C1 | Backend 全生命周期与 Buffer 接口（Init/Deinit/Capability/CreateBuffer/DestroyBuffer/ConvertParam/Upload/Download/CreateKernel/DestroyKernel/Synchronize） |
| C2 | `KernelDesc.op_type` 与 `BackendCapability.supported_ops` 统一使用 `runtime_graph::OpType` |
| C3 | 枚举去重：图侧使用 `runtime_graph::{Precision,Layout,DeviceType}`；`DataType` 仅作后端边界类型 |
| C4 | `OpEnvFactory` 由 ExecBackend 实现并暴露 `CreateDefaultOpEnvFactory()`；由 Loader 内部创建使用，不作为 Engine 成员 |
| C5 | `IExecutor` 按本文件签名实现（含 `Init(std::unique_ptr<Backend>)` 与 `backend()`）；持有 Backend 所有权；不释放 KV、不做停止判定 |
| C6 | KV/workspace 契约：每层 K/V 视图、slot 公式、GraphEnv shared storage 登记与回填 |
| C7 | 错误码只接受 `Runtime/common/status.h` 的追加式扩展 |
| C8 | 构建分工：RuntimeCore 维护 `Runtime/CMakeLists.txt`；ExecBackend 提供 executor/backend（含 backend/kernel）子构建 |
| C9 | `BackendCapability.attention_block_size` 与 `kv_layout` 由 Attention Kernel 声明；KVCache 据此计算块大小与容量 |

**与 ToolChain（G1–G2）**

| 编号 | 契约 |
| --- | --- |
| G1 | Graph 暴露 C++ `DumpIr`，Python 绑定复用同一实现 |
| G2 | 图格式语义冻结：`graph.ops` 创建顺序、`<op>.out[.<i>]` 命名、Input/Output 重建、常量 `data_ptr=0` |

## 附录 C：术语

| 术语 | 说明 |
| --- | --- |
| IFB | In-flight batching，迭代级组批，prefill/decode 同批 |
| 赤字调度 | 以 `num_computed_tokens` 为真源，每步只推进未计算 token |
| chunked prefill | 长 prompt 跨步切块执行 |
| IR 哈希 | 重建图 `DumpIr()` 文本的 SHA-256，用于加载期一致性校验 |
| block pool | Paged KV 的物理块池 |
| slot | 逻辑 token 在分页 KV 中的物理写入位置 |
| ResourceStack | 逆序释放栈，用于销毁与回滚 |
| 统一接口 | 每个模块对外暴露的唯一纯虚基类，Engine 通过它调用 |
