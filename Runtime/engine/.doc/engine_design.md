# Engine 模块设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 文档状态 | 详细设计（已实现，待评审） |
| 编写日期 | 2026-11-17 |
| 模块 | `Runtime/engine/` |
| 权威架构文档 | [`Runtime/.doc/Runtime_design.md`](../../.doc/Runtime_design.md) |
| 上游需求 | `.doc/MobileNet_design.md` FR-1.13、FR-1.14、4.3.5、4.3.6 |
| 强制规则 | memory.md：Google Style、中文注释、四件套、目录规范 |

**文档定位**：本文件是 Engine 模块的详细设计文档，细化统一架构文档中的
Engine 章节；跨模块契约以 `Runtime_design.md` 为准，本文件不重复定义。

---

## 1. 背景与目标

### 1.1 背景

Runtime 的 Loader、Scheduler、KVCache、Executor、Backend、Kernel、Destroy
均以接口形式冻结；Engine 是唯一对外门面，需要把加载、单步推理、请求生命
周期与资源释放串起来，并在不实现模块算法的前提下提供可替换、可测试的编
排层。

### 1.2 目标

1. 实现 Engine 生命周期状态机：`CREATED → LOADED → RUNNING → DESTROYING →
   DESTROYED`。
2. 以 `unique_ptr<接口>` 持有模块，支持 `EngineModules` 注入不同实现。
3. 实现 Load 编排、失败逆序回滚、Backend 所有权移交 Executor。
4. 实现单步执行 `Schedule → BuildView → Execute → Update` 与回调、统计。
5. 实现单线程 `RunOnce` 与 threaded 推理线程两种驱动方式。
6. 实现 `SetInput/SetOutput` 的图 I/O 绑定与校验。
7. 实现幂等 Destroy，并复用加载失败回滚路径。

### 1.3 非目标

- 不实现调度、KV、执行、后端或 Kernel 算法；
- 不解析模型包、不重建 Graph、不上传权重（Loader 负责）；
- 不接触文本/Tokenizer；
- 不实现阶段三的投机解码、异步调度、HiCache、多类型 KV。

---

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收要点 | 追溯 |
| --- | --- | --- | --- |
| E-01 | 生命周期状态机与非法状态拒绝 | 状态迁移合法；错误码为 `kStateError` | FR-1.13 |
| E-02 | 模块注入与默认创建 | `SetModules` 仅 CREATED 可用；缺失模块调默认工厂 | FR-1.13 |
| E-03 | Load 编排 | Loader → KV → Scheduler → Executor → BindGraphIo → LOADED | FR-1.13 |
| E-04 | Load 失败回滚 | 逆序释放；state 回 CREATED | FR-1.14 |
| E-05 | Backend 所有权移交 | Loader 创建、Executor 持有；Engine 不持有 | FR-1.13 |
| E-06 | 单步执行 | Schedule → BuildView → Execute → Update；Scheduler 释放 KV | FR-1.12/1.13 |
| E-07 | 请求 API | AddRequest/CancelRequest/Update；线程安全边界明确 | FR-1.10 |
| E-08 | 流式回调 | 每 token 回调；finished 标记；异常不穿透 | FR-1.13 |
| E-09 | I/O 绑定 | 校验精度/布局/字节数/形状后写 `attr["storage"]` | FR-1.12 |
| E-10 | 线程模型 | 默认 RunOnce；threaded 推理线程；Destroy join | NFR-1/5 |
| E-11 | 统计 | StepStats + EngineStats；running/waiting/KV 为最近 Step 快照 | NFR-4 |
| E-12 | 幂等销毁 | 重复 Destroy 安全；释放顺序经 Destroyer | FR-1.14 |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| EN-01 | 单步编排开销可控；不在 Engine 内做 O(n) 图遍历或数据拷贝 |
| EN-02 | 模块调用全部经接口，无后端专有头文件 |
| EN-03 | 线程安全边界清晰：AddRequest/CancelRequest 任意线程；RunOnce 单调用者 |
| EN-04 | 错误码可定位；设备级错误进入 fatal 状态 |
| EN-05 | Google C++ Style；模块/类/函数中文注释 |

### 2.3 约束条件

- 单进程、单设备；默认单线程驱动；
- Backend 由 Executor 持有；Engine 不直接引用 Backend；
- Scheduler 的 Merge/Stop/HasRequests 需内部线程安全；
- Executor::Init 失败必须释放已接管的 Backend；
- 阶段一/二不做投机解码/异步调度/HiCache/多类型 KV。

### 2.4 验收标准

| 编号 | 通过条件 |
| --- | --- |
| EA-01 | 生命周期与错误路径测试通过 |
| EA-02 | Fake 模块下 Load 成功、I/O 绑定、单步执行与回调通过 |
| EA-03 | threaded 模式启动、回调、RunUntilIdle、销毁通过 |
| EA-04 | 无内存泄漏（由 Fake 模块计数与后续真实模块测试补充） |
| EA-05 | 代码通过 `-Wall -Wextra` 编译 |

---

## 3. 概要设计

### 3.1 模块组成

```text
Engine
 ├── ILoader       加载期编排（内部 OpEnvFactory）
 ├── Graph         Loader 产出；Engine 持有
 ├── IScheduler    请求队列/组批/停止/抢占
 ├── IExecutor     执行面；拥有 Backend
 ├── IKVCache      分页 KV；Engine 内部创建
 ├── IDestroyer    逆序释放
 └── ResourceStack 释放/回滚闭包
```

`EngineModules` 单独定义在 `engine_modules.h`，包含 Loader/Scheduler/
Executor/Destroyer 四个可注入接口；`engine.h` 只前置声明，不引入
Backend/Loader 依赖。

### 3.2 生命周期

```text
CREATED ──Load──▶ LOADED ──Start──▶ RUNNING
   ▲                 │                │
   └──Rollback───────┘                │
              DESTROYING ◀──Destroy───┘
                   │
                   ▼
              DESTROYED（终态）
```

### 3.3 Load 流程

```text
Load(config)
  → 校验 CREATED、package_dir 非空
  → EnsureModules（缺失模块调默认工厂）
  → 校验 ResourceStack 为空
  → loader_->Load(...)
  → graph_ = std::move(result.graph)
  → BuildSchedulerConfig / Backend::Capability / BuildKvCacheSpec / ComputeKvBudget
  → kv_cache_->Init(backend)
  → scheduler_->Init(kv_cache, model)
  → executor_->Init(std::move(backend), graph_, this)
  → BindGraphIo()
  → state = LOADED
```

失败路径统一调用 Destroyer 回滚，state 回 `CREATED`。

### 3.4 单步流程

```text
ExecuteStep
  → scheduler_->Schedule(&plan)
  → kv_cache_->BuildView(&view)
  → executor_->Execute(plan, view, &result)
  → scheduler_->Update(token)（终态释放 KV）
  → TokenCallback
  → 填充 StepStats/EngineStats
```

### 3.5 线程模型

- 默认：调用方 `RunOnce` 驱动；
- threaded：`Start` 启动唯一推理线程，循环 `ExecuteStep`；无请求时在
  `idle_cv_` 上等待 10 ms；`AddRequest` 唤醒；
- `Destroy` 设置 stop 标记、join 线程，再调用 Destroyer。

### 3.6 关键取舍

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| Backend 所有权 | Executor 持有 | 设备资源随执行器生命周期；Engine 不暴露 Backend |
| KV 释放 | Scheduler 经 KvAllocator | 请求终态唯一来源，避免 Engine/Scheduler 双释放 |
| EngineModules | 独立头文件 | 避免 engine.h 引入 Loader/Backend 依赖 |
| Loader/Destroyer | 成对可注入 | 资源 Init/UnInit 对称 |
| 默认模块缺失 | 返回 `kUnsupportedOp` | 明确失败，不静默降级 |
| stats | 最近 Step 快照 + 累计 | 避免跨线程查询模块内部状态 |

---

## 4. 详细设计

### 4.1 数据结构

Engine 私有成员：

```cpp
EngineState state_;
RuntimeConfig runtime_config_;
ModelConfig model_config_;
std::unique_ptr<ILoader> loader_;
std::unique_ptr<runtime_graph::Graph> graph_;
std::unique_ptr<IScheduler> scheduler_;
std::unique_ptr<IExecutor> executor_;
std::unique_ptr<IKVCache> kv_cache_;
std::unique_ptr<IDestroyer> destroyer_;
std::unique_ptr<ResourceStack> resource_stack_;
std::unordered_map<std::string, runtime_graph::Op*> input_ops_;
std::unordered_map<std::string, runtime_graph::Op*> output_ops_;
bool backend_initialized_;
bool executor_initialized_;
TokenCallback token_callback_;
EngineStats stats_;
std::thread inference_thread_;
mutable std::mutex mutex_;
std::condition_variable idle_cv_;
std::atomic<bool> stop_requested_;
std::atomic<bool> fatal_;
std::atomic<ErrorCode> fatal_code_;
```

### 4.2 配置推导

**SchedulerConfig**

- `max_num_tokens = runtime.max_num_tokens > 0 ? runtime : model`；
- 若 runtime 覆盖值大于 `model.max_num_tokens`，返回 `kInvalidArgument`；
- `max_concurrent_requests = runtime > 0 ? runtime : min(max_num_tokens, 8)`；
- `max_prefill_chunk = max_num_tokens`；
- `context_length = model.context_length`；
- `stop_ids = model.stop_token_ids`。

**KvCacheSpec**

- `block_size = runtime.kv_block_size > 0 ? runtime : capability.attention_block_size`；
- 两者均非正返回 `kInvalidArgument`；
- 类型 FullAttention；层数/KV 头数/头维度/精度来自 ModelConfig。

**KV 预算**

- `kv_cache_bytes > 0` 时直接使用；
- 否则 `memory_budget_bytes - weights_bytes - arena_bytes - io_bytes -
  64 MiB`；
- 预算不足返回 `kDeviceOom`。

### 4.3 错误处理与回滚

- 设备级错误（Device/Kernel/Buffer/BackendInit）置 fatal；后续调用直接返回；
- 单请求执行失败调 `Scheduler::Fail`，不阻塞其他请求；
- Load 失败回滚：
  - Executor 未接管 Backend：DestroyContext.backend 使用局部 Backend；
  - Executor 已接管：`backend = executor->backend()`，由 `Executor::Deinit`
    释放；
  - 回滚后替换 ResourceStack，state 回 CREATED；
  - 回滚失败时在原始错误消息后追加原因。

### 4.4 I/O 绑定

- Load 后建立 Input/Output op 索引；
- `SetInput/SetOutput` 校验：
  - `device_type == kHost`；
  - `data_ptr != nullptr`；
  - 精度/布局与图侧一致；
  - 字节数一致（图侧必须已设置）；
  - 若双方 shape 都有维度，则逐维一致；
- 校验通过后写 `attr["storage"]`。

### 4.5 性能与内存

- 单步只做一次 plan 遍历与一次结果遍历；
- 不复制 token/KV 数据，只传递向量与视图；
- `StepStats` 为值对象，不持有资源；
- 回调在推理线程同步执行，避免额外队列。

### 4.6 实现文件

| 文件 | 内容 |
| --- | --- |
| `Runtime/engine/engine.h` | Engine 类、状态、配置、统计、私有方法 |
| `Runtime/engine/engine_modules.h` | EngineModules |
| `Runtime/engine/engine.cc` | 全部方法实现 |
| `Runtime/common/resource_stack.cc` | 释放栈实现 |

---

## 5. 实施步骤

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| E0 | 接口冻结（engine.h/engine_modules.h） | 完成 |
| E1 | ResourceStack 实现 | 完成 |
| E2 | Engine 生命周期与 Load/回滚 | 完成 |
| E3 | ExecuteStep/回调/统计 | 完成 |
| E4 | threaded 推理线程与 RunUntilIdle | 完成 |
| E5 | I/O 绑定与销毁 | 完成 |
| E6 | Fake 模块测试 | 完成 |
| E7 | 真实模块集成测试 | 待 R1–R4 |

风险与应对：

| 风险 | 应对 |
| --- | --- |
| 默认模块未实现 | Fake 模块测试；真实模块落地后替换 |
| Scheduler 线程安全缺失 | 接口已注明 Merge/Stop/HasRequests 内部加锁 |
| Backend 所有权误用 | Executor 持有；Init 失败必须释放 |
| KV 预算配置错误 | 启动期返回明确错误码 |

---

## 6. 测试与验证设计

### 6.1 测试策略

- 主机侧单元测试，不依赖真实设备；
- Fake Loader/Scheduler/Executor/KVCache/Destroyer/Backend；
- 覆盖正常路径、边界与错误路径；
- 记录实际命令与输出。

### 6.2 用例设计

| 用例 | 场景 | 通过标准 |
| --- | --- | --- |
| TC-ENG-01 | 生命周期/非法状态/空包/默认模块缺失/幂等销毁 | 错误码正确；state 正确 |
| TC-ENG-02 | Fake Loader 加载成功 | state LOADED；Backend 移交 Executor |
| TC-ENG-03 | SetInput/SetOutput 成功与不匹配 | 成功绑定；不匹配返回 `kInvalidArgument` |
| TC-ENG-04 | AddRequest → RunOnce → 回调 | token/finished 正确；统计更新 |
| TC-ENG-05 | threaded Start/回调/RunUntilIdle/Destroy | 线程正常退出；回调一次 |
| TC-ENG-06 | Destroy 幂等 | 重复调用安全 |

### 6.3 测试环境

Windows 11 + zig clang wrapper + C++17 + Graph 静态库；详见
`Runtime/engine/tests/test_results.md`。

### 6.4 通过标准与结果

- 编译 `-Wall -Wextra` 无警告；
- 所有用例通过；
- 实际输出：`all engine checks passed`（退出码 0）。

### 6.5 未覆盖项与风险

- 真实 Loader/Graph 序列化/IR 哈希；
- 真实 Scheduler/KVCache/Executor/Backend 的算法与数值；
- 多请求并发、取消/超时、抢占；
- 真实设备传输与 Kernel；
- 性能/内存指标。
