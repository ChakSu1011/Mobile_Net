# Destroy 模块设计文档

| 项目 | 内容 |
| --- | --- |
| 模块路径 | `Runtime/destroy/` |
| 权威接口 | `Runtime/destroy/destroyer.h`（`IDestroyer`） |
| 上游设计 | `Runtime/.doc/Runtime_design.md` 4.2.5；`MobileNet_design.md` 4.3.6 |
| 实现文件 | `Runtime/destroy/destroyer.cc` |
| 测试 | `Runtime/destroy/tests/destroyer_test.cc`、`Runtime/destroy/tests/test_results.md` |
| 状态 | 阶段一/二默认实现 |

## 1. 背景与目标

### 1.1 背景

Runtime 的资源跨越多个模块：Backend 的设备上下文与缓冲、Graph 的 GraphEnv、
Op 的 OpEnv/Kernel、KVCache 的 KV 缓冲，以及 Loader 登记到 `ResourceStack`
的其它资源。正常销毁与加载失败回滚必须走同一逆序路径，否则会出现悬空指针、
重复释放或设备资源泄漏。

Engine 在 Destroy 前已经停止并 join 推理线程；随后构造 `DestroyContext`
交给 `IDestroyer`。Destroyer 只按固定阶段释放，不创建资源、不直接调用设备
API、不删除 Engine 自身对象。

### 1.2 目标

1. 按“同步 → OpEnv → ResourceStack → GraphEnv → Executor/Backend”的顺序
   释放资源，满足依赖逆序。
2. 单个阶段失败不中断后续释放，最终返回首个错误码。
3. 重复调用相同上下文幂等；不重复 `Synchronize`/`Deinit`。
4. `DestroyContext` 字段允许为空，覆盖加载失败的部分初始化场景。
5. 不直接调用设备 API，全部经 Backend/Executor/ResourceStack/Graph 接口。

### 1.3 非目标

- 不停止推理线程或拒绝新请求（由 Engine 完成）。
- 不遍历/取消 Scheduler 请求（`IScheduler` 没有 Shutdown 接口）。
- 不创建、不重置任何资源；不负责 Engine 对象析构。
- 不实现 Backend/Kernel 的设备细节。
- 不保证多线程调用安全；由调用方串行化。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| FR-DES-01 | `backend_initialized && backend != nullptr` 时先 `Backend::Synchronize(nullptr)` | TC-DES-01/02 |
| FR-DES-02 | 遍历 `GetOrderlyOPList`、`GetAllInputOp`、`GetAllOutputOp` 调用 `Op::ClearOpEnv()`，同一 Op 只清理一次 | TC-DES-01 |
| FR-DES-03 | `resource_stack != nullptr` 时调用 `Clear()` 逆序释放缓冲 | TC-DES-01/03 |
| FR-DES-04 | `graph != nullptr` 时调用 `ClearGraphEnv()`，且必须在 OpEnv 清理之后 | TC-DES-01 |
| FR-DES-05 | `executor_initialized && executor != nullptr` 时只调用 `Executor::Deinit()`（由 Executor 释放 Backend）；否则在 Backend 已初始化时调用 `Backend::Deinit()` | TC-DES-01/02 |
| FR-DES-06 | 收集首个错误并继续执行后续阶段，最终返回首个错误 | TC-DES-03 |
| FR-DES-07 | 相同 `DestroyContext` 重复调用幂等，返回上次结果且不重复释放 | TC-DES-01/04 |
| FR-DES-08 | 空指针字段跳过；`backend_initialized/executor_initialized` 为真但指针为空安全跳过 | TC-DES-04 |

### 2.2 非功能需求

- **NFR-DES-01 无设备 API**：不 include 厂商头，不直接调用 `cl*`/`vk*`/CUDA。
- **NFR-DES-02 非拥有**：Destroyer 不删除任何对象，只调用释放接口。
- **NFR-DES-03 有界复杂度**：O(op 数 + ResourceStack 条目数)。
- **NFR-DES-04 可观测**：错误消息带阶段前缀（`Destroy::...`），保留原错误码。
- **NFR-DES-05 安全**：重复调用不触碰已释放对象。

### 2.3 约束条件

- 调用方必须先停止推理线程；Destroyer 不负责线程同步。
- `context.backend` 为借用指针；Executor 已初始化时应为
  `executor->backend()`，否则为 Load 阶段的局部 Backend。
- `backend_initialized=false` 时不得调用 Synchronize/Deinit。
- `executor_initialized=true` 时 Backend 所有权在 Executor，不得直接 Deinit。
- `ResourceStack` 必须在 Backend 仍有效时清理（本实现顺序保证）。

### 2.4 验收标准

1. `destroyer_test` 全部断言通过，退出码 0（当前 26 条断言）。
2. 正常路径验证同步、3 个 OpEnv、缓冲、GraphEnv、Executor 的调用顺序。
3. Executor 已初始化时不直接调用 Backend::Deinit。
4. 首个错误（同步/缓冲/Executor）被返回，后续阶段仍执行。
5. 相同上下文第二次调用无任何副作用。

## 3. 概要设计

### 3.1 总体架构

```text
Engine::Destroy / Load rollback
  └── IDestroyer (DefaultDestroyer)
        ├── ContextKey：上下文身份快照（六指针 + 两标志）
        ├── last_context_/last_status_：幂等缓存
        └── Destroy(context)：
              1. Synchronize
              2. ClearOpEnv（拓扑 + 输入 + 输出，去重）
              3. ResourceStack::Clear
              4. Graph::ClearGraphEnv
              5. Executor::Deinit 或 Backend::Deinit
```

### 3.2 模块划分

| 组成 | 职责 |
| --- | --- |
| `ContextKey` | 上下文指针与初始化标志的身份快照 |
| `MakeContextKey` | 从 `DestroyContext` 生成快照 |
| `RecordFirstError` | 保留首个失败状态 |
| `DefaultDestroyer` | 固定阶段释放与幂等缓存 |

### 3.3 关键流程

```text
Destroy(context):
  key = MakeContextKey(context)
  if has_last_context_ and key == last_context_: return last_status_
  first = Ok
  if backend_initialized and backend: first = first_or(Synchronize)
  if graph:
      cleared = set()
      for op in orderly/input/output: if op not in cleared: ClearOpEnv()
  if resource_stack: first = first_or(Clear())
  if graph: ClearGraphEnv()
  if executor_initialized and executor: first = first_or(Executor::Deinit())
  else if backend_initialized and backend: first = first_or(Backend::Deinit())
  has_last_context_ = true; last_context_ = key; last_status_ = first
  return first
```

### 3.4 技术选型与取舍

- **固定阶段而非通用图遍历**：销毁顺序是跨模块契约，写在代码中比配置更
  可控，也便于审计。
- **OpEnv 去重**：输入/输出 op 可能同时出现在拓扑序列与边界列表，使用
  `unordered_set<Op*>` 保证每个 Op 只清理一次。
- **上下文级幂等**：用六指针 + 两标志作为 key；不依赖引用计数或全局状态。
  新生命周期通常产生不同指针，因而不会误跳过。
- **错误前缀**：`PrefixStatus` 只改消息、保留错误码，便于定位阶段。

## 4. 详细设计

### 4.1 数据结构

```cpp
struct ContextKey {
  const void* scheduler; const void* kv_cache; const void* executor;
  const void* graph; const void* backend; const void* resource_stack;
  Bool backend_initialized; Bool executor_initialized;
};
```

`DefaultDestroyer` 字段：

| 字段 | 含义 |
| --- | --- |
| `has_last_context_` | 是否记录过销毁 |
| `last_context_` | 最近一次上下文快照 |
| `last_status_` | 最近一次返回状态 |

### 4.2 释放阶段与依赖

| 顺序 | 阶段 | 依赖原因 |
| --- | --- | --- |
| 1 | `Synchronize` | 等待在途 Kernel，避免释放正在使用的缓冲 |
| 2 | `ClearOpEnv` | OpEnv/Kernel 可能引用 Backend 与 GraphEnv |
| 3 | `ResourceStack::Clear` | 释放 KV/权重/IO 等设备缓冲；此时 Backend 仍有效 |
| 4 | `ClearGraphEnv` | OpEnv 已全部释放，GraphEnv 无悬空引用 |
| 5 | `Executor::Deinit` / `Backend::Deinit` | 最后释放 Backend 内部资源 |

阶段 1 必须有 `backend_initialized`；阶段 5 若 Executor 已初始化，由
`Executor::Deinit` 统一 `Backend::Deinit` 并释放 Backend，Destroyer 不再
直接调用 Backend。

### 4.3 错误处理

- `RecordFirstError` 仅在 `first_error.ok()` 时覆盖，保证返回首个错误。
- 所有阶段都执行完才返回；失败不能导致后续资源泄漏。
- `PrefixStatus` 加阶段前缀：
  - `Destroy::Synchronize`
  - `Destroy::ResourceStack`
  - `Destroy::Executor::Deinit`
  - `Destroy::Backend::Deinit`
- OpEnv 清理与 GraphEnv 清理接口返回 `void`，无错误上报。

### 4.4 幂等语义

- 第一次调用后记录 `ContextKey` 与返回状态。
- 相同 key 的第二次调用直接返回 `last_status_`，不触碰任何指针。
- 不同 key（新 Backend/Graph/ResourceStack/Executor 指针或标志变化）视为
  新生命周期，重新执行全部阶段。
- 该设计依赖调用方传入完整、稳定的上下文；若调用方在两个生命周期间复用
  完全相同的六个地址，Destroyer 无法区分（指针 ABA），属于已知限制。

### 4.5 性能与内存

- 时间复杂度 O(op 数 + ResourceStack 条目数)。
- `unordered_set` 只在单次 Destroy 内使用，规模为图 op 数。
- 不持有资源、不复制图或缓冲；常量级额外内存。

### 4.6 限制与扩展

1. **Scheduler**：`context.scheduler` 只参与幂等 key；Destroyer 不取消请求。
   后续若 `IScheduler` 增加 Shutdown/StopAll，可在此阶段调用。
2. **部分失败**：若 Executor::Deinit 失败且未释放 Backend，Destroyer 不会
   回退到 Backend::Deinit（避免双重释放），由调用方决定后续处理。
3. **线程**：不阻塞推理线程；调用方必须先 join。
4. **资源栈清理失败**：继续后续阶段；失败的缓冲可能泄漏，由 Backend::Deinit
   兜底报告（取决于后端实现）。

## 5. 实施步骤

### 5.1 阶段划分

| 阶段 | 任务 | 产出 |
| --- | --- | --- |
| P1 | 阅读 4.3.6 释放顺序与模块所有权 | 设计 3.3/4.2 |
| P2 | 实现 ContextKey 与幂等缓存 | 重复调用安全 |
| P3 | 实现 Synchronize、OpEnv 清理、ResourceStack、GraphEnv | 主机资源路径 |
| P4 | 实现 Executor/Backend 反初始化分支与错误收集 | 完整销毁 |
| P5 | 编写真实 Graph + Fake Backend/Executor 单测 | `tests/` |

### 5.2 依赖与里程碑

- 依赖 `Graph/include/{graph,op}.h`、`Runtime/backend/backend.h`、
  `Runtime/executor/executor.h`、`Runtime/common/{resource_stack,status}.h`。
- 里程碑 1：正常顺序与 GraphEnv/OpEnv 清理通过。
- 里程碑 2：Executor 优先于 Backend 的分支通过。
- 里程碑 3：首个错误收集与幂等通过。

### 5.3 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 指针 ABA | 误跳过新生命周期 | 六指针 + 两标志 key；文档声明限制 |
| Executor Deinit 失败 | Backend 可能未释放 | 不回退，避免双重释放；上报错误 |
| 资源栈清理失败 | 缓冲泄漏 | 继续 Deinit；由后端兜底 |
| 重复清理 OpEnv | 未定义行为 | unordered_set 去重 |

## 6. 测试与验证设计

### 6.1 测试策略

- 真实 `runtime_graph::Graph`：输入 + Add + 输出，拓扑排序后为三个 Op 绑定
  计数型 OpEnv；设置 GraphEnv。
- Fake Backend/Executor：记录 Synchronize/Deinit 调用与事件顺序，可注入失败。
- 断言阶段顺序、调用次数、GraphEnv 清理、错误码与幂等。

### 6.2 用例设计

| 用例 | 覆盖需求 | 内容 |
| --- | --- | --- |
| TC-DES-01 | FR-DES-01~05/07 | 正常销毁：sync→openv→buffer→executor_deinit；3 个 OpEnv；GraphEnv 清空；第二次调用无副作用 |
| TC-DES-02 | FR-DES-01/05 | Executor 未初始化：Synchronize + Backend::Deinit 各一次 |
| TC-DES-03 | FR-DES-06 | Synchronize 失败仍清理缓冲/OpEnv/Executor，返回首个错误；Executor Deinit 失败传播 |
| TC-DES-04 | FR-DES-08 | 空上下文与“标志为真但指针为空”安全返回 |

### 6.3 测试环境与命令

```bash
cd d:/Mobile_Net
mkdir -p Runtime/build
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  Runtime/destroy/tests/destroyer_test.cc \
  Runtime/destroy/destroyer.cc \
  Runtime/common/resource_stack.cc \
  Graph/build/libgraph.a \
  -o Runtime/build/destroyer_test.exe
./Runtime/build/destroyer_test.exe
```

说明：`Graph/build/libgraph.a` 已包含 `Runtime/common/status.cc` 的符号，
因此不再单独链接 `status.cc`，避免重复定义。

通过标准：退出码 0，`failures=0`。实际输出见
`Runtime/destroy/tests/test_results.md`。

### 6.4 未覆盖项与残余风险

- 未测试真实 Backend 的 Synchronize/Deinit 设备行为。
- 未测试 Executor::Deinit 内部释放 Backend 的真实所有权路径（Fake 只记录）。
- 未测试 Scheduler 请求取消（接口未提供）。
- 未做多线程/多生命周期 ABA 压力测试。
- **Executor::Init 失败回滚的所有权缺口**：`Engine::Load` 在
  `executor_->Init` 失败时传入 `context.backend=nullptr、
  backend_initialized=false`（Executor 已按契约释放 Backend），但
  `ResourceStack` 中仍可能保留 KVCache/Loader 的释放闭包，这些闭包捕获了
  已释放的 `Backend*`。Destroyer 按接口要求调用 `ResourceStack::Clear`
  时会触发对已释放 Backend 的 `DestroyBuffer`（潜在 UAF）。修复需要
  Engine/Executor 调整所有权时序（例如在 Executor 接管前清空依赖 Backend
  的闭包，或让闭包持有可延长生命周期的 Backend 句柄）。该问题无法在
  Destroyer 内安全判定（空 backend 也可能对应“加载早期失败”），因此本
  实现保持契约顺序并在此声明风险。

## 附录 A：需求-设计-测试追溯

| 需求 | 设计章节 | 测试用例 |
| --- | --- | --- |
| FR-DES-01 | 4.2 | TC-DES-01/02 |
| FR-DES-02 | 3.3/4.3 | TC-DES-01 |
| FR-DES-03 | 4.2 | TC-DES-01/03 |
| FR-DES-04 | 4.2 | TC-DES-01 |
| FR-DES-05 | 4.2 | TC-DES-01/02 |
| FR-DES-06 | 4.3 | TC-DES-03 |
| FR-DES-07 | 4.4 | TC-DES-01/04 |
| FR-DES-08 | 4.4 | TC-DES-04 |
