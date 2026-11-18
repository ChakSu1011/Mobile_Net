# Engine 模块开发 Skill

> 迁移提示：`Runtime_design.md` v0.4 已将请求/响应对象迁移到
> `Runtime/common/sequence.h`。本 skill 中涉及 Request/Response/Status/
> SampleParams/Token/Sequence 时，以 `sequence.h` 与新接口为准；代码迁移
> 完成后同步更新本文件中的 API 名称。

## 1. 适用场景

- 实现或修改 `Runtime/engine/` 的门面、生命周期、加载编排、单步执行、
  线程模型、I/O 绑定与销毁；
- 为 Engine 编写 Fake 模块单元测试；
- 评审 Engine 与 Loader/Scheduler/KVCache/Executor/Destroyer 的边界。

不适用：调度/KV/执行/后端算法实现、模型转换、Service 层。

## 2. 输入

1. `Runtime/.doc/Runtime_design.md`（跨模块契约，权威）；
2. `Runtime/engine/.doc/engine_design.md`（模块详细设计）；
3. 各模块接口头文件：`loader.h`、`scheduler.h`、`kv_cache_manager.h`、
   `executor.h`、`destroyer.h`、`backend.h`、`kernel.h`；
4. `Graph/.doc/Graph_design.md` 与 `Graph/include/*`；
5. memory.md（Google Style、中文注释、四件套）。

## 3. 步骤

1. **对齐所有权**：Backend 由 Executor 持有；Graph/GraphEnv/OpEnv 归
   Graph/Op；Engine 不直接引用 Backend。
2. **定义 EngineModules**：Loader/Scheduler/Executor/Destroyer 可注入，
   Loader 与 Destroyer 成对；`EngineModules` 放在独立头文件，避免
   `engine.h` 引入 Backend。
3. **实现状态机**：CREATED/LOADED/RUNNING/DESTROYING/DESTROYED；非法
   状态返回 `kStateError`。
4. **实现 Load**：
   - 校验 package_dir、ResourceStack 为空；
   - EnsureModules；
   - Loader 产出 Graph/Backend/ModelConfig；
   - 推导 SchedulerConfig、KvCacheSpec、KV 预算；
   - KVCache → Scheduler → Executor（Backend 移交）→ BindGraphIo；
   - 任一失败走统一回滚。
5. **实现 ExecuteStep**：
   - Schedule → BuildView → Execute → Update；
   - Scheduler 在终态释放 KV，Engine 不重复释放；
   - 回调异常不穿透；设备级错误置 fatal。
6. **实现线程模型**：默认 RunOnce；threaded 用唯一推理线程 + 条件变量；
   Destroy 停线程并 join。
7. **实现 I/O 绑定**：建立 input/output op 索引；校验 host Storage 后写
   `attr["storage"]`。
8. **实现 Destroy**：经 Destroyer 逆序释放；幂等；回滚复用同一路径。
9. **测试**：Fake Loader/Scheduler/Executor/KVCache/Destroyer/Backend
   覆盖生命周期、加载成功、I/O 绑定、单步/线程回调与销毁；记录实际输出。
10. **沉淀**：更新 `test_results.md` 与本 skill；真实模块落地后替换 Fake。

## 4. 输出

- `engine.h` / `engine_modules.h` / `engine.cc`；
- `Runtime/common/resource_stack.cc`（依赖）；
- `tests/engine_test.cc` 与 `tests/test_results.md`；
- 本 skill。

## 5. 常见陷阱

1. **Backend 所有权**：Load 成功时用 `std::move(result.backend)` 交给
   Executor；Init 失败必须由 Executor 释放，Engine 不再触碰。
2. **KV 释放双点**：只允许 Scheduler 经 `KvAllocator::Release` 释放；
   Engine 不要重复调用。
3. **`num_computed_tokens`**：只由 Scheduler 在 Schedule 推进；Engine/
   Update 不要重复推进。
4. **IR 哈希**：Loader 必须在绑定 `data_ptr` 之前计算；Engine 不参与。
5. **Scheduler 线程安全**：`Merge/Stop/HasRequests` 由调用方线程与推理
   线程并发调用，实现必须内部加锁。
6. **ResourceStack**：Load 前必须为空；失败回滚后替换为新的空栈。
7. **stats 语义**：EngineStats 是最近 Step 快照 + 累计，不实时查询模块。
8. **回调**：在推理线程同步执行；不得阻塞/重入；异常必须被捕获。
9. **线程模式**：threaded 模式下 RunOnce 返回 `kStateError`；Destroy 必须
   join。
10. **默认工厂缺失**：返回 `kUnsupportedOp`，不要静默创建空对象。
11. **I/O 校验**：host Storage 必须 kHost、非空指针、精度/布局/字节数/
    形状匹配；图侧字节数未设置视为错误。
12. **Load 失败回滚**：Executor 已接管 Backend 时通过
    `executor->backend()` 获取；未接管时使用局部 Backend。

## 6. 检查清单

- [ ] Engine 只依赖接口与 common 数据，不 include 后端专有头文件；
- [ ] Backend 不在 Engine 成员中；
- [ ] EngineModules 包含 Loader/Destroyer 且成对；
- [ ] Load/Start/Destroy 状态校验完整；
- [ ] Load 失败回滚后 state=CREATED、ResourceStack 为空；
- [ ] ExecuteStep 中 Scheduler 释放 KV、Engine 不重复；
- [ ] 回调异常被捕获；
- [ ] threaded 模式 Destroy 能 join；
- [ ] SetInput/SetOutput 校验完整；
- [ ] `-Wall -Wextra` 无警告；
- [ ] 测试结果记录实际命令与输出；
- [ ] 文档/skill 与代码同步。
