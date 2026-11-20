# Skill: Runtime Destroy 默认实现与回归

## 适用场景

- 修改 `Runtime/destroy/destroyer.cc` 的释放顺序、幂等或错误收集；
- 新增需要销毁的模块（GraphEnv、KVCache、OpEnv、Backend）；
- 排查“重复释放”“Synchronize 已释放 Backend”“ResourceStack 清理顺序”；
- 需要验证 Load 失败回滚与正常 Destroy 共用路径。

不适用：停止推理线程（Engine 负责）、Scheduler 请求取消、KVCache 块回收
策略（见各自 skill）。

## 输入

1. `Runtime/.doc/Runtime_design.md` 4.2.5 与 `MobileNet_design.md` 4.3.6；
2. `Runtime/destroy/.doc/destroy_design.md`（本模块权威设计）；
3. `Runtime/destroy/destroyer.h`（DestroyContext/IDestroyer）；
4. `Graph/include/{graph,op}.h`、`Runtime/backend/backend.h`、
   `Runtime/executor/executor.h`、`Runtime/common/{resource_stack,status}.h`；
5. 编译器：`python -m ziglang c++`。

## 步骤

### 1. 固定释放顺序

```text
1) backend_initialized && backend != null → Backend::Synchronize(nullptr)
2) graph != null → 对 GetOrderlyOPList/GetAllInputOp/GetAllOutputOp
   调用 Op::ClearOpEnv()（Op 去重）
3) resource_stack != null → ResourceStack::Clear()
4) graph != null → Graph::ClearGraphEnv()
5) executor_initialized && executor != null → Executor::Deinit()
   否则 backend_initialized && backend != null → Backend::Deinit()
```

顺序不可交换：OpEnv 可能引用 GraphEnv；缓冲释放需要 Backend 仍有效；
GraphEnv 必须在 OpEnv 之后；Executor::Deinit 最后释放 Backend。

### 2. 错误收集

- 每个返回 Status 的阶段用 `RecordFirstError` 保留首个错误；
- 失败不中断后续阶段；
- 用 `PrefixStatus` 加阶段前缀，保留原错误码。

### 3. 幂等

- 用六指针 + 两初始化标志生成 `ContextKey`；
- 相同 key 第二次调用直接返回 `last_status_`，不触碰指针；
- 新生命周期（不同指针/标志）重新执行全部阶段。

### 4. 编译与运行单测

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

`Graph/build/libgraph.a` 已包含 `Runtime/common/status.cc` 的符号，不要再
单独链接 `status.cc`，否则重复定义。

### 5. 更新文档与结果

- 顺序/幂等语义变化先改 `Runtime/destroy/.doc/destroy_design.md`；
- 实际命令/输出写入 `Runtime/destroy/tests/test_results.md`；
- 新增阶段需在追溯表中补需求与用例。

## 输出

- 可编译的 `destroyer.cc`；
- `destroyer_test` 可执行文件与全部 PASS 输出；
- 释放顺序、错误收集与幂等的回归结论。

## 常见陷阱

1. **Executor 与 Backend 双重释放**：`executor_initialized=true` 时只能调用
   `Executor::Deinit`，不能再调用 `Backend::Deinit`。
2. **Synchronize 悬空指针**：相同上下文重复调用必须直接返回；不要再次
   Synchronize 已 Deinit 的 Backend。
3. **OpEnv 重复清理**：输入/输出 op 可能同时在拓扑列表与边界列表；用
   `unordered_set<Op*>` 去重。
4. **GraphEnv 顺序**：必须在所有 OpEnv 清理后调用 `ClearGraphEnv`，否则
   OpEnv 持有悬空 GraphEnv 指针。
5. **ResourceStack 顺序**：必须在 Backend::Deinit 之前 Clear，否则缓冲
   释放闭包访问已反初始化的 Backend。
6. **状态前缀丢失**：`PrefixStatus` 保持原错误码；不要改成统一 `kInternal`。
7. **FakeExecutor 签名**：`IExecutor::Init` 的 `IoBinder` 位于
   `mobilenet::runtime`，测试中需 include `Runtime/common/io_binder.h`。
8. **Graph 链接**：测试可用 `Graph/build/libgraph.a`；如自行编译
   `Graph/src/*.cc`，只能编译一次 `Runtime/common/status.cc`。
9. **ContextKey ABA**：若调用方复用完全相同的六指针，幂等缓存无法区分
   新生命周期；这是文档声明的已知限制。
10. **Executor Init 失败回滚**：Engine 传入空 backend/未初始化标志时，
    ResourceStack 仍可能持有捕获已释放 Backend 的闭包；Clear 会触发
    `DestroyBuffer` UAF。这是 Engine/Executor 所有权时序问题，不要在
    Destroyer 内通过“跳过 Clear”掩盖（会泄漏其它资源）；集成前需修复
    上游契约。
