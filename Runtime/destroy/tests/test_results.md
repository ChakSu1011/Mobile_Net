# Destroy 模块测试结果

## 1. 测试范围

本文件记录 `Runtime/destroy/destroyer.cc`（默认 `IDestroyer` 实现）的主机
单元测试结果，覆盖：

- 释放顺序：Synchronize → ClearOpEnv → ResourceStack → GraphEnv →
  Executor/Backend；
- 真实 Graph（输入 + Add + 输出）的 OpEnv 清理与 GraphEnv 清理；
- Executor 已初始化时只调用 `Executor::Deinit`，不直接调用
  `Backend::Deinit`；
- 首个错误收集：同步失败/缓冲失败/Executor 失败仍继续后续阶段；
- 相同上下文重复 Destroy 幂等；
- 空上下文与“标志为真但指针为空”的安全处理。

不覆盖：真实设备销毁、推理线程停止、Scheduler 请求取消、指针 ABA 压力。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-20 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | `python -m ziglang c++`（zig 0.17.1，Clang 兼容） |
| 语言标准 | C++17 |
| 编译选项 | `-Wall -Wextra` |
| 包含路径 | `-I "d:/Mobile_Net"` |
| 测试源 | `Runtime/destroy/tests/destroyer_test.cc` |
| 被测源 | `Runtime/destroy/destroyer.cc`、`Runtime/common/resource_stack.cc` |
| 链接库 | `Graph/build/libgraph.a`（已含 `Runtime/common/status.cc` 符号） |
| Fake 模块 | Fake Backend、Fake Executor、计数型 OpEnv |

## 3. 测试命令

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

构建日志（stdout+stderr）：

```text
（无输出）
```

构建退出码：`0`。

## 4. 实际运行输出

```text
Runtime Destroy unit test
[CASE] destroy order and idempotency
[PASS] BuildGraph creates input/add/output with OpEnv
[PASS] SetGraphEnv installs a GraphEnv
[PASS] CreateDefaultDestroyer returns non-null
[PASS] first Destroy succeeds (got OK)
[PASS] Synchronize is called once
[PASS] Executor::Deinit is called once
[PASS] Backend::Deinit is not called when Executor owns it
[PASS] all three OpEnv objects are cleared
[PASS] ResourceStack releases its buffer entry
[PASS] GraphEnv is cleared
[PASS] Synchronize happens before ClearOpEnv
[PASS] OpEnv is cleared before buffered resources
[PASS] buffers are released before Executor::Deinit
[PASS] backend Deinit is not invoked directly
[PASS] second Destroy on the same context succeeds (got OK)
[PASS] second Destroy is a no-op for the same context
[PASS] Destroy with explicit null scheduler succeeds (got OK)
[CASE] backend-only destroy path
[PASS] backend-only Destroy succeeds (got OK)
[PASS] Synchronize and Backend::Deinit are both called
[CASE] error collection
[PASS] Destroy returns the first failure (Synchronize) (got DEVICE_ERROR)
[PASS] later release phases still run after an earlier failure
[PASS] ResourceStack is cleared despite failure
[PASS] Executor::Deinit failure is propagated (got BUFFER_ERROR)
[CASE] null context
[PASS] empty DestroyContext succeeds (got OK)
[PASS] empty DestroyContext is idempotent (got OK)
[PASS] initialized flags with null pointers are skipped safely (got OK)
checks=26 failures=0
```

运行退出码：`0`。

## 5. 用例与需求覆盖

| 用例 | 覆盖需求 | 结论 |
| --- | --- | --- |
| 正常顺序 + 幂等 | FR-DES-01~05/07 | 通过 |
| Executor 未初始化 → Backend::Deinit | FR-DES-01/05 | 通过 |
| 首个错误收集与阶段继续 | FR-DES-06 | 通过 |
| 空上下文/空指针 | FR-DES-08 | 通过 |

## 6. 结论

- 编译：退出码 0，`-Wall -Wextra` 无警告。
- 运行：退出码 0，`checks=26 failures=0`。
- 释放顺序、错误收集与幂等符合
  `Runtime/destroy/.doc/destroyer_design.md` 的 FR-DES-01~08。

## 7. 未覆盖项与风险

1. Fake Backend/Executor 不模拟真实设备同步、Kernel 释放与 Backend 所有权
   转移；真实 `Executor::Deinit` 路径需在 Executor 集成测试中验证。
2. 未测试 Scheduler 请求取消（`IScheduler` 没有 Shutdown/StopAll 接口）。
3. 未做多生命周期指针 ABA 压力测试；相同六指针复用会命中幂等缓存。
4. 若 `Executor::Deinit` 失败且未释放 Backend，Destroyer 不回退到
   `Backend::Deinit`，避免双重释放；该场景由调用方处理。
5. 测试通过 `Graph/build/libgraph.a` 链接；若 Graph 库未构建，需先按
   Graph 模块的构建说明生成静态库。
6. 未覆盖 `Executor::Init` 失败回滚的所有权缺口：Engine 在 Executor 已
   释放 Backend 后传入空 backend，而 ResourceStack 闭包仍捕获该
   Backend；Destroyer 按契约 Clear 时会触发 UAF。该问题需上游
   Engine/Executor 修复，Destroyer 无法安全区分“加载早期空 backend”与
   “Executor 已释放 backend”。
