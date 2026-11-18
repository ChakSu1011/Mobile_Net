# Engine 模块测试结果

## 1. 测试范围

本轮验证 Engine 模块的实现代码：

- 生命周期状态机、非法状态调用、空包校验、默认模块缺失；
- 加载成功路径（Fake Loader 产出最小图与 FakeBackend）；
- `SetInput/SetOutput` 成功绑定与参数校验；
- `AddRequest → RunOnce → Update → ResponseCallback` 单步执行；
- `RequestStatus/SampleParams/Token/Sequence/Response` 新对象模型；
- Engine/Step 统计；
- 单线程与 threaded 两种驱动方式；
- 幂等销毁与销毁后调用。

未覆盖（需要真实模块或更复杂的 Fake）：

- 真实模型包加载、真实 Graph 序列化与 IR 哈希；
- 真实 Scheduler/KVCache/Executor/Backend 的算法与数值正确性；
- 多请求并发、取消/超时、抢占；
- 真实设备 H2D/D2H 与 Kernel 执行；
- 性能与内存指标。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-19 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | zig clang wrapper（Clang 兼容） |
| 语言标准 | C++17 |
| 测试文件 | `Runtime/engine/tests/engine_test.cc` |
| 被测实现 | `Runtime/engine/engine.cc`、`Runtime/common/resource_stack.cc` |
| 依赖库 | `Graph/build/libgraph.a` |

## 3. 测试命令

```bash
cd d:/Mobile_Net
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  Runtime/engine/tests/engine_test.cc \
  Runtime/engine/engine.cc \
  Runtime/common/resource_stack.cc \
  Graph/build/libgraph.a \
  -o TempSpace/runtime_interface_check_20261116_070500/engine_test.exe

./TempSpace/runtime_interface_check_20261116_070500/engine_test.exe
```

## 4. 实际输出

```text
all engine checks passed
```

链接退出码：`0`；运行退出码：`0`。

## 5. 用例与结论

| 用例 | 输入/动作 | 期望 | 实际 |
| --- | --- | --- | --- |
| 初始状态 | 构造 Engine | `CREATED` | 通过 |
| 空模块注入 | `SetModules(EngineModules{})` | `kOk` | 通过 |
| 空包加载 | `Load` 且 `package_dir` 为空 | `kInvalidArgument` | 通过 |
| 未加载调用 | RunOnce/AddRequest/Cancel/Update/RunUntilIdle/SetInput/SetOutput | `kStateError` | 通过 |
| 销毁 | CREATED 状态 `Destroy()` | `kOk`，`DESTROYED` | 通过 |
| 重复销毁 | 第二次 `Destroy()` | `kOk` | 通过 |
| 默认模块缺失 | 非空包，默认工厂桩返回 nullptr | `kUnsupportedOp` | 通过 |
| 失败加载回退 | 上述失败后查询状态 | `CREATED` | 通过 |
| 加载成功 | Fake Loader + FakeBackend + FakeKVCache | `kOk`，`LOADED` | 通过 |
| I/O 绑定成功 | 匹配精度/布局/字节数/形状的 host storage | `kOk` | 通过 |
| 请求入队 | `AddRequest` | `kOk`，请求 ID 非 0 | 通过 |
| 单步执行 | `RunOnce` | `kOk`，生成 1 token | 通过 |
| Response 回调 | 单步结束回调 | status=kStopped，current_token.token_id=42，generated_count=1 | 通过 |
| Engine 统计 | 执行后 `stats()` | total_steps=1、total_tokens=1 | 通过 |
| 空闲等待 | `RunUntilIdle(100)` | `kOk` | 通过 |
| 执行后销毁 | `Destroy()` | `kOk`，`DESTROYED` | 通过 |
| threaded 启动 | `threaded=true` + `Start()` | `kOk` | 通过 |
| threaded 禁用 RunOnce | `RunOnce` | `kStateError` | 通过 |
| threaded 回调 | 推理线程生成 Response | status=kStopped，current_token.token_id=42 | 通过 |
| threaded 空闲等待 | `RunUntilIdle(1000)` | `kOk` | 通过 |
| threaded 销毁 | `Destroy()` | `kOk`，`DESTROYED` | 通过 |

## 6. 测试桩说明

- `FakeLoader`：构造 `input → add → sample → output` 的最小图，返回
  `FakeBackend` 与测试 `ModelConfig`；
- `FakeScheduler`：维护 `Sequence`，`Update` 追加 Token 并填充
  `Response`，随后结束请求；
- `FakeExecutor`：为每个计划项返回 token 42，持有并释放 Backend；
- `FakeKVCache`：固定视图/块表/slot，不做真实分配；
- `FakeDestroyer`：清空 ResourceStack 并反初始化 Executor/Backend；
- 默认工厂桩中只有 `CreateDefaultKVCache` 返回 FakeKVCache，其余返回
  nullptr，用于覆盖“默认模块不可用”的失败路径。

## 7. 结论与风险

- Engine 生命周期、加载成功、I/O 绑定、单步执行、回调、统计、threaded
  驱动与销毁路径均通过。
- 测试桩只验证 Engine 的编排与状态机，不验证模块内部算法；模块替换实现
  时必须保持 `IScheduler/KvAllocator/IExecutor/IDestroyer` 契约。
- 真实模块落地后，应将本文件的 Fake 替换为模块实现的集成测试，并补充
  多请求、取消/超时、抢占与真实数值对齐测试。
