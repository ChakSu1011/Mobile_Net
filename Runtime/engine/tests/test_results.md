# Engine 模块测试结果

## 1. 测试范围

本轮验证 Engine 模块的实现代码：

- 生命周期状态机与非法状态调用；
- 默认模块缺失时的错误路径；
- 空包路径校验；
- 幂等销毁与销毁后调用；
- 初始统计与状态查询。

未覆盖（需要 Fake 模块/真实模块，后续补充）：

- `Load` 成功路径（Loader 返回 Graph/Backend/ModelConfig）；
- `ExecuteStep` 的 Schedule/Execute/Update 流程；
- `SetInput/SetOutput` 成功绑定；
- threaded 推理线程、`RunUntilIdle` 等待与回调；
- 与真实 Scheduler/KVCache/Executor/Destroyer 的集成。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-17 |
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
all engine lifecycle checks passed
```

链接退出码：`0`；运行退出码：`0`。

## 5. 用例与结论

| 用例 | 输入/动作 | 期望 | 实际 |
| --- | --- | --- | --- |
| 初始状态 | 构造 Engine | `CREATED` | 通过 |
| 初始统计 | `stats()` | 全 0 | 通过 |
| 空模块注入 | `SetModules(EngineModules{})` | `kOk` | 通过 |
| 空包加载 | `Load` 且 `package_dir` 为空 | `kInvalidArgument` | 通过 |
| 未加载 RunOnce | `RunOnce(nullptr)` | `kStateError` | 通过 |
| 未加载 AddRequest | 合法 Request | `kStateError` | 通过 |
| 未加载 Cancel | `CancelRequest(1)` | `kStateError` | 通过 |
| 未加载 Update | `Update(1, {1})` | `kStateError` | 通过 |
| 未加载 RunUntilIdle | `RunUntilIdle(1)` | `kStateError` | 通过 |
| 未加载 SetInput | `SetInput` | `kStateError` | 通过 |
| 未加载 SetOutput | `SetOutput` | `kStateError` | 通过 |
| 销毁 | `Destroy()` | `kOk`，状态 `DESTROYED` | 通过 |
| 重复销毁 | 第二次 `Destroy()` | `kOk` | 通过 |
| 销毁后加载 | `Load` | `kStateError` | 通过 |
| 默认模块缺失 | 非空包，工厂桩返回 nullptr | `kUnsupportedOp` | 通过 |
| 失败加载回退 | 上述失败后查询状态 | `CREATED` | 通过 |
| 失败后销毁 | `Destroy()` | `kOk` | 通过 |

## 6. 结论与风险

- Engine 生命周期、非法状态、空包校验、默认模块缺失与幂等销毁路径通过。
- 测试中的默认工厂为返回 `nullptr` 的桩，真实模块实现落地后需替换为 Fake 模块，
  补充 Load 成功、Step 执行、I/O 绑定与 threaded 模式测试。
- 当前测试未覆盖任何数值/设备行为；这些属于 Loader/Scheduler/KVCache/
  Executor/Backend 的测试范围。
