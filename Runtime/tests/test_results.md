# Runtime 接口阶段测试结果

## 1. 测试范围

本文件只记录 **I0 接口冻结阶段** 的验证结果：Runtime 全部公共头文件的
声明是否自洽、可被 C++17 编译器编译。此阶段不实现任何 `.cc` 内部逻辑，
因此不包含功能、数值、性能与稳定性测试；这些测试在 R0–R5 阶段补充。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-16 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | zig clang wrapper（`Graph/tools/zig_cxx.cmd`，Clang 兼容） |
| 语言标准 | C++17 |
| 包含路径 | 仓库根目录 `d:/Mobile_Net` |
| 测试文件 | `Runtime/tests/interface_check.cc` |

## 3. 测试命令

```bash
cd d:/Mobile_Net
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/tests/interface_check.cc \
  -o TempSpace/runtime_interface_check_20261116_070500/interface_check.o
```

## 4. 实际输出

```text
（无输出）
```

退出码：`0`。

复核：按选项 A（Engine → Executor → Backend 所有权）同步接口后再次执行同一
命令，输出仍为空，退出码 `0`。

## 5. 其他接口检查

### 5.1 TC-ARCH-02：文档接口与头文件一致性

检查方式：从 `Runtime/.doc/Runtime_design.md` 提取全部 `Runtime/**.h`
引用，逐一检查文件是否存在。

实际输出：

```text
header refs: 23
OK    Runtime/backend/backend.h
OK    Runtime/backend/kernel.h
OK    Runtime/backend/kernel_context.h
OK    Runtime/backend/kernel_types.h
OK    Runtime/common/data_types.h
OK    Runtime/common/io_binder.h
OK    Runtime/common/kv_allocator.h
OK    Runtime/common/kv_view.h
OK    Runtime/common/model_config.h
OK    Runtime/common/op_env_factory.h
OK    Runtime/common/plan.h
OK    Runtime/common/resource_stack.h
OK    Runtime/common/runtime_config.h
OK    Runtime/common/sequence.h
OK    Runtime/common/status.h
OK    Runtime/destroy/destroyer.h
OK    Runtime/engine/engine.h
OK    Runtime/engine/engine_modules.h
OK    Runtime/executor/executor.h
OK    Runtime/kvcache/kv_cache_manager.h
OK    Runtime/loader/loader.h
OK    Runtime/scheduler/scheduler.h
OK    Utils/cpp/types.h
missing: []
```

结论：文档引用的 23 个头文件全部存在，无缺失。

### 5.2 TC-ARCH-03：依赖方向检查

检查方式：扫描 `Runtime/common/*.h` 是否包含 Graph/Backend/Kernel 头文件。

实际输出：

```text
(common headers do not include Graph/Backend/Kernel)
```

结论：公共数据层不反向依赖执行层；`io_binder.h`/`op_env_factory.h` 只使用
前置声明，符合“Common ← 所有模块”的依赖方向。

### 5.3 TC-ARCH-04：Engine 成员检查

检查方式：扫描 `Runtime/engine/engine.h` 的 `std::unique_ptr` 成员。

实际输出（节选）：

```text
296:  std::unique_ptr<ILoader> loader_;
297:  std::unique_ptr<runtime_graph::Graph> graph_;
298:  std::unique_ptr<IScheduler> scheduler_;
299:  std::unique_ptr<IExecutor> executor_;
300:  std::unique_ptr<IKVCache> kv_cache_;
301:  std::unique_ptr<IDestroyer> destroyer_;
304:  std::unique_ptr<ResourceStack> resource_stack_;
```

结论：

- Loader/Graph/Scheduler/Executor/KVCache/Destroyer 为 Engine 的指针成员，
  可通过 `EngineModules` 替换 Scheduler/Executor/Destroyer；
- **Backend 不是 Engine 成员**，由 Loader 创建后移交 Executor 持有；
- **OpEnvFactory 不是 Engine 成员**，由 Loader 内部创建；
- `ResourceStack` 为 Engine 内部辅助成员。

### 5.4 目录迁移后的实现文件编译

检查方式：编译与本次 `Runtime/kernel/` → `Runtime/backend/` 迁移相关的 `.cc`
文件。

命令：

```bash
for f in Runtime/common/status.cc Runtime/common/data_types.cc \
         Runtime/backend/kernel_types.cc; do
  python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
    -c "$f" -o "TempSpace/runtime_interface_check_20261116_070500/$(basename "$f" .cc).o"
done
```

实际输出：

```text
OK Runtime/common/status.cc
OK Runtime/common/data_types.cc
OK Runtime/backend/kernel_types.cc
```

结论：迁移后的 Kernel 枚举实现与公共状态/类型实现均可编译，无警告。

### 5.5 TC-ARCH-05：EngineModules 注入对称性

检查方式：扫描 `Runtime/engine/engine_modules.h` 的可注入模块字段。

实际输出：

```text
45:  std::unique_ptr<ILoader> loader;
48:  std::unique_ptr<IScheduler> scheduler;
51:  std::unique_ptr<IExecutor> executor;
54:  std::unique_ptr<IDestroyer> destroyer;
```

结论：Loader/Destroyer 成对出现，对应资源 Init/UnInit；Scheduler/Executor
也可替换；`EngineModules` 单独放在 `engine_modules.h`，避免 `engine.h` 引入
Loader/Backend 依赖。

### 5.6 TC-ARCH-06：请求/响应对象唯一性

检查方式：确认 `sequence.h` 存在、`request.h` 已删除，且无旧 include 残留。

实际输出：

```text
Runtime/common/sequence.h
request.h absent
no request.h includes
37:enum class RequestStatus
58:struct SampleParams
146:struct Request
184:struct Token
208:struct Sequence
250:struct Response
```

结论：Request/Response/RequestStatus/SampleParams/Token/Sequence 只在
`sequence.h` 定义；`request.h` 已删除，代码不再引用旧头文件。

### 5.7 TC-ARCH-07：C++ 类型别名约定

检查方式：

1. 扫描 Graph/Runtime/Tests/Tools 下全部 C++ 文件；
2. 检查是否包含 `Utils/cpp/types.h`；
3. 检查是否仍有原始 `uint64_t/int32_t/size_t/float/double/bool/char/int` 等类型名。

实际输出：

```text
files missing types.h include: (none)
include count: 98
remaining raw token counts: {}
```

结论：98 个 C++ 文件全部包含 `Utils/cpp/types.h`，无原始类型名残留。
类型别名集合包含 `FP16/BF16/FP32/FP64`；其中 `FP16/BF16` 为 16 位存储位型，
转换由 Kernel/`half_utils` 完成。

回归编译：

```text
cmake --build Graph/build          -> 16/16 成功
Graph/build/graph_test.exe         -> graph_test: ALL TESTS PASSED
python Graph/tests/test_graph.py   -> 11 tests OK
Engine test                         -> all engine checks passed
Runtime 全部 .cc 单文件编译          -> failed=0
```

### 5.8 模块实现测试汇总

| 模块 | 命令/结果 | 结论 |
| --- | --- | --- |
| Scheduler | `scheduler_test.exe`：`checks=83 failures=0` | 通过 |
| KVCache | `kvcache_test.exe`：`checks=63 failures=0` | 通过 |
| Destroy | `destroyer_test.exe`：`checks=26 failures=0` | 通过 |
| Executor | `executor_test.exe`：54 PASS / 0 FAIL | 通过 |
| Loader | `loader_smoke.exe` T1 全量加载：PASSED，9.328 s | 通过 |
| Engine | `engine_test.exe`：`all engine checks passed` | 通过 |
| Runtime CMake | `TempSpace/runtime_cmake_20261120` 构建 66/66 成功 | 通过 |

细节见各模块 `tests/test_results.md`。

### 5.9 Mock 端到端 Qwen3-0.6B

测试程序：`Runtime/tests/mock_e2e_test.cc`；模型包：
`Models/Qwen3/artifacts/qwen3_0_6b`；模式：
`recompute_attention=true`（Mock attention 不支持 KV），单步完整序列前向，
比较 `sample.out` 的 argmax token 与 PyTorch 参考。

命令：

```bash
./Runtime/build/mock_e2e_test.exe \
  Models/Qwen3/artifacts/qwen3_0_6b 0
./Runtime/build/mock_e2e_test.exe \
  Models/Qwen3/artifacts/qwen3_0_6b 1
./Runtime/build/mock_e2e_test.exe \
  Models/Qwen3/artifacts/qwen3_0_6b 2
```

实际输出：

```text
Hello                    expected=21806 actual=21806
all mock e2e checks passed
real 7m58.325s

The capital of France is expected=12095 actual=12095
all mock e2e checks passed
CASE1_EXIT=0

1+1=                     expected=17 actual=17
all mock e2e checks passed
CASE2_EXIT=0
```

参考值由 [generate_qwen3_golden.py](../../Tests/golden/generate_qwen3_golden.py)
使用 Transformers `Qwen3ForCausalLM`（float32 CPU）生成，见
[Tests/golden/qwen3_0_6b.json](../../Tests/golden/qwen3_0_6b.json)。

## 6. 结论

- Runtime 全部模块已实现并通过链接：Loader/Scheduler/KVCache/Executor/
  Engine/Destroy/Common；
- 全部公共头文件与实现 `-Wall -Wextra` 编译无警告；
- Mock 后端加载 Qwen3-0.6B 全量包成功，IR/C++ DumpIr 与 manifest 哈希一致；
- 三条 prompt 的单步前向 argmax token 与 PyTorch 参考一致；
- 未覆盖项：真实 OCL/Adreno 设备、logits 数值误差、chunked prefill 端到端、
  KV 共享/COW、抢占、性能与 1 小时稳定性。这些在后续阶段补充。

## 7. 复现说明

`interface_check.cc` 只包含头文件并返回 0，不依赖任何实现符号；删除
`TempSpace/` 下的对象文件后可用上述命令重新生成。
