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

## 5. 其他接口检查

### 5.1 TC-ARCH-02：文档接口与头文件一致性

检查方式：从 `Runtime/.doc/Runtime_design.md` 提取全部 `Runtime/**.h`
引用，逐一检查文件是否存在。

实际输出：

```text
header refs: 19
OK   Runtime/backend/backend.h
OK   Runtime/common/data_types.h
OK   Runtime/common/io_binder.h
OK   Runtime/common/kv_allocator.h
OK   Runtime/common/kv_view.h
OK   Runtime/common/model_config.h
OK   Runtime/common/op_env_factory.h
OK   Runtime/common/plan.h
OK   Runtime/common/request.h
OK   Runtime/common/resource_stack.h
OK   Runtime/common/runtime_config.h
OK   Runtime/common/status.h
OK   Runtime/destroy/destroyer.h
OK   Runtime/engine/engine.h
OK   Runtime/executor/executor.h
OK   Runtime/kernel/kernel.h
OK   Runtime/kvcache/kv_cache_manager.h
OK   Runtime/loader/loader.h
OK   Runtime/scheduler/scheduler.h
missing: []
```

结论：文档引用的 19 个头文件全部存在，无缺失。

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
297:  std::unique_ptr<ILoader> loader_;
298:  std::unique_ptr<runtime_graph::Graph> graph_;
299:  std::unique_ptr<Backend> backend_;
300:  std::unique_ptr<IScheduler> scheduler_;
301:  std::unique_ptr<IExecutor> executor_;
302:  std::unique_ptr<IKVCache> kv_cache_;
303:  std::unique_ptr<IDestroyer> destroyer_;
```

结论：Loader/Graph/Backend/Scheduler/Executor/KVCache/Destroy 均为 Engine
的指针成员，满足“可动态绑定不同实现”的架构要求；`OpEnvFactory` 与
`ResourceStack` 为内部辅助成员。

## 6. 结论

- 全部 Runtime 公共头文件（common、loader、scheduler、kvcache、engine、
  destroy、executor、backend、kernel）编译通过，`-Wall -Wextra` 无警告。
- 接口声明与 `Runtime/.doc/Runtime_design.md` 的接口清单一致。
- 未覆盖项：链接、实现逻辑、运行时行为、设备后端、数值正确性。这些在
  后续阶段按模块补充测试结果。

## 7. 复现说明

`interface_check.cc` 只包含头文件并返回 0，不依赖任何实现符号；删除
`TempSpace/` 下的对象文件后可用上述命令重新生成。
