# Backend P0 测试结果

## 1. 测试范围

本文件记录 Backend 模块 **P0 契约与骨架阶段** 的实际验证结果：

- 新增并编译 `Runtime/backend/kernel_context.h`、`backend_registry.h/.cc`、
  `exec_op_env.h/.cc`；
- 清理 `Runtime/backend/kernel_types.h/.cc`（删除历史遗留的
  `CopyKind/BlockStage/RopeLayout`，改为 `Precision`↔`DataType` 映射）；
- `Runtime/backend/CMakeLists.txt` 独立构建 `runtime_backend` 静态库；
- 可执行检查 `backend_header_check`：公共头编译、Precision↔DataType、
  BackendRegistry、SetStepContext、KernelContext 默认值；
- 回归 Runtime 接口编译检查（`Runtime/tests/interface_check.cc`）。

本阶段**不包含** mock/OCL 后端、8 个算子 Kernel、Loader 集成、算子精度测试；
这些在 P1–P4 按设计文档继续。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-17 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | Clang 22.1.8（`Graph/tools/zig_cxx.cmd`，zig 0.17.1） |
| 归档器 | `Graph/tools/zig_ar.cmd`、`Graph/tools/zig_ranlib.cmd` |
| CMake | 4.4.4 |
| Ninja | 1.13.2 |
| 语言标准 | C++17 |
| 包含路径 | 仓库根目录 `d:/Mobile_Net` |
| 构建目录 | `Runtime/backend/build` |
| 工作目录 | `d:/Mobile_Net` |

## 3. 测试命令

```bash
cd d:/Mobile_Net

# 1. 配置
cmake -S Runtime/backend -B Runtime/backend/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="d:/Mobile_Net/Graph/tools/zig_cxx.cmd" \
  -DCMAKE_AR="d:/Mobile_Net/Graph/tools/zig_ar.cmd" \
  -DCMAKE_RANLIB="d:/Mobile_Net/Graph/tools/zig_ranlib.cmd" \
  -DBACKEND_AR="d:/Mobile_Net/Graph/tools/zig_ar.cmd" \
  -DBACKEND_RANLIB="d:/Mobile_Net/Graph/tools/zig_ranlib.cmd" \
  -DGRAPH_AR="d:/Mobile_Net/Graph/tools/zig_ar.cmd" \
  -DGRAPH_RANLIB="d:/Mobile_Net/Graph/tools/zig_ranlib.cmd"

# 2. 构建（--clean-first 保证记录的是完整构建输出）
cmake --build Runtime/backend/build --clean-first

# 3. 运行 CTest
ctest --test-dir Runtime/backend/build --output-on-failure

# 4. 直接运行检查程序
./Runtime/backend/build/backend_header_check.exe

# 5. 回归 Runtime 接口编译检查
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/tests/interface_check.cc \
  -o TempSpace/backend_p0_20261117_070907/interface_check.o
```

## 4. 实际输出

### 4.1 配置

```text
-- The CXX compiler identification is Clang 22.1.8
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: d:/Mobile_Net/Graph/tools/zig_cxx.cmd - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done (9.9s)
-- Generating done (0.1s)
-- Build files have been written to: D:/Mobile_Net/Runtime/backend/build
```

### 4.2 构建（`--clean-first` 重新完整构建）

```text
[1/1] Cleaning all built files...
Cleaning... 18 files.
[1/18] Building CXX object graph/CMakeFiles/graph.dir/src/op_type.cc.obj
[2/18] Building CXX object CMakeFiles/runtime_backend.dir/kernel_types.cc.obj
[3/18] Building CXX object graph/CMakeFiles/graph.dir/src/types.cc.obj
[4/18] Building CXX object graph/CMakeFiles/graph.dir/src/shape.cc.obj
[5/18] Building CXX object graph/CMakeFiles/graph.dir/src/op_env.cc.obj
[6/18] Building CXX object graph/CMakeFiles/graph.dir/D_/Mobile_Net/Runtime/common/status.cc.obj
[7/18] Building CXX object graph/CMakeFiles/graph.dir/src/graph_env.cc.obj
[8/18] Building CXX object graph/CMakeFiles/graph.dir/src/tensor.cc.obj
[9/18] Building CXX object graph/CMakeFiles/graph.dir/src/attr.cc.obj
[10/18] Building CXX object graph/CMakeFiles/graph.dir/src/storage.cc.obj
[11/18] Building CXX object CMakeFiles/runtime_backend.dir/exec_op_env.cc.obj
[12/18] Building CXX object CMakeFiles/runtime_backend.dir/backend_registry.cc.obj
[13/18] Building CXX object graph/CMakeFiles/graph.dir/src/op.cc.obj
[14/18] Building CXX object CMakeFiles/backend_header_check.dir/tests/header_check.cc.obj
[15/18] Building CXX object graph/CMakeFiles/graph.dir/src/graph.cc.obj
[16/18] Linking CXX static library graph\libgraph.a
[17/18] Linking CXX static library libruntime_backend.a
[18/18] Linking CXX executable backend_header_check.exe
```

构建过程无警告、无错误（`-Wall -Wextra`）。

### 4.3 CTest

```text
Test project D:/Mobile_Net/Runtime/backend/build
    Start 1: backend_header_check
1/1 Test #1: backend_header_check .............   Passed    0.08 sec

100% tests passed out of 1

Total Test time (real) =   0.21 sec
```

### 4.4 `backend_header_check` 实际输出（46 条 PASS）

```text
PASS precision float16 -> DataType float16
PASS DataType float16 -> precision float16
PASS precision float8e4m3 has no DataType
PASS precision int16 has no DataType
PASS DataType unknown is unsupported
PASS DataType float16 is supported
PASS supported precision maps to DataType
PASS supported precision round-trips
PASS supported precision maps to DataType
PASS supported precision round-trips
PASS supported precision maps to DataType
PASS supported precision round-trips
PASS supported precision maps to DataType
PASS supported precision round-trips
PASS supported precision maps to DataType
PASS supported precision round-trips
PASS supported precision maps to DataType
PASS supported precision round-trips
PASS supported precision maps to DataType
PASS supported precision round-trips
PASS supported precision maps to DataType
PASS supported precision round-trips
PASS supported precision maps to DataType
PASS supported precision round-trips
PASS register fake backend
PASS duplicate backend is rejected
PASS empty backend name is rejected
PASS empty backend factory is rejected
PASS registry names contain fake
PASS unknown backend returns kUnsupportedBackend
PASS null backend output is rejected
PASS create fake backend
PASS fake backend capability
PASS capability attention block size
PASS capability supported ops non-empty
PASS backend registered by macro is creatable
PASS set step context on ExecOpEnv
PASS clear step context on ExecOpEnv
PASS non-ExecOpEnv is rejected with kInternal
PASS null OpEnv is rejected with kInvalidArgument
PASS KernelContext inputs default empty
PASS KernelContext outputs default empty
PASS KernelContext optional pointers default null
PASS KernelContext step fields default
PASS KernelContext workspace default empty
PASS TensorView layout defaults row major
ALL TESTS PASSED
```

退出码：`0`。

### 4.5 Runtime 接口回归

```text
（无输出）
```

退出码：`0`；生成 `TempSpace/backend_p0_20261117_070907/interface_check.o`。

### 4.6 公共头自包含检查

对每个公共头单独生成一个只包含该头的 TU 并编译：

```bash
cd d:/Mobile_Net
OUT=TempSpace/backend_p0_selfcheck_20261117_071133
for h in backend.h kernel.h kernel_context.h exec_op_env.h \
         backend_registry.h kernel_types.h; do
  printf '#include "Runtime/backend/%s"\nint main() { return 0; }\n' \
    "$h" > "$OUT/tu_$h.cc"
  python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
    -c "$OUT/tu_$h.cc" -o "$OUT/tu_$h.o"
done
```

实际输出：

```text
OK   backend.h
OK   kernel.h
OK   kernel_context.h
OK   exec_op_env.h
OK   backend_registry.h
OK   kernel_types.h
```

结论：6 个公共头均可独立包含并通过 `-Wall -Wextra` 编译。

## 5. 结论

| 检查项 | 结论 |
| --- | --- |
| 公共头可编译、自洽 | 通过（`-Wall -Wextra` 无警告） |
| `Precision`↔`DataType` 映射 | 通过（9 个可映射精度往返一致；Float8/Int16 明确不支持） |
| BackendRegistry 注册/重复/非法/未知/工厂执行 | 通过 |
| 自注册宏静态注册 | 通过（`fake_macro` 可创建） |
| SetStepContext 类型校验与清空 | 通过（非 ExecOpEnv 返回 `kInternal`，空 env 返回 `kInvalidArgument`） |
| KernelContext/TensorView 默认值 | 通过 |
| Runtime 接口回归 | 通过 |

P0 范围内**通过**；模块整体仍未完成，P1–P4 的 mock/OCL、8 个 Kernel、Loader 集成与算子精度
测试尚未开始。

## 6. 未覆盖项与风险

| 项目 | 说明 |
| --- | --- |
| mock/OCL 后端 | P1/P4 实现；本阶段只有 FakeBackend 测试桩 |
| 8 个融合算子 Kernel | P1/P4 实现；本阶段无算子数值测试 |
| Loader/Destroyer 集成 | 依赖 RuntimeCore R1/R3；P0 未调用 |
| Buffer 地址语义 | `Storage.data_ptr = Buffer*` 需与 RuntimeCore 在 Loader 中落地（BE-C-01/02/03） |
| `kernel_types.h` 文档同步 | `Runtime/.doc/Runtime_design.md` 附录 A 仍列旧枚举，需 RuntimeCore 同步（BE-C-07） |
| OCL 主机编译 | 本机无 Khronos `CL/cl.h`，P4 前需决策 vendor 头或仅 Android 构建 |
| 真机 | 无参考机；OCL 数值验收推迟到 P4 |

## 7. 交付物状态（P0）

| 四件套 | 状态 |
| --- | --- |
| 设计文档 | ✅ `Runtime/backend/.doc/backend_design.md` |
| 代码 | ✅ P0 框架头/注册表/边界类型/CMake |
| 测试结果 | ✅ 本文件（实际执行） |
| skill | ✅ `Runtime/backend/.skill/SKILL.md` |

> P0 的“完成”仅指契约与骨架；Backend 模块整体按设计文档 P1–P5 继续交付。
