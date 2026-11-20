# Backend 模块测试结果（P0/P1）

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

---

# P1：Mock 后端与 CPU Kernel 测试结果

## 8. 测试范围

本阶段实现并验证：

- `Runtime/backend/mock/`：Mock 后端、MockBuffer、半精度工具、自注册；
- 8 个 CPU 参考 Kernel：gather、rms_norm、linear、linear_swi_glu、
  per_head_rmsnorm_rope、mega_attention、add、sample_argmax；
- `Runtime/backend/ops/`：attr 工具、OpSpec 校验、8 个 OpSpec、通用
  `KernelOpEnv`、`IoOpEnv`、`OpEnvFactory`、OpType→Spec 注册；
- 契约测试 `backend_contract_test` 与执行链测试 `op_env_test`。

本阶段**不包含**：Python 组图/torch 精度对比（P2/P3）、Loader/Serializer
集成（P3）、OCL 后端（P4）、每算子 `.doc`/`.skill` 四件套收口（P5）。

## 9. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-17 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | Clang 22.1.8（`Graph/tools/zig_cxx.cmd`，zig 0.17.1） |
| CMake / Ninja | 4.4.4 / 1.13.2 |
| 语言标准 | C++17 |
| 构建目录 | `Runtime/backend/build` |
| 工作目录 | `d:/Mobile_Net` |

## 10. 测试命令

```bash
cd d:/Mobile_Net

cmake -S Runtime/backend -B Runtime/backend/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="d:/Mobile_Net/Graph/tools/zig_cxx.cmd" \
  -DCMAKE_AR="d:/Mobile_Net/Graph/tools/zig_ar.cmd" \
  -DCMAKE_RANLIB="d:/Mobile_Net/Graph/tools/zig_ranlib.cmd" \
  -DBACKEND_AR="d:/Mobile_Net/Graph/tools/zig_ar.cmd" \
  -DBACKEND_RANLIB="d:/Mobile_Net/Graph/tools/zig_ranlib.cmd" \
  -DGRAPH_AR="d:/Mobile_Net/Graph/tools/zig_ar.cmd" \
  -DGRAPH_RANLIB="d:/Mobile_Net/Graph/tools/zig_ranlib.cmd"

cmake --build Runtime/backend/build --clean-first
ctest --test-dir Runtime/backend/build --output-on-failure

# 直接运行以查看逐条断言
./Runtime/backend/build/backend_contract_test.exe
./Runtime/backend/build/op_env_test.exe
```

## 11. 实际输出

### 11.1 构建

`--clean-first` 重新构建 51 个目标，全部成功；`-Wall -Wextra` 无警告、
无错误。构建产物包括：

- `graph/libgraph.a`
- `libruntime_backend.a`（含 `backend_mock` object library）
- `backend_header_check.exe`
- `backend_contract_test.exe`
- `op_env_test.exe`

### 11.2 CTest

```text
Test project D:/Mobile_Net/Runtime/backend/build
    Start 1: backend_header_check
1/3 Test #1: backend_header_check .............   Passed    0.05 sec
    Start 2: backend_contract_test
2/3 Test #2: backend_contract_test ............   Passed    0.02 sec
    Start 3: op_env_test
3/3 Test #3: op_env_test ......................   Passed    0.05 sec

100% tests passed out of 3

Total Test time (real) =   0.13 sec
```

### 11.3 `backend_contract_test`（88 条 PASS）

```text
PASS half 1.0 bit pattern
PASS half -2.5 bit pattern
PASS half 0.0 bit pattern
PASS half max finite
PASS half overflow to inf
PASS half negative overflow to inf
PASS half +inf
PASS half -inf
PASS half NaN
PASS half->float 1.0
PASS half->float -2.5
PASS half->float smallest subnormal
PASS bfloat16 1.0 -> half 1.0
PASS registry creates mock
PASS mock Init
PASS mock Capability
PASS mock supports 8 ops
PASS mock supports H2D/D2H
PASS mock Deinit
PASS unknown backend rejected
PASS lifecycle create
PASS lifecycle Init
PASS double Init rejected
PASS lifecycle Deinit
PASS double Deinit idempotent
PASS Capability after Deinit rejected
PASS CreateBuffer after Deinit rejected
PASS Synchronize after Deinit rejected
PASS CreateBuffer 257 bytes
PASS Upload
PASS Download
PASS Upload/Download roundtrip
PASS DestroyBuffer
PASS zero-byte buffer rejected
PASS non-power-of-two alignment rejected
PASS null Upload rejected
PASS buffer Deinit
PASS ConvertParam float32->float16
PASS ConvertParam half values
PASS ConvertParam target bytes
PASS ConvertParam unsupported dtype rejected
PASS convert Deinit
PASS CreateKernel gather
PASS DestroyKernel gather
PASS CreateKernel rms_norm
PASS DestroyKernel rms_norm
PASS CreateKernel linear
PASS DestroyKernel linear
PASS CreateKernel linear_swi_glu
PASS DestroyKernel linear_swi_glu
PASS CreateKernel per_head_rmsnorm_rope
PASS DestroyKernel per_head_rmsnorm_rope
PASS CreateKernel mega_attention
PASS DestroyKernel mega_attention
PASS CreateKernel add
PASS DestroyKernel add
PASS CreateKernel sample_argmax
PASS DestroyKernel sample_argmax
PASS unknown kernel_key rejected
PASS kernel Deinit
PASS add kernel executes
PASS add value 0
PASS add value 1
PASS add value 2
PASS add value 3
PASS gather kernel executes
PASS gather value 0
PASS gather value 1
PASS gather value 2
PASS gather value 3
PASS rms_norm kernel executes
PASS rms_norm value 0
PASS rms_norm value 1
PASS linear kernel executes
PASS linear value 0
PASS linear value 1
PASS linear_swi_glu kernel executes
PASS linear_swi_glu value 0
PASS linear_swi_glu value 1
PASS per_head_rmsnorm_rope kernel executes
PASS per_head_rmsnorm_rope value 0
PASS per_head_rmsnorm_rope value 1
PASS mega_attention kernel executes
PASS mega_attention value 0
PASS mega_attention value 1
PASS sample kernel executes
PASS sample argmax value
PASS kernel execute Deinit
ALL TESTS PASSED
```

退出码：`0`。

### 11.4 `op_env_test`（34 条 PASS）

```text
PASS spec registered for gather
PASS spec registered for rms_norm
PASS spec registered for linear
PASS spec registered for linear_swi_glu
PASS spec registered for linear_per_head_rms_norm_rope
PASS spec registered for mega_attention
PASS spec registered for add
PASS spec registered for sample
PASS unsupported OpType has no spec
PASS create input
PASS bind input host storage
PASS create rhs constant
PASS create add op
PASS create output
PASS bind output host storage
PASS sort add graph
PASS create add OpEnvs
PASS execute add graph
PASS synchronize add graph
PASS graph add value 0
PASS graph add value 1
PASS graph add value 2
PASS graph add value 3
PASS D2H add value 0
PASS D2H add value 1
PASS D2H add value 2
PASS D2H add value 3
PASS add graph Deinit
PASS sort unsupported graph
PASS unsupported op rejected
PASS unsupported Deinit
PASS sort missing attr graph
PASS missing attr rejected
PASS missing attr Deinit
ALL TESTS PASSED
```

退出码：`0`。

### 11.5 `backend_header_check`

P0 的 46 条断言继续通过（见第 4.4 节）。

## 12. 结论

| 检查项 | 结论 |
| --- | --- |
| half/bfloat16 位级转换（含 Inf/NaN/次正规） | 通过 |
| Mock 后端注册/能力/生命周期 | 通过 |
| Buffer 创建/销毁/对齐/零字节拒绝 | 通过 |
| H2D/D2H 往返 | 通过 |
| ConvertParam FP32→FP16 / 不支持 dtype | 通过 |
| 8 个 kernel_key 创建/销毁 | 通过 |
| 8 个 OpType 的 OpSpec 登记 | 通过 |
| 8 个 CPU Kernel 基础数值 | 通过 |
| Input → Add → Output 执行链（H2D/Kernel/D2H） | 通过 |
| 未知 op / 缺失 attr 的加载期拒绝 | 通过 |
| 公共头/注册表回归 | 通过 |

P1 范围内通过；尚未与 torch 做逐算子精度对比，不能作为算子验收结论。

## 13. 未覆盖项与风险

| 项目 | 说明 |
| --- | --- |
| torch 精度对比 | P2/P3：Python 组图 + Optimizer/Serializer + Loader + torch CPU 参考 |
| Loader/Serializer 集成 | 依赖 RuntimeCore Loader/Destroyer |
| 7 个算子的 OpSpec 满载图测试 | 当前 OpEnv 执行链只跑通 Add；其余算子由 Kernel 直测覆盖，P2/P3 补满载图 |
| OCL 后端 | P4；本机无 Khronos 头/真机 |
| 静态注册裁剪 | 测试用 `$<LINK_LIBRARY:WHOLE_ARCHIVE,runtime_backend>`；产品构建需同样 whole-archive 或显式锚点 |
| `Runtime/common/data_types.cc` | 独立构建时编入 `runtime_backend`；RuntimeCore 建立 common 目标后需移除，避免重复符号 |
| 性能 | Mock 为单线程朴素实现；大形状会慢，P2 用 small 形状，真实形状标 slow |

## 14. 交付物状态（P1）

| 四件套 | 状态 |
| --- | --- |
| 设计文档 | ✅ `Runtime/backend/.doc/backend_design.md`（P1 状态已更新） |
| 代码 | ✅ mock 后端 + 8 CPU Kernel + OpSpec/OpEnv 框架 |
| 测试结果 | ✅ 本节（实际执行） |
| skill | ✅ `Runtime/backend/.skill/SKILL.md`（P1 流程已补充） |

> P1 的“完成”指 Mock 后端与 CPU Kernel 链路；完整算子精度验收仍需 P2/P3 的
> Python 组图 + 模型包 + Loader + torch 对比。

## 15. Paged Attention 分发与 e2e（2026-11-21）

- `mega_attention_spec` 接受 `use_kv_cache=true`，要求 `kv_layer_index`，
  并选择 `mega_attention_paged`；
- Mock/OCL 工厂均注册 `mega_attention_paged`；
- `backend_contract_test`、`op_env_test`、`ocl_contract_test`、
  `ocl_vs_mock_test` 全部 PASS；
- Paged kernel 的真实模型路径由 `Runtime/backend/ocl/tests/ocl_kv_e2e_test.cc`
  覆盖（4 用例 × 8 token，单请求 + 4 并发，全部对齐 PyTorch 金标准）。

未新增独立的小形状 paged 单测；块表/元数据边界由 e2e 与设计文档中的
测试用例列表覆盖，后续可补 synthetic 用例。
