# Skill: Backend 公共契约与 P0 骨架落地

## 适用场景

- 新增/修改 `Runtime/backend/` 的公共头（`backend.h`、`kernel.h`、
  `kernel_context.h`、`exec_op_env.h`、`kernel_types.h`、`backend_registry.h`）；
- 为 Backend 模块搭建或回归 CMake 构建与公共头检查；
- 对照 `Runtime/.doc/Runtime_design.md` 校验 Backend/Kernel 接口与所有权；
- 后续 P1–P5 的 mock/OCL 后端、算子模块与测试，均以本 skill 的契约检查为前置。

不适用：Loader/Engine/Scheduler/KVCache 的内部实现；OCL Kernel 调优；
模型转换与端到端精度验收。

## 输入

1. `Runtime/.doc/Runtime_design.md`（接口与所有权权威）；
2. `Runtime/backend/.doc/backend_design.md`（本模块权威设计）；
3. 现有 `Runtime/backend/*.h/.cc` 与 `Graph/include/*.h`；
4. 编译器/构建环境：C++17、CMake 3.20+、Ninja；本机使用
   `Graph/tools/zig_cxx.cmd`（zig clang）。

## 步骤

### 1. 冻结契约

- 任何跨终端接口变更先改 `Runtime_design.md` 与附录 B，再改头文件；
- 公共头不得出现 `cl_*`/`Vk*`/`cuda*` 类型；
- 图侧类型使用 `runtime_graph::{OpType,Precision,Layout,DeviceType}`，
  后端边界精度使用 `mobilenet::DataType`；
- 每个新头/结构/函数写中文注释：功能与边界、输入输出、主要逻辑。

### 2. 修改/新增头文件

- `kernel_context.h`：定义 `TensorView/AttentionContext/KernelContext`；
  命名空间固定 `mobilenet::runtime_graph`（由 `kernel.h` 前置声明决定）；
- `exec_op_env.h/.cc`：定义 `StepContext/ExecOpEnv/SetStepContext`；
- `backend_registry.h/.cc`：名称→工厂、重复/空名拒绝、锁外执行工厂、
  `MOBILENET_REGISTER_BACKEND` 自注册；
- `kernel_types.h/.cc`：只保留 `Precision`↔`DataType` 映射，
  不恢复 `CopyKind/BlockStage/RopeLayout`。

### 3. 更新构建

```cmake
# Runtime/backend/CMakeLists.txt
if(NOT TARGET graph)
  set(GRAPH_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(GRAPH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${MOBILENET_ROOT}/Graph" "${CMAKE_BINARY_DIR}/graph")
endif()
add_library(runtime_backend STATIC backend_registry.cc exec_op_env.cc kernel_types.cc)
target_link_libraries(runtime_backend PUBLIC graph)
```

- `Runtime/backend/build/` 已加入 `.gitignore`；
- P1/P4 的 mock/ocl 以 object library 并入 `runtime_backend`，避免自注册被裁剪。

### 4. 配置、构建、测试

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
./Runtime/backend/build/backend_header_check.exe
```

### 5. 回归 Runtime 接口

```bash
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/tests/interface_check.cc \
  -o TempSpace/backend_p0_<timestamp>/interface_check.o
```

退出码必须为 0。

### 6. 记录结果

- 实际命令、输出、退出码写入 `Runtime/backend/tests/test_results.md`；
- 不得只写“测试通过”；失败保留日志与复现命令；
- P0/P1/P4 各阶段分别追加，不覆盖历史结论。

## 输出

- 可编译的 `runtime_backend` 静态库；
- `backend_header_check` 可执行检查（46 条断言，P0 基线）；
- `Runtime/backend/tests/test_results.md` 的实际执行记录；
- 契约变更说明（如有）与未覆盖项/风险。

## 常见陷阱

1. **重复 `Status` 符号**：不要一边 `add_subdirectory(Graph)` 一边把
   `Runtime/common/status.cc` 再加进自己的 target；Graph 已包含它。
2. **`KernelContext` 命名空间**：已冻结在 `mobilenet::runtime_graph`，
   不要改到 `mobilenet` 或 `mobilenet::kernel`，否则 `kernel.h` 前置声明失配。
3. **设备地址**：`Storage.data_ptr`（`kDevice`）是 `Buffer*`，不是裸设备地址；
   `TensorView.offset` 是字节偏移；非 SVM OpenCL 需要句柄+偏移。
4. **自注册被裁剪**：静态库中的 `BackendRegistrar` 可能被链接器丢弃，
   后端必须用 object library 或 whole-archive 并入。
5. **宏参数含逗号**：`MOBILENET_REGISTER_BACKEND` 传具名函数，
   不要直接传含逗号的 lambda。
6. **静态注册失败静默**：`BackendRegistrar` 在静态初始化阶段无法返回错误；
   用注册表测试（`fake_macro`）确保自注册路径可创建。
7. **`SetStepContext` 类型校验**：只接受本模块工厂创建的 `ExecOpEnv`；
   普通 `OpEnv` 返回 `kInternal`，空 env 返回 `kInvalidArgument`。
8. **zig 归档器**：未传 `BACKEND_AR/GRAPH_AR` 时 CMake 可能把 `CMAKE_AR`
   置为 NOTFOUND；必须显式覆盖。
9. **头文件自包含**：`kernel_context.h` 只前置声明 `Backend/Stream`，
   不 include `backend.h`，避免与 `kernel.h` 形成循环。
10. **构建目录**：`Runtime/backend/build/` 已忽略；不要提交构建产物。

## 后续扩展（P1–P5）

- P1：mock 后端、通用 `KernelOpEnv`/`IoOpEnv`、8 个 CPU Kernel；
- P2：Python 单算子组图 + torch 散算子参考 + Optimizer/Serializer 打包；
- P3：`op_runner` + Loader/Destroy 集成；
- P4：OCL 后端与 8 个 OCL Kernel；
- P5：每算子 `.doc`/`.skill`/`tests/test_results.md` 四件套收口。

每阶段完成后更新本 skill 与 `backend_design.md` 的对应章节。
