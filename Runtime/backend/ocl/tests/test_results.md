# OCL 后端测试结果（P4.1–P4.6）

## 1. 测试范围

本文件记录：

> 当前 OCL 开发/测试主环境为**本机 Intel Arc GPU**（第 9 节）；
> 第 2–8 节为手机 Mali-G925 的历史能力/契约结果，手机已断开。

1. **P4 前置环境验证**：手机型号、SoC、GPU、Android 版本、OpenCL 库与设备能力、
   NDK 工具链（C++17 `std::variant`）；
2. **P4.1 基础设施验证**：OclApi/OclContext/OclBuffer/OclProgramCache/OclBackend、
   SHA-256、Buffer 往返、ConvertParam、Program 编译/二进制缓存、生命周期；
   主机（Intel Arc）与手机（Mali-G925）各 38 条 PASS；SHA-256 4 条 PASS。

尚未实现/未执行：8 个算子 OCL Kernel（P4.2–P4.3）、逐算子数值对齐、性能对比
（P4.5）、端到端 OCL vs Mock Token 一致性（P4.6）。本文件不得提前写“算子通过”。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-20 |
| 手机 | vivo V2426A（产品 PD2426） |
| SoC | MediaTek MT6991（Dimensity 9400） |
| GPU | ARM Mali-G925-Immortalis MC12 r0p1 |
| GPU 驱动 | `v1.r49p1-03bet0.bc90a72cfac7e7049eb7f4a59891303a` |
| Android | 16（SDK 36），arm64-v8a |
| OpenCL | OpenCL 3.0（driver 3.0，OpenCL C 3.0） |
| OpenCL 库 | `/vendor/lib64/libOpenCL.so`，列入 `/vendor/etc/public.libraries.txt` |
| 主机 | Windows 11 Home China 10.0.26200 |
| NDK | `Reference/Neat/NEAT_Develop/android-ndk-r17b`（r27d zip 为坏文件） |
| ADB | `Reference/Neat/NEAT_Develop/platform-tools/adb.exe` |
| 探针源码 | `TempSpace/ocl_probe_20261120_212351/ocl_probe.cc`（临时文件，不提交 Git） |

## 3. 测试命令

```bash
cd d:/Mobile_Net
ADB=Reference/Neat/NEAT_Develop/platform-tools/adb.exe

# 1. 设备识别
"$ADB" devices -l
"$ADB" shell "getprop ro.product.manufacturer; getprop ro.product.model; \
  getprop ro.board.platform; getprop ro.hardware.egl; \
  getprop ro.build.version.release; getprop ro.build.version.sdk"
"$ADB" shell "dumpsys SurfaceFlinger | grep -i GLES | head -2"

# 2. OpenCL 库
"$ADB" shell "ls -la /vendor/lib64/libOpenCL.so; \
  cat /vendor/etc/public.libraries.txt | grep -i opencl"

# 3. NDK 编译探针
NDK=d:/Mobile_Net/Reference/Neat/NEAT_Develop/android-ndk-r17b
OUT=d:/Mobile_Net/TempSpace/ocl_probe_20261120_212351
cmake -S "$OUT" -B "$OUT/build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_NDK="$NDK" -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release
cmake --build "$OUT/build"

# 4. 推送/运行
MSYS_NO_PATHCONV=1 "$ADB" push "$OUT/build/ocl_probe" /data/local/tmp/ocl_probe
MSYS_NO_PATHCONV=1 "$ADB" shell \
  "chmod 755 /data/local/tmp/ocl_probe && /data/local/tmp/ocl_probe"

# 5. C++17 variant 探针
MSYS_NO_PATHCONV=1 "$ADB" push "$OUT/build/cpp17_probe" /data/local/tmp/cpp17_probe
MSYS_NO_PATHCONV=1 "$ADB" shell \
  "chmod 755 /data/local/tmp/cpp17_probe && /data/local/tmp/cpp17_probe; echo exit=\$?"
```

## 4. 实际输出

### 4.1 设备识别

```text
List of devices attached
10AF15170Z0026N        device product:PD2426 model:V2426A device:PD2426 transport_id:1

vivo
V2426A
mt6991
mali
16
36
```

```text
GLES: ARM, Mali-G925-Immortalis MC12, OpenGL ES 3.2
v1.r49p1-03bet0.bc90a72cfac7e7049eb7f4a59891303a...
```

### 4.2 OpenCL 库

```text
-rw-r--r-- 1 root root 150280 2009-01-01 08:00 /vendor/lib64/libOpenCL.so
libOpenCL.so
```

### 4.3 OpenCL 探针

```text
Could not open module param file '/sys/module/mali_kbase/parameters/large_page_conf'
platforms: 1
platform[0].name: ARM Platform
platform[0].vendor: ARM
platform[0].version: OpenCL 3.0 v1.r49p1-03bet0.bc90a72cfac7e7049eb7f4a59891303a
platform[0].devices: 1
--- device[0/1] ---
name: Mali-G925-Immortalis MC12 r0p1
vendor: ARM
version: OpenCL 3.0 v1.r49p1-03bet0.bc90a72cfac7e7049eb7f4a59891303a
driver_version: 3.0
opencl_c_version: OpenCL C 3.0 v1.r49p1-03bet0.bc90a72cfac7e7049eb7f4a59891303a
extensions: cl_khr_global_int32_base_atomics ... cl_khr_fp16 ... cl_khr_subgroups ...
  cl_khr_integer_dot_product cl_arm_scheduling_controls cl_arm_matrix_multiply ...
max_compute_units: 12
max_clock_mhz: 1612
global_mem_bytes: 11853258752
max_mem_alloc_bytes: 11853258752
local_mem_bytes: 32768
local_mem_type: 2
max_constant_buffer_bytes: 11853258752
max_constant_args: 128
max_work_group_size: 1024
max_work_item_dimensions: 3
max_work_item_sizes: 1024 1024 1024
cacheline_bytes: 64
global_cache_bytes: 2097152
mem_base_addr_align_bits: 1024
min_data_type_align_bytes: 128
max_samplers: 16
image_support: 1
preferred_vector_width_half: 8
preferred_vector_width_float: 4
svm_capabilities: 1
profiling_timer_resolution_ns: 1000
```

说明：

- `local_mem_type=2` 对应 `CL_GLOBAL`，即 Mali 的 local memory 实际走全局内存；
  local memory 优化必须实测，优先寄存器分块/subgroup；
- `cl_khr_fp16` 可用；`preferred_vector_width_half=8`；
- `svm_capabilities=1` 为 coarse-grain buffer，首期不使用 SVM；
- `max_mem_alloc_bytes` 约 11.85 GB，权重/激活单缓冲足够；
- `profiling_timer_resolution_ns=1000`，`cl_event` 计时可用。

### 4.4 NDK / C++17

```text
[1/2] Building CXX object CMakeFiles/cpp17_probe.dir/cpp17_probe.cc.o
[2/2] Linking CXX executable cpp17_probe
...
exit=0
```

NDK r17b 可编译并在设备上运行使用 `std::variant` 的 C++17 程序；`cpp17_probe` 退出码 0。
当前 r27d zip 经 `unzip -l` 检查不是有效 zip（HTML/错误页），不能作为工具链。

## 5. 结论

| 检查项 | 结论 |
| --- | --- |
| 设备连接与型号识别 | 通过 |
| GPU/OpenGL 型号 | 通过（Mali-G925-Immortalis MC12） |
| OpenCL 库可用性 | 通过（public.libraries.txt） |
| OpenCL 平台/设备查询 | 通过（OpenCL 3.0） |
| fp16/subgroup/内存/workgroup 能力 | 通过 |
| NDK r17b arm64 构建 | 通过 |
| C++17 `std::variant` 真机运行 | 通过 |

**结论**：该手机具备运行 OCL 后端的条件；P4 可以按
[`ocl_design.md`](../.doc/ocl_design.md) 实现 Backend 与 8 个 Kernel。

## 6. 未覆盖项与风险

| 项目 | 说明 |
| --- | --- |
| OCL Backend 实现 | 未实现；P4.1–P4.2 |
| 8 个 OCL Kernel | 未实现；P4.3 |
| 逐算子 vs Mock 数值 | 未执行；P4.3 |
| 逐算子 vs Mock 性能 | 未执行；P4.5 |
| 端到端 OCL vs Mock Token 一致性 | 未执行；P4.6 |
| NDK r27d | 当前包损坏；建议安装正式版本 |
| OpenCL 头 | NDK 不自带；P4 需 vendored Khronos 头 |
| 热节流/频率 | 未记录长期稳定性；P4.5 补 |

## 7. 后续测试计划

1. P4.1：OclBackend 契约测试（Buffer/ConvertParam/Program/Kernel/错误路径/泄漏）；
2. P4.3：8 个算子 vs Mock 的数值对齐；
3. P4.5：8 个算子真实形状性能，与 Mock 对比；
4. P4.6：同一模型包、同一输入 Token，OCL 与 Mock 贪心生成 Token 逐 Step 比较；
   通过标准见 `ocl_design.md` 6.6。

---

# 8. P4.1 实现结果（上下文/Buffer/传输/Program 缓存/Backend 契约）

## 8.1 本阶段实现

| 文件 | 内容 |
| --- | --- |
| `ocl/ocl_api.h/.cc` | dlopen/LoadLibrary 动态加载 OpenCL；函数表 |
| `ocl/ocl_context.h/.cc` | 平台/GPU 选择、Context/Queue（profiling）、设备能力查询 |
| `ocl/ocl_buffer.h/.cc` | `cl_mem` → `Buffer` 封装；`Storage.data_ptr=OclBuffer*` |
| `ocl/ocl_program_cache.h/.cc` | SHA-256 键、源码编译、二进制磁盘缓存、build log |
| `ocl/ocl_sha256.h/.cc` | SHA-256 实现 |
| `ocl/ocl_convert.h/.cc` | FP32/BF16→FP16、FP16→FP32 主机转换 |
| `ocl/ocl_backend.h/.cc` | Init/Deinit/Capability/Buffer/传输/ConvertParam/Synchronize；`CreateKernel` 暂返回 `kUnsupportedOp` |
| `ocl/register_ocl.cc` | 注册 `"ocl"` |
| `ocl/third_party/CL/` | vendored Khronos `cl.h`/`cl_platform.h`（MIT，见 NOTICE） |
| `ocl/tests/ocl_sha256_test.cc` | SHA-256 已知向量 |
| `ocl/tests/ocl_contract_test.cc` | Backend/Buffer/传输/ConvertParam/ProgramCache 契约；无设备返回 77 |

## 8.2 构建

### 主机（Windows + Intel Arc）

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
cmake --build Runtime/backend/build
ctest --test-dir Runtime/backend/build --output-on-failure
```

主机实测发现本机存在 Intel Arc OpenCL 3.0 设备，因此 `ocl_contract_test` 不仅编译，
也实际执行（不是 skip）。

### Android（NDK r17b + Mali-G925）

```bash
NDK=d:/Mobile_Net/Reference/Neat/NEAT_Develop/android-ndk-r17b
cmake -S Runtime/backend -B Runtime/backend/build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_NDK="$NDK" -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release \
  -DMOBILENET_BACKEND_MOCK=OFF -DMOBILENET_BACKEND_OCL=ON
cmake --build Runtime/backend/build-android

ADB=Reference/Neat/NEAT_Develop/platform-tools/adb.exe
MSYS_NO_PATHCONV=1 "$ADB" push Runtime/backend/build-android/ocl_contract_test \
  /data/local/tmp/ocl_contract_test
MSYS_NO_PATHCONV=1 "$ADB" shell \
  "chmod 755 /data/local/tmp/ocl_contract_test && /data/local/tmp/ocl_contract_test; echo exit=\$?"
```

## 8.3 实际输出

### 主机 CTest

```text
Test project D:/Mobile_Net/Runtime/backend/build
1/5 Test #1: backend_header_check .............   Passed
2/5 Test #2: backend_contract_test ............   Passed
3/5 Test #3: op_env_test ......................   Passed
4/5 Test #4: ocl_sha256_test ..................   Passed
5/5 Test #5: ocl_contract_test ................   Passed

100% tests passed out of 5
```

### 主机 `ocl_contract_test`（38 条 PASS）

```text
device.name: Intel(R) Arc(TM) 130T GPU (16GB)
device.version: OpenCL 3.0 NEO
device.driver: 32.0.101.8801
device.cu: 112
device.max_clock_mhz: 2200
device.global_bytes: 17612341248
device.max_alloc_bytes: 4294959104
device.local_bytes: 65536
device.max_wg: 1024
device.fp16: true
device.subgroups: true
PASS program cache Init
...（Program 编译/执行/缓存命中/二进制重载 全部 PASS）
PASS registry creates ocl
PASS ocl Init
...（Buffer/传输/ConvertParam/生命周期 全部 PASS）
PASS P4.1 CreateKernel unsupported
ALL TESTS PASSED
```

完整输出保存于本地运行日志；此阶段总计 **38 条 PASS**。

### 手机 Mali-G925 `ocl_contract_test`（38 条 PASS）

```text
device.name: Mali-G925-Immortalis MC12 r0p1
device.version: OpenCL 3.0 v1.r49p1-03bet0...
device.driver: 3.0
device.opencl_c: OpenCL C 3.0
device.cu: 12
device.max_clock_mhz: 1612
device.global_bytes: 11853258752
device.max_alloc_bytes: 11853258752
device.local_bytes: 32768
device.max_wg: 1024
device.fp16: true
device.subgroups: true
PASS program cache Init
PASS program compile
PASS create fill kernel
PASS create fill buffer
PASS set fill arg 0
PASS set fill arg 1
PASS enqueue fill
PASS finish fill
PASS read fill
PASS fill values
PASS program cache hit
PASS second cache Init
PASS program binary reload
PASS probe Deinit
PASS registry creates ocl
PASS ocl Init
PASS double Init rejected
PASS ocl Capability
PASS ocl supports 8 ops
PASS ocl supports H2D/D2H
PASS ocl max buffer bytes
PASS ocl min alignment
PASS CreateBuffer 257
PASS Upload
PASS Download
PASS Upload/Download roundtrip
PASS DestroyBuffer
PASS zero-byte buffer rejected
PASS non-power-of-two alignment rejected
PASS ConvertParam fp32->fp16
PASS download converted weights
PASS converted half values
PASS half roundtrip 1.0
PASS Synchronize
PASS P4.1 CreateKernel unsupported
PASS ocl Deinit
PASS double Deinit idempotent
PASS Capability after Deinit rejected
ALL TESTS PASSED
exit=0
```

### SHA-256（主机与手机）

```text
PASS sha256 empty
PASS sha256 abc
PASS sha256 quick brown fox
PASS sha256 changes with input
ALL TESTS PASSED
```

## 8.4 结论

| 检查项 | 结论 |
| --- | --- |
| OclApi 动态加载 | 通过（主机 OpenCL.dll / 手机 libOpenCL.so） |
| 平台/GPU 选择与设备信息 | 通过（Intel Arc / Mali-G925） |
| Context/Queue（profiling） | 通过 |
| Buffer 创建/销毁/对齐校验 | 通过 |
| H2D/D2H 往返 | 通过 |
| ConvertParam FP32→FP16 与 half 值 | 通过 |
| Program 编译/内存命中/二进制重载 | 通过（手机与主机） |
| Backend 生命周期/能力/Deinit 幂等 | 通过 |
| SHA-256 已知向量 | 通过 |
| P4.1 `CreateKernel` 返回 `kUnsupportedOp` | 符合预期，P4.2 接入 |

**注意**：P4.1 的 `CreateKernel` 是占位；8 个算子 Kernel、逐算子数值/性能、
端到端 OCL vs Mock Token 一致性仍未执行，不能把本阶段当作 OCL 功能完成。

## 8.5 发现并修复的问题

1. **SHA-256 旋转方向错误**：初次实现使用了左旋，导致所有已知向量失败；
   改为右旋（ROTR）后通过 empty/abc/quick brown fox 向量。
2. **`clBuildProgram` 设备指针**：不能对 `context_->device()` 返回值取地址；
   改为本地 `cl_device_id` 变量。
3. **测试路径**：CMake 中 OCL 测试源位于 `ocl/tests/`，不是 `tests/`；
   已修正 target 源路径。
4. **主机 OCL 可用**：本机存在 Intel Arc OpenCL 3.0，契约测试可直接在主机执行；
   手机 Mali-G925 同样通过。

## 8.6 未覆盖项与风险

| 项目 | 说明 |
| --- | --- |
| 8 个 OCL Kernel | P4.2/P4.3 |
| 逐算子 vs Mock 数值 | P4.3 |
| 逐算子性能 vs Mock | P4.5 |
| 端到端 OCL vs Mock Token 一致性 | P4.6 |
| NDK r27d | 仍为坏包；当前 r17b 可用 |
| OpenCL 头版本 | vendored 2.1；后续可换官方 OpenCL-Headers |
| 二进制缓存跨驱动 | 哈希含驱动/设备；换驱动会重建，符合预期 |
| `min_alignment` | 取设备 base alignment（Mali 实测 128 B）；MemoryPlan 256 是其倍数 |

---

# 9. P4.2 结果（OclKernel + 源码嵌入 + gather/add/sample）

**环境变更（2026-11-20）**：后续 OCL 开发/测试主环境改为 **本机 Intel Arc GPU**；
手机 Mali-G925 已断开，之前的手机结果保留为历史参考，重连后再补测。

## 9.1 本阶段实现

| 文件 | 内容 |
| --- | --- |
| `ocl/ocl_kernel.h/.cc` | OclKernel 基类：首次 Execute 前 Prepare、编译/创建 cl_kernel、参数绑定、1D NDRange、视图/offset 校验 |
| `ocl/ocl_kernel_factory.h/.cc` | kernel_key → Kernel 分发；P4.2 接入 gather/add/sample_argmax |
| `ocl/ocl_source_registry.h/.cc` | 构建期嵌入的 `.cl` 源码注册表 |
| `ocl/kernels/gather/` | `gather.cl` + `gather_kernel.h/.cc` |
| `ocl/kernels/add/` | `add.cl` + `add_kernel.h/.cc` |
| `ocl/kernels/sample/` | `sample.cl` + `sample_kernel.h/.cc` |
| `ocl/CMakeLists.txt` | 配置期把 `.cl` 生成 `generated/<Op>_source.h`；加入 3 个 Kernel 源 |
| `ocl/ocl_backend.cc` | `CreateKernel` 改为调用工厂；Capability 只声明 3 个已实现 op |

## 9.2 主机构建与测试（当前主环境）

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
cmake --build Runtime/backend/build
ctest --test-dir Runtime/backend/build --output-on-failure
./Runtime/backend/build/ocl_contract_test.exe
```

实际结果：

```text
设备：Intel(R) Arc(TM) 130T GPU (16GB)
OpenCL 3.0 NEO，OpenCL C 1.2，112 CU，2200 MHz，local 64 KB，max WG 1024

CTest：5/5 Passed
ocl_contract_test：48 条 PASS，exit=0
  PASS ocl add executes / ocl add values
  PASS ocl add vector executes / ocl add vector values
  PASS ocl gather executes / ocl gather values
  PASS ocl gather vector executes / ocl gather vector values
  PASS ocl sample executes / ocl sample argmax
  PASS ocl unknown kernel_key rejected
  ALL TESTS PASSED
```

说明：

- Add 测试了标量路径（4 元素）与 half8 向量路径（16 元素）；
- Gather 测试了标量路径（hidden=2）与 half8 向量路径（hidden=8）；
- Sample 测试 `[1,1,3]` argmax；
- 所有 Kernel 首次 Execute 时编译 Program，之后复用；Program 二进制缓存命中。

## 9.3 手机 Mali-G925（历史参考，已断开）

在断开前，同一 `ocl_contract_test` 在 Mali-G925 上也得到 **48 条 PASS**，
包括 add/gather 标量与 half8 路径、sample argmax、Program 编译/二进制重载。
手机当前断开，后续以 Intel Arc 为主；Mali 重连后补性能与端到端 Token 测试。

## 9.4 结论

| 检查项 | 结论 |
| --- | --- |
| OclKernel 首次 Prepare + 后续 Execute | 通过 |
| `.cl` 构建期源码嵌入 | 通过（无运行时文件读取） |
| kernel_key 分发 | gather/add/sample_argmax 通过；未知 key 拒绝 |
| Add 标量/half8 | 通过 |
| Gather 标量/half8 | 通过 |
| Sample argmax | 通过 |
| 主机 Intel Arc | 48 条 PASS（当前主环境） |
| 手机 Mali-G925 | 48 条 PASS（历史，已断开） |

## 9.5 未覆盖项

- 其余 5 个算子（rms_norm、linear、swi_glu、rope、attention）仍返回
  `kUnsupportedOp`；Capability 只声明 3 个 op；
- 逐算子 vs Mock/torch 精度、性能、端到端 Token 一致性未执行；
- Intel 与 Mali 的性能差异、workgroup 调优、half8 之外优化未做；
- 手机断开，Mali 真机性能/Token 测试暂不可执行。

---

# 10. P4.3 结果（8 个基线 Kernel + OCL vs Mock 数值对齐）

## 10.1 本阶段实现

| 目录 | 内容 |
| --- | --- |
| `kernels/rms_norm/` | `rms_norm.cl` + 宿主包装；一行一 WG、FP32 平方和 + local 归约 |
| `kernels/linear/` | `linear.cl` + 宿主包装；NT GEMM，4×4 输出块/寄存器累加 |
| `kernels/linear_swi_glu/` | `linear_swi_glu.cl` + 宿主包装；gate/up 双点积 + SiLU epilogue |
| `kernels/linear_per_head_rmsnorm_rope/` | `linear_per_head_rmsnorm_rope.cl` + 宿主包装；一 (row,head) 一 WG，per-head norm + NeoX RoPE |
| `kernels/mega_attention/` | `mega_attention.cl` + 宿主包装；causal GQA，一 query 一 work-item，FP32 online softmax |
| `ocl_source_registry.cc` | 注册 8 个 `.cl` 源码 |
| `ocl_kernel_factory.cc` | 分发 8 个 kernel_key |
| `ocl_backend.cc` | `supported_ops` 扩展为 8 个 |
| `ocl/tests/ocl_vs_mock_test.cc` | 8 个算子 OCL vs Mock 数值对齐 |

## 10.2 主机（Intel Arc，主环境）测试

```bash
cd d:/Mobile_Net
cmake --build Runtime/backend/build
ctest --test-dir Runtime/backend/build --output-on-failure
./Runtime/backend/build/ocl_vs_mock_test.exe
```

CTest：

```text
1/6 backend_header_check ......... Passed
2/6 backend_contract_test ........ Passed
3/6 op_env_test .................. Passed
4/6 ocl_sha256_test .............. Passed
5/6 ocl_contract_test ............ Passed
6/6 ocl_vs_mock_test ............. Passed

100% tests passed out of 6
```

`ocl_contract_test`：48 条 PASS（Backend/Buffer/ConvertParam/ProgramCache/
gather/add/sample 标量与 half8 路径/未知 key）。

`ocl_vs_mock_test` 实际输出：

```text
PASS run add
INFO add max_abs=0 cosine=1
PASS add matches mock
PASS run gather
INFO gather max_abs=0 cosine=1
PASS gather matches mock
PASS run rms_norm
INFO rms_norm max_abs=0 cosine=1
PASS rms_norm matches mock
PASS run linear
INFO linear max_abs=0 cosine=1
PASS linear matches mock
PASS run linear_swi_glu
INFO linear_swi_glu max_abs=0 cosine=1
PASS linear_swi_glu matches mock
PASS run per_head_rmsnorm_rope
INFO per_head_rmsnorm_rope max_abs=0 cosine=1
PASS per_head_rmsnorm_rope matches mock
PASS run mega_attention
INFO mega_attention max_abs=0 cosine=1
PASS mega_attention matches mock
PASS run sample_argmax
PASS sample_argmax token exact
ALL TESTS PASSED
```

说明：small 形状下 OCL 与 Mock 的累加顺序恰好一致，因此 `max_abs=0`；
这只能说明两条实现对同一算法/顺序一致，不能替代真实形状/随机数据的
统计误差评估（P4.5 用更大形状与多 seed 复测）。

## 10.3 Android 交叉编译

```bash
NDK=d:/Mobile_Net/Reference/Neat/NEAT_Develop/android-ndk-r17b
cmake -S Runtime/backend -B Runtime/backend/build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_NDK="$NDK" -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release \
  -DMOBILENET_BACKEND_MOCK=OFF -DMOBILENET_BACKEND_OCL=ON
cmake --build Runtime/backend/build-android
```

结果：arm64 交叉编译通过（14/14 目标）；手机已断开，未在 Mali 上运行本阶段。
Mali 的 P4.2 48 PASS 结果仍为历史参考。

## 10.4 结论

| 检查项 | 结论 |
| --- | --- |
| 8 个 OCL Kernel 接入 | 通过（Capability 8 个 op） |
| gather/add/sample 回归 | 通过 |
| rms_norm / linear / swi_glu / rope / attention | 通过（vs Mock） |
| OCL vs Mock 数值对齐 | 8/8 算子通过 |
| Android arm64 交叉编译 | 通过 |
| 主机 CTest | 6/6 通过 |

## 10.5 未覆盖项与风险

- 未做真实形状（S=256、H=1024、V=151936）数值与性能；
- 未做多 seed/随机形状的误差分布；
- 未做 torch 金标准对比（P4.4/P4.5）；
- 未做端到端 OCL vs Mock Token 一致性（P4.6）；
- Intel Arc 的主机性能不代表 Mali；Mali 重连后补测；
- 当前 linear/attention 为 naive 基线，性能优化尚未开始（P4.5）。

## 11. P4.5/P4.6 有效 token 截断与端到端一致性（2026-11-20/21）

### 11.1 本阶段实现

| 文件 | 内容 |
| --- | --- |
| `Runtime/backend/backend.h` | 新增 `UploadAt/DownloadAt`：带缓冲内字节偏移的 H2D/D2H；默认实现做指针加法（Mock），OCL 覆盖为 `clEnqueue*Buffer` 的 `buffer_offset` |
| `Runtime/executor/executor.cc` | `sample.out` 回读改用 `Backend::DownloadAt`，不再把 `Storage.offset` 加到 `cl_mem` 句柄上 |
| `Runtime/backend/ops/io_op_env.cc` | Input/Output 传输携带 `Storage.offset`，修复 arena 切片读写 |
| `Runtime/backend/kernel_context.h` | `KernelContext.packed_tokens`：生产执行时置 true，Kernel 可截断 padding 行 |
| `Runtime/backend/ops/kernel_op_env.cc` | 构造 KernelContext 时置 `packed_tokens=true` |
| `Runtime/backend/ocl/ocl_kernel.h/.cc` | `OclKernel::EffectiveRows`：`packed_tokens && batch==1` 时返回 `min(token_count, static_rows)` |
| 8 个 OCL Kernel | gather/add/rms_norm/linear/swi_glu/rope/attention/sample 全部接入有效行截断；attention 分离 `seq`（有效长度）与 `seq_stride`（静态行跨度） |
| `mega_attention.cl` | 单遍 online softmax（历史累加按 alpha 缩放，score 只算一次） |
| `linear.cl` | K 维 `vload_half8` 向量化（返回 float8）、单行尾部路径，保留 K 非 8 倍数标量尾部 |
| `linear_swi_glu.cl` / `linear_per_head_rmsnorm_rope.cl` | K 维 `vload_half8` 向量化 gate/up 双点积与 per-head GEMM |
| `Runtime/backend/ocl/tests/ocl_e2e_test.cc` | 同进程 OCL/Mock 端到端 token 对比 driver；`no-mock` 可跳过 Mock 调试 |
| `Runtime/CMakeLists.txt` | 新增 `ocl_e2e_test` 目标与 CTest `ocl_e2e_qwen3`（无设备 SKIP 77） |
| `Runtime/backend/CMakeLists.txt` | 新增 `ocl_benchmark` 目标（真实形状 OCL vs Mock，不注册 CTest） |

### 11.2 回归结果

```bash
cd d:/Mobile_Net
cmake -S Runtime -B Runtime/build
cmake --build Runtime/build
ctest --test-dir Runtime/build -E ocl_e2e_qwen3 --output-on-failure
```

实际输出（摘要）：

```text
1/7 Test #1: mock_e2e_qwen3 ...................   Passed  500.83 sec
2/7 Test #2: backend_header_check .............   Passed    0.12 sec
3/7 Test #3: backend_contract_test ............   Passed    0.01 sec
4/7 Test #4: op_env_test ......................   Passed    0.01 sec
5/7 Test #5: ocl_sha256_test ..................   Passed    0.10 sec
6/7 Test #6: ocl_contract_test ................   Passed    0.20 sec
7/7 Test #7: ocl_vs_mock_test .................   Passed    0.14 sec

100% tests passed out of 7
```

- `ocl_contract_test`：64 条 PASS（较 P4.3 的 48 条新增 5 条
  `UploadAt/DownloadAt` 切片、6 条 packed add、5 条 packed attention）；
- `backend_contract_test`：93 条 PASS（新增 Mock 默认偏移实现 5 条）；
- `ocl_vs_mock_test`：8/8 算子通过；linear 新增 K=16 向量路径与 M=1 单行
  路径用例，small 形状 `max_abs=0`、`cosine=1`；
- 在 linear 单行尾部改动后复跑非 e2e 子集（6 个目标）：100% passed，
  总耗时 0.61 s；`ocl_contract_test` 仍为 64 PASS；
- swiglu/rope half8 向量化后再次复跑同一子集：6/6 passed，总耗时 0.49 s。

### 11.3 基线端到端（P4.3 Kernel，优化前）

命令：

```bash
./Runtime/build/ocl_e2e_test.exe Models/Qwen3/artifacts/qwen3_0_6b -1
```

实际输出（旧二进制，静态 S=256 全量计算）：

```text
[ocl] loading model package...
[ocl] load done in 9446.9 ms
[ocl] Hello                    token=21806 step=6216.1 ms
[ocl] The capital of France is token=12095 step=6111.3 ms
[ocl] 1+1=                     token=17 step=6151.5 ms
[mock] loading model package...
[mock] load done in 9304.2 ms
[mock] Hello                    token=21806 step=497205.5 ms
[mock] The capital of France is token=12095 step=494373.3 ms
[mock] 1+1=                     token=17 step=487236.6 ms
Hello                    expected=21806 ocl=21806 mock=21806
The capital of France is expected=12095 ocl=12095 mock=12095
1+1=                     expected=17 ocl=17 mock=17
all ocl e2e checks passed
```

结论：OCL 与 Mock 在 3 条 prompt 上生成的 token 与 PyTorch 参考值完全一致。
Mock 单步约 487–497 s 是 CPU 参考实现的预期结果（静态 256 行全量计算）；
OCL 单步约 6.1–6.2 s。

### 11.4 优化后端到端（P4.5，当前源码）

命令（`no-mock` 只影响是否复跑 Mock，OCL 路径与 CTest 默认命令一致）：

```bash
cmake --build Runtime/build --target ocl_e2e_test
./Runtime/build/ocl_e2e_test.exe Models/Qwen3/artifacts/qwen3_0_6b -1 no-mock
```

最终源码（含 linear 单行尾部、swiglu/rope half8）实际输出：

```text
[ocl] loading model package...
[ocl] load done in 9787.7 ms
[ocl] Hello                    token=21806 step=52.2 ms
[ocl] The capital of France is token=12095 step=86.6 ms
[ocl] 1+1=                     token=17 step=68.4 ms
Hello                    expected=21806 ocl=21806
The capital of France is expected=12095 ocl=12095
1+1=                     expected=17 ocl=17
all ocl e2e checks passed
```

与 11.3 对比：同一模型包、同一 prompt、同一 token 输出；单步时间从
6216/6111/6152 ms 降到 52.2/86.6/68.4 ms（约 119x/71x/90x）。下降来自
有效 token 截断（不再计算 S=256 的 padding 行）、linear half8 + 单行尾部、
swiglu/rope half8 向量化与 attention 单遍 online softmax。

#### 11.4.1 优化后 OCL vs Mock 直接复跑

命令：

```bash
./Runtime/build/ocl_e2e_test.exe Models/Qwen3/artifacts/qwen3_0_6b -1
```

该次复跑使用加入 swiglu/rope 向量化之前的优化二进制（capping + linear
half8 + 单遍 attention），用于直接验证优化路径的 OCL/Mock token 一致性。
复跑期间同时有本地构建/基准运行，Mock 单步时间偏高；**只比较 token，不用该
段的时间数据**，干净环境耗时见 11.3：

```text
[ocl] loading model package...
[ocl] load done in 9402.7 ms
[ocl] Hello                    token=21806 step=46.9 ms
[ocl] The capital of France is token=12095 step=157.6 ms
[ocl] 1+1=                     token=17 step=128.5 ms
[mock] loading model package...
[mock] load done in 9245.0 ms
[mock] Hello                    token=21806 step=522288.6 ms
[mock] The capital of France is token=12095 step=443112.8 ms
[mock] 1+1=                     token=17 step=443640.0 ms
Hello                    expected=21806 ocl=21806 mock=21806
The capital of France is expected=12095 ocl=12095 mock=12095
1+1=                     expected=17 ocl=17 mock=17
all ocl e2e checks passed
```

结论：优化路径的 OCL 与 Mock 直接对比仍为 3/3 一致。最终源码在此基础上
只增加了 swiglu/rope 向量化与 linear 单行尾部（数值单测 `max_abs=0`），
并用上面的最终 OCL-only 复跑确认 token 未变化。

### 11.5 逐算子性能（真实形状，Intel Arc 130T）

方法：`ocl_benchmark`，每个形状先 warmup 1 次、连续 3 次
`Execute+Synchronize` 取平均；基线与优化后 **交替运行 5 组**、取中位数（ms），
以抵消 GPU 时钟/系统负载波动。基线为 P4.3 Kernel（4×4 标量 GEMM、两遍
attention、swiglu/rope 标量点积）；优化后为当前源码（half8 GEMM + 单行
尾部、单遍 online softmax、swiglu/rope half8 点积）。`lm_head` 的 Mock 过慢，
按设计跳过。

```text
bench                                   base_ms     p45_ms    gain    mock_ms    p45_vs_mock
add_262144                                0.063      0.063    0.99      2.160        34.14
gather_256x151936x1024                    0.037      0.048    0.76      0.015         0.30
rms_norm_256x1024                         0.046      0.064    0.72      2.919        45.87
linear_256x1024x1024                      2.689      1.622    1.66    833.313       513.81
linear_256x1024x3072                     13.583      5.297    2.56   2597.460       490.35
swiglu_256x1024x3072                    104.773     41.856    2.50   3795.810        90.69
rope_q_256x1024x2048                     22.668     10.218    2.22   1699.400       166.32
attention_256h16kv8d128                  39.855     32.687    1.22    587.705        17.98
sample_256x151936                         1.632      1.644    0.99     63.718        38.75
lm_head_256x1024x151936                 918.614    372.242    2.47        nan          nan
```

说明：

- `linear`（含 lm_head）提升 1.66–2.56x；`swiglu` 2.50x；`rope` 2.22x；
  `attention` 1.22x（单遍 softmax 省掉一次 score 计算，但多了累加缩放）；
- 早期非交替运行曾测得 attention 22.19 ms（1.75x），但交替复测稳定在
  32.7 ms 附近；本表采用交替中位数，不采用偏乐观的单次结果；
- `gather`/`rms_norm` 的 OCL 时间为微秒级，**launch 开销主导**：该形状只有
  256×1024 元素的查表/归一化，OCL kernel 下发+同步成本可高于 Mock 的主机
  memcpy 式实现；不是数值或功能问题；`add`/`sample` 同样已到 launch 下限；
- `Mock` 为单线程 CPU 参考实现，时间随主机负载波动；`p45_vs_mock` 仅作
  数量级参考；
- 性能数据来自主机 Intel Arc；Mali-G925 已断开，真机数据待重连后补测。

### 11.6 结论与未覆盖项

| 检查项 | 结论 |
| --- | --- |
| OCL 端到端 token（优化前） | 3/3 与 Mock、PyTorch 一致 |
| OCL 端到端 token（优化后） | 3/3 直接 OCL vs Mock 一致（11.4.1）；最终源码 OCL-only 3/3 与 golden 一致（11.4） |
| cl_mem 偏移缺陷 | 已修复（`UploadAt/DownloadAt`） |
| packed token 截断 | 8 个 Kernel 接入；unit + e2e 覆盖 |
| attention 有效长度/行跨度 | 已分离并通过 S=8/有效 S=3 单测 |
| 真实形状性能 | 交替 5 组中位数：linear/lm_head 1.66–2.56x、swiglu 2.50x、rope 2.22x、attention 1.22x |
| 单步耗时 | 6.1–6.2 s → 52.2/86.6/68.4 ms（1/5/4 token） |

未覆盖：

- 多 batch（`batch>1`）不启用截断，仍按静态 shape 全量计算；
- 未做多 token 自回归生成（每用例 max_tokens=1）与 KV cache 路径；
- 未做多 seed/随机形状误差分布；未做 torch 直接对比（用 PyTorch golden token）；
- Mali-G925 真机性能/端到端待重连；主机数据不代表手机；
- attention 仍为一 query 一 work-item，寄存器分块与 GQA K/V 复用待下一轮；
- `Mock` 在 11.4.1 的时间受并发负载影响，不可用于性能结论。
# 12. P4.8 Paged Attention 与 KV 自回归生成（2026-11-21）

## 12.1 环境

| 项目 | 内容 |
| --- | --- |
| 主机 | Windows 11，Intel Arc 130T（OpenCL 3.0 NEO） |
| 编译器 | zig clang wrapper（`Graph/tools/zig_cxx.cmd`），C++17 |
| 模型包 | `Models/Qwen3/artifacts/qwen3_0_6b_kv` |
| 图属性 | `use_kv_cache=true`、`kv_layer_index=0..27` |
| 采样 | 贪心（argmax），`max_tokens=8`，`ignore_eos=true` |

## 12.2 模型包构建

```bash
python -m ToolChain.pipeline --adapter qwen3_kv --config d:/Mobile_Net/Models/Qwen3/config.json --output d:/Mobile_Net/Models/Qwen3/artifacts/qwen3_0_6b_kv --work-dir d:/Mobile_Net/Models/Qwen3/artifacts/qwen3_0_6b_kv_work --weights d:/Mobile_Net/Models/Qwen3/model.safetensors
```

实际输出（节选）：

```text
adapter qwen3_kv
model qwen3_0_6b
tensors 624
ops 312
constants 310
weights_available 310
weights_total_bytes 1192099840
ir_sha256 07f8136982de22293ee07df2c2d40a784586e58a0e7d30429d99c33f557af28c
manifest_sha256 44f660b20c609c5baeb86cfc3d3931e32067834be9da6856becb5d202ebb9be2
package d:\Mobile_Net\Models\Qwen3\artifacts\qwen3_0_6b_kv
```

## 12.3 测试命令与结果

```bash
./Runtime/build/ocl_kv_e2e_test.exe "d:/Mobile_Net/Models/Qwen3/artifacts/qwen3_0_6b_kv"
```

实际输出：

```text
[ocl] loading KV model package...
[ocl] loading KV package for concurrent test...
Hello                    expected=21806,0,358,2776,264,2699,21815,911, ocl=21806,0,358,2776,264,2699,21815,911, concurrent=21806,0,358,2776,264,2699,21815,911, PASS
The capital of France is expected=12095,13,576,6722,315,15344,374,21718, ocl=12095,13,576,6722,315,15344,374,21718, concurrent=12095,13,576,6722,315,15344,374,21718, PASS
1+1=                     expected=17,11,220,16,10,16,28,17, ocl=17,11,220,16,10,16,28,17, concurrent=17,11,220,16,10,16,28,17, PASS
The weather is nice today expected=11,323,279,9315,374,220,17,15, ocl=11,323,279,9315,374,220,17,15, concurrent=11,323,279,9315,374,220,17,15, PASS
all ocl KV e2e checks passed

real    1m50.516s
```

说明：

- 第一组（`ocl=`）单请求执行 prefill + 8 步 decode；
- 第二组（`concurrent=`）4 请求同时提交，`max_num_tokens=8`，触发
  chunked prefill、prefill/decode 混排与多个物理块的非连续块表；
- 8 条生成序列（4 条单请求 + 4 条并发）与 2 条 20-token 多块序列全部与
  PyTorch 贪心金标准逐 token 一致（6/6 用例）。

### 12.3.1 多块 / 非连续块表用例

同一测试还包含 2 个 20-token 生成用例：每条序列 21/25 token，需要 2 个
KV 块；两个请求并发时物理块表为 `[0,2]` / `[1,3]` 这类非连续映射。

实际输出（节选）：

```text
Hello (20)                   expected=21806,0,358,2776,264,2699,21815,911,279,3491,13,6771,752,1430,311,3535,432,1156,13,576, ocl=21806,0,358,2776,264,2699,21815,911,279,3491,13,6771,752,1430,311,3535,432,1156,13,576, PASS
The capital of France is (20) expected=12095,13,576,6722,315,15344,374,21718,13,576,6722,315,17689,374,24081,13,576,6722,315,5616, ocl=12095,13,576,6722,315,15344,374,21718,13,576,6722,315,17689,374,24081,13,576,6722,315,5616, PASS
all ocl KV e2e checks passed
```

### 12.3.2 错峰提交 / 混合 prefill+decode

前两个请求先进入 decode，第 3 个请求在第 3 步后提交；同一步计划中同时
包含 decode 条目与新请求的 prefill chunk。

实际输出（节选）：

```text
Hello                    mixed=21806,0,358,2776,264,2699,21815,911, PASS
The capital of France is mixed=12095,13,576,6722,315,15344,374,21718, PASS
1+1=                     mixed=17,11,220,16,10,16,28,17, PASS
all ocl KV e2e checks passed
```

## 12.4 加载期模式一致性

```bash
./Runtime/build/ocl_kv_e2e_test.exe "d:/Mobile_Net/Models/Qwen3/artifacts/qwen3_0_6b_kv" mismatch
```

实际输出：

```text
mismatch rejected: Engine::Load rollback: Engine::Load: graph uses KV cache but recompute_attention=true
```

## 12.5 旧 prefill 路径回归

```bash
./Runtime/build/ocl_e2e_test.exe "d:/Mobile_Net/Models/Qwen3/artifacts/qwen3_0_6b" 0 no-mock
```

实际输出：

```text
[ocl] loading model package...
[ocl] load done in 9252.9 ms
[ocl] Hello                    token=21806 step=40.8 ms
Hello                    expected=21806 ocl=21806
all ocl e2e checks passed
```

## 12.6 结论与未覆盖项

| 检查项 | 结论 |
| --- | --- |
| Paged KV 写入/读取 | 通过；非连续物理块表、chunked prefill、decode 均在 e2e 覆盖 |
| OCL vs PyTorch 金标准 | 4 用例 × 8 token、2 用例 × 20 token、3 用例错峰混合 batch 全部逐 token 一致 |
| 模式错配拒绝 | 通过 |
| 旧 `use_kv_cache=false` 包 | 回归通过 |
| 单元测试 | `ocl_vs_mock_test` 全 PASS；`ocl_contract_test` 全 PASS |

未覆盖 / 说明：

- 全模型 Mock vs OCL 对比默认跳过：Mock CPU Kernel 不截断静态 256 行，
  单步耗时约 8 分钟，8 步生成不可接受；OCL 已直接对齐 PyTorch 金标准。
  需要时以 `ocl_kv_e2e_test.exe <package> mock` 运行；
- OCL 端到端未触发 KV 抢占；调度器抢占（victim 选择、recompute、统计）由
  `Runtime/scheduler/tests/scheduler_test.cc` 的 `TestKvPreemption` 覆盖；
- Paged kernel 为“一 token 一 work-item”baseline，长上下文性能未调优；
- Mali-G925 真机待重连。

---

# 13. 交互式 Token REPL（2026-11-21）

## 13.1 程序与用法

程序：`Runtime/backend/ocl/tests/ocl_token_repl.cc`，构建目标
`ocl_token_repl`。

```bash
./Runtime/build/ocl_token_repl.exe \
  --package Models/Qwen3/artifacts/qwen3_0_6b_kv \
  --max-new 8
```

终端输入一行 token id（空格/逗号/方括号分隔），程序执行 prefill +
decode 并逐 token 打印；支持多轮输入。

## 13.2 实际输出

输入：

```text
9707
quit
```

输出：

```text
loading package Models/Qwen3/artifacts/qwen3_0_6b_kv on backend ocl...
ready. enter token ids (e.g. 9707), or :help
tokens> token 21806
token 0
token 358
token 2776
token 264
token 2699
token 21815
token 911
[done] status=stopped generated=8 tokens=21806 0 358 2776 264 2699 21815 911
```

第二组：

```text
tokens> 785, 6722, 315, 9625, 374
token 12095
token 13
token 576
token 6722
```

## 13.3 结论

- REPL 在本地 Intel Arc GPU 上完成 prefill/decode 与自回归 token 输出；
- 支持 `--max-new`、`--max-num-tokens`、`--kv-mb`、`--backend`、
  `--recompute`、`--ignore-eos`、`--verbose`；
- 只处理 token id，不做 tokenizer；文本输入/输出需外部 tokenizer；
- 该程序是交互工具，不注册 CTest。

---

# 14. KV 调度性能：4 并发 × Prompt 256 × Decode 8（2026-11-21）

## 14.1 环境

| 项目 | 内容 |
| --- | --- |
| 主机 | Windows 11，Intel Arc 130T（OpenCL 3.0 NEO，driver 32.0.101.8801） |
| GPU | 112 CU，2200 MHz，64 KiB local，16.4 GiB 共享内存 |
| CPU/内存 | Core Ultra 5 225H，32 GB DDR5-5600 双通道 |
| 模型包 | `Models/Qwen3/artifacts/qwen3_0_6b_kv` |
| 调度配置 | 4 并发，prompt=256，decode=8，max_num_tokens=256，KV=512 MiB |
| 采样 | 贪心，`ignore_eos=true`；prefill 首 token + 8 次 decode |

## 14.2 方法

- 程序：[ocl_kv_benchmark.cc](ocl_kv_benchmark.cc)，目标 `ocl_kv_benchmark`；
- 先运行一次小 prompt 预热（编译 kernel），再提交 4 个 256-token prompt；
- 每个 `RunOnce` 记录墙钟时间与 `StepStats.execute_ns`；
- prefill 阶段因 `max_num_tokens=256` 被切成 4 步，每步 256 token；
- decode 阶段每步 4 token（4 个请求各 1 token）；
- 连续运行 3 次，取中位数。

命令：

```bash
./Runtime/build/ocl_kv_benchmark.exe \
  --concurrency 4 --prompt-len 256 --decode-steps 8 \
  --max-num-tokens 256 --kv-mb 512
```

## 14.3 结果（3 次中位数）

| 阶段 | 步数 | token 数 | 总墙钟 | 平均每步 | 吞吐 |
| --- | --- | --- | --- | --- | --- |
| Prefill | 4 | 1024 | 12.95 s | 3.24 s / 256 token | **79.1 token/s** |
| Decode | 8 | 32 | 2.71 s | 0.339 s / 4 token | **11.8 token/s** |
| 合计 | 12 | — | 15.66 s | — | — |

单次 TTFT（含排队等待前面的 prefill chunk）：

```text
request 1 = 3.51 s
request 2 = 6.65 s
request 3 = 9.79 s
request 4 = 12.87 s
avg       = 8.20 s
```

一次完整输出（节选）：

```text
[load]    10843.3 ms
[warmup]  done
[prefill] steps=4 tokens=1024 total_wall_ms=12873.94 avg_step_ms=3218.49 wall_tok_per_s=79.54 exec_tok_per_s=79.54
[decode]  steps=8 tokens=32 total_wall_ms=2663.99 avg_step_ms=333.00 per_token_ms=83.25 wall_tok_per_s=12.01 exec_tok_per_s=12.01
[ttft]    request 1 (id=2) = 3508.29 ms
[ttft]    request 2 (id=3) = 6650.64 ms
[ttft]    request 3 (id=4) = 9786.73 ms
[ttft]    request 4 (id=5) = 12873.94 ms
[ttft]    avg=8204.90 ms
[total]   wall_ms=15537.95
```

## 14.4 关键发现：多请求 decode 的 EffectiveRows 截断

首次测量时 4 并发 decode 每步约 **2.47 s**（1.6 token/s），而单请求
decode 约 46 ms。原因是 `OclKernel::EffectiveRows` 只在 `batch==1` 时
截断静态 256 行；KV 模式下 `batch` 是请求数，4 并发 decode 因此仍计算
256 行。

已修复为：`packed_tokens=true && token_count>0` 时统一截断到
`token_count`，不再要求 `batch==1`。修复后：

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| Decode 每步（4 token） | 2.47 s | **0.339 s** |
| Decode 吞吐 | 1.6 token/s | **11.8 token/s** |
| 提升 | — | **约 7.3×** |

Prefill 每步仍约 3.2 s，因为该阶段每步只有一个请求（batch=1），截断
行为不变。

## 14.5 分析与后续优化

- **Prefill 是当前瓶颈**：每 256 token 约 3.2 s，聚合 79 token/s；
  `lm_head` 对全部 256 行计算 logits（只有最后一行被采样），单次
  `lm_head_256` 约 344 ms；28 层 GEMM/MLP/Attention 占主要时间。
- **Decode**：4 并发每步约 0.333 s，折合每 token 83 ms（聚合 12 token/s）；
  单请求约 46 ms/token，4 并发理想值约 184 ms/step，当前 333 ms，
  批处理效率约 55%。
- 可选优化：
  1. Prefill 只对最后一个 token 计算 `lm_head`/Sample，跳过其余 255 行；
  2. GEMM 的 N 方向分块/多 workgroup，提升 256-token prefill；
  3. Paged attention 增加 query tile / GQA K/V 复用；
  4. 检查 512 MiB KV 下 GPU 是否发生共享内存带宽降级（可用更小 KV 复测）。
- 未覆盖：GPU 频率/功耗/温度采样、长时间稳定性、Mali 真机、不同
  batch/prompt 组合的曲线。

## 14.6 结论

- OCL 在 4 并发、prompt 256、decode 8 的调度场景下可完成 prefill/decode；
- 当前实测：prefill 约 **79 token/s**，decode 约 **11.8 token/s**（聚合）；
- 多请求 decode 的 `EffectiveRows` 截断缺陷已修复，decode 提升约 7.3×；
- Prefill 的 `lm_head` 全行计算是下一轮最明确的优化点。
