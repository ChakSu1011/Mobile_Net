# OCL 后端测试结果（P4 前置设备能力 + P4.1 基础设施）

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
