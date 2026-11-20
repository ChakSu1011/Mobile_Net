# OCL 后端测试结果（P4 前置：设备与 OpenCL 能力）

## 1. 测试范围

本文件当前只记录 **P4 前置环境验证**：

- 连接手机型号、SoC、GPU、Android 版本；
- OpenCL 库可用性与平台/设备能力探针；
- NDK 工具链可用性（C++17 `std::variant`）。

OCL Backend、8 个 Kernel、数值对齐、性能测试与端到端 Token 一致性尚未实现；
这些结果在 P4.1–P4.6 完成后追加，本文件不得提前写“通过”。

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
