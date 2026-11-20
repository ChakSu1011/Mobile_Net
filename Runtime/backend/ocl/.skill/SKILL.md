# Skill: OCL 后端与融合算子开发/优化

## 适用场景

- 实现或修改 `Runtime/backend/ocl/` 的 Backend、Program 缓存、Kernel 基类；
- 新增/优化 `Runtime/backend/ocl/kernels/<op>/` 的某个融合算子 Kernel；
- 在 Mali/Adreno 真机上做 OCL 数值对齐、性能测试与调优；
- 需要把 OCL 后端与 Mock 做逐算子、端到端 Token 一致性对比。

不适用：Loader/Scheduler/Engine 内部实现；模型转换；量化 Kernel（后续阶段）。

## 输入

1. `Runtime/backend/.doc/backend_design.md`（Backend 契约）；
2. `Runtime/backend/ocl/.doc/ocl_design.md`（OCL 权威设计）；
3. `Runtime/backend/ocl/kernels/<op>/.doc/<op>_design.md`（算子契约与优化方案）；
4. Mock 参考实现：`Runtime/backend/mock/kernels/`；
5. 设备能力：`adb shell /data/local/tmp/ocl_probe` 输出（或 `tests/test_results.md` 第 2 节）；
6. NDK：`Reference/Neat/NEAT_Develop/android-ndk-r17b`（当前可用）；
7. 工具：`Reference/Neat/NEAT_Develop/platform-tools/adb.exe`。

## 步骤

### 1. 确认设备与 OpenCL

```bash
ADB=Reference/Neat/NEAT_Develop/platform-tools/adb.exe
"$ADB" devices -l
"$ADB" shell "dumpsys SurfaceFlinger | grep -i GLES"
MSYS_NO_PATHCONV=1 "$ADB" push <ocl_probe> /data/local/tmp/ocl_probe
MSYS_NO_PATHCONV=1 "$ADB" shell "chmod 755 /data/local/tmp/ocl_probe && /data/local/tmp/ocl_probe"
```

记录：GPU 型号、OpenCL 版本、CUs、时钟、global/local mem、max workgroup、fp16/extension。
当前实测：Mali-G925-Immortalis MC12，OpenCL 3.0，12 CU，1.6 GHz，32 KB local（CL_GLOBAL），
max workgroup 1024，`cl_khr_fp16`，SVM coarse。

### 2. 实现 Backend 基础设施

- `OclContext`：dlopen `libOpenCL.so` → platform → GPU device → context → queue
  （`CL_QUEUE_PROFILING_ENABLE`）；
- `OclBuffer`：`cl_mem` + bytes；`DevicePtr()` 返回 `cl_mem` 的 `void*` 表示；
- `OclBackend`：与 `MockBackend` 同签名；`Storage.data_ptr=OclBuffer*`，`TensorView.offset` 为字节偏移；
- `OclProgramCache`：源码 SHA-256 + 选项 + 驱动版本；内存/磁盘二进制缓存；编译失败落 build log；
- `OclKernel`：Prepare 编译/静态参数；Execute 绑定动态参数 + NDRange；Release 释放 `cl_kernel`。

### 3. 实现单个算子 Kernel

1. 在 `kernels/<op>/` 建目录：`.doc/<op>_design.md`、`.skill/SKILL.md`、`tests/test_plan.md`；
2. 实现 `<op>.cl` 与 `<op>_kernel.h/.cc`，提供 `Create<Op>OclKernel`；
3. 在 OCL 的 `kernel_key` 分发表登记；
4. 先在 small 形状与 Mock 对比（max_abs/relative_l2/cosine）；
5. 通过后进入真实形状性能优化。

### 4. 优化与测量

- 用 `cl_event` 记录 kernel 时间；预热 ≥3 次，重复 ≥10 次取中位数；
- 先记录 baseline，再逐项优化：half8、寄存器分块、workgroup、权重打包、subgroup、grid-stride；
- 每次优化后重跑数值对齐，防止“变快但算错”；
- 记录 kernel 时间、Mock 时间、加速比、有效带宽/FLOPs；
- Mali 注意：local memory 是 CL_GLOBAL（不要假设专用 SRAM）；优先寄存器与 subgroup；
  控制寄存器压力；cacheline 64 B；2 MB L2；
- 禁止 `-cl-fast-relaxed-math`；FP16 存储、FP32 累加。

### 5. 端到端 Token 一致性（所有算子完成后）

1. 同一模型包、同一 IR/权重哈希；
2. 至少 3 条 prompt（短/中/长），同一输入 Token；
3. 贪心 argmax，固定 `max_new_tokens`；
4. 分别用 `backend=mock` 与 `backend=ocl` 运行同一 driver；
5. 逐 Step 比较 Token；记录一致率、首个分歧 Step、logits margin；
6. 通过标准：一致率 ≥95%（目标 100%）；分歧必须定位并记录，不得放宽阈值掩盖。

### 6. 记录与收口

- `tests/test_results.md`：环境、命令、实际输出、数值/性能指标、结论、未覆盖项；
- `.skill/SKILL.md`：更新为“已验证的优化方法 + 实测数据 + 陷阱”；
- `.doc/<op>_design.md`：更新实际 workgroup/参数/性能，保持与代码一致。

## 输出

- `OclBackend` + `OclBuffer` + `OclProgramCache` + `OclKernel`；
- 8 个算子 OCL Kernel 与逐算子文档/skill/test_results；
- 后端契约测试、逐算子 vs Mock 数值/性能报告、端到端 Token 一致性报告。

## 常见陷阱

1. **local memory 类型**：Mali `CL_DEVICE_LOCAL_MEM_TYPE=CL_GLOBAL`，local 数组实际在全局内存；
   优化前先测收益，优先 subgroup/寄存器。
2. **cl_mem 不能当裸地址**：`TensorView.data` 是 `OclBuffer*`；Kernel 参数是 `(cl_mem, offset)`。
3. **偏移**：Arena 子分配必须同时传 `cl_mem` 与 `offset_bytes`；不要只传 `cl_mem`。
4. **Program 缓存键**：必须含源码、编译选项、驱动版本、设备名；否则换驱动会加载错二进制。
5. **编译选项**：禁止 fast-relaxed-math；fp16 必须确认 `cl_khr_fp16`；`-cl-mad-enable` 可用。
6. **workgroup**：max 1024；local size 必须整除 global（或使用 non-uniform 扩展）；小形状启动开销大。
7. **同步**：Kernel 内不要 `clFinish`；每 Step 由 Executor 调一次 `Synchronize`。
8. **Android 库**：`libOpenCL.so` 在 `/vendor/lib64` 且列入 public.libraries.txt；用 dlopen，
   不要链接期依赖。
9. **NDK**：当前 r27d zip 是坏文件；r17b 可用但旧；P4 建议安装正式 r27d。
10. **热节流**：手机长时间跑 GEMM 会降频；测试前静置，记录频率/温度，取中位数。
11. **端到端**：单算子通过不代表整图 Token 一致；必须做同输入 Token 的 OCL vs Mock 对比。
12. **静态注册**：OCL 后端用 object library + whole-archive，避免 `register_ocl.cc.obj` 被裁剪。

## 命令速查

```bash
# NDK 构建（当前 r17b）
NDK=d:/Mobile_Net/Reference/Neat/NEAT_Develop/android-ndk-r17b
cmake -S Runtime/backend -B Runtime/backend/build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_NDK=$NDK -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
  -DMOBILENET_BACKEND_OCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build Runtime/backend/build-android

# 部署与运行
ADB=Reference/Neat/NEAT_Develop/platform-tools/adb.exe
MSYS_NO_PATHCONV=1 "$ADB" push <binary> /data/local/tmp/<binary>
MSYS_NO_PATHCONV=1 "$ADB" shell "chmod 755 /data/local/tmp/<binary> && /data/local/tmp/<binary>"
```

## P4.1 已实现的基础设施（2026-11-20）

- `OclApi`：dlopen/LoadLibrary 动态加载；主机 Intel Arc（OpenCL 3.0）与手机
  Mali-G925 均通过；
- `OclContext`：平台/GPU 选择、Context/Queue（profiling）、设备信息查询；
- `OclBuffer`：`cl_mem` 封装；`Storage.data_ptr=OclBuffer*`；
- `OclProgramCache`：SHA-256 键、源码编译、二进制磁盘缓存、build log；
- `OclBackend`：Init/Deinit/Capability/CreateBuffer/DestroyBuffer/ConvertParam/
  Upload/Download/Synchronize；`CreateKernel` 在 P4.1 返回 `kUnsupportedOp`；
- `ocl_contract_test`：主机 38 PASS、手机 38 PASS；`ocl_sha256_test` 4 PASS。

P4.1 实测命令：

```bash
# 主机
cmake --build Runtime/backend/build --clean-first
ctest --test-dir Runtime/backend/build --output-on-failure

# Android（r17b）
cmake -S Runtime/backend -B Runtime/backend/build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake \
  -DANDROID_NDK=<NDK> -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 -DMOBILENET_BACKEND_MOCK=OFF \
  -DMOBILENET_BACKEND_OCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build Runtime/backend/build-android
MSYS_NO_PATHCONV=1 adb push Runtime/backend/build-android/ocl_contract_test \
  /data/local/tmp/ocl_contract_test
MSYS_NO_PATHCONV=1 adb shell \
  "chmod 755 /data/local/tmp/ocl_contract_test && /data/local/tmp/ocl_contract_test"
```

P4.1 新增陷阱：

1. **SHA-256 是右旋（ROTR）**：不要写成左旋；否则所有向量失败。
2. `clBuildProgram`/`clCreateProgramWithBinary` 需要 `cl_device_id` 的地址；
   先把 `context_->device()` 存入本地变量，不能对返回值取地址。
3. 本机存在 Intel Arc OpenCL 3.0，主机 OCL 测试可实际执行；不要条件反射地
   把所有失败当 skip。
4. `min_alignment` 取设备 base alignment（Mali 128 B），不是 MemoryPlan 的 256；
   Loader 只需验证 256 是其倍数。
5. vendored CL 头是 2.1；设备是 3.0；核心 API 兼容；P4.2 需要的 subgroup
   函数可通过扩展字符串/函数指针补充。

## P4.2 已实现的 Kernel 框架与首批算子（2026-11-20）

- `OclKernel`：首次 Execute 前 Prepare（编译 Program、创建 cl_kernel），
  之后 ExecuteImpl 只绑定参数/下发；提供 SetMemArg/SetIntArg/SetFloatArg/
  Enqueue1D/ValidateView/ElementOffset；
- `.cl` 源码由 CMake 在配置期生成 `generated/<Op>_source.h`，避免运行期读文件
  与手写重复源码；
- `OclKernelFactory`：kernel_key → Kernel；已接入 gather/add/sample_argmax；
- 各 Kernel 的 `.cl` + `<op>_kernel.h/.cc` 位于 `kernels/<op>/`；
- `BackendCapability.supported_ops` 只声明已实现的 3 个 op。

主机（当前主环境）验证：

```bash
cmake --build Runtime/backend/build
ctest --test-dir Runtime/backend/build --output-on-failure
./Runtime/backend/build/ocl_contract_test.exe
```

实际：Intel Arc 130T，5/5 CTest 通过，`ocl_contract_test` 48 条 PASS；
add/gather 的标量与 half8 路径、sample argmax 均通过。

P4.2 新增陷阱：

1. **源码嵌入 CMake**：`file(READ)` + `file(WRITE)` 生成 raw string header；
   源码中不要出现 `)OCLSRC"` 结束标记。
2. **offset 单位**：KernelContext 的 `TensorView.offset` 是字节；传给 Kernel 前
   必须转成元素 offset（`ElementOffset`），不要直接传字节。
3. **half8 路径条件**：元素数、a/b/out offset 均为 8 的倍数时才走向量路径；
   否则走标量，避免越界/未对齐。
4. **Sample local size**：`.cl` 固定 `__local` 数组 256，host 必须用 local=256；
   改 local size 时同步改数组。
5. **Capability 诚实**：只声明已实现 op；P4.2 为 3 个，不要提前写 8 个。
6. **Intel OpenCL C 1.2**：主机编译器按 OpenCL C 1.2 报错；不要使用 3.0-only
   语法；同一源码仍要在 Mali 可编译。
7. **首次 Execute 编译**：加载期 `CreateKernel` 只创建对象；Program 在首次
   Execute 的 Prepare 中编译，错误会带 kernel_key；P4.3 可考虑前移到 Init。

## 当前状态

- P4.1 基础设施、P4.2 Kernel 框架与 gather/add/sample 已落地；
- **当前主测试环境：本机 Intel Arc GPU**；手机 Mali-G925 已断开，历史结果保留；
- 待继续：其余 5 个算子 Kernel、逐算子 vs Mock/torch 精度、性能、端到端 Token；
- 本 skill 不得在未实测时声称算子性能达标。
