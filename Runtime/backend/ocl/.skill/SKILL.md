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
13. **改 `.cl` 后必须重新 configure**：源码是在 CMake 配置期用 `file(READ)`
    嵌入 `generated/<Op>_source.h` 的；只改 `.cl` 时 Ninja 可能显示 "no work
    to do"，运行的是旧内核。执行 `cmake -S Runtime -B Runtime/build` 后再 build。
14. **`vload_halfN` 返回 `floatN`**：不是 `halfN`；写成 `half8 v = vload_half8(...)`
    会编译失败（Intel 报 incompatible type）。向量化 GEMM 直接声明 `float8`。

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

## P4.3 已实现的 8 个基线 Kernel（2026-11-20）

已接入全部 8 个融合算子（Capability 8 个 op）：

| Kernel | 基线策略 | 备注 |
| --- | --- | --- |
| gather | 一 token 一 WG；half8/标量 | P4.2 已实现 |
| add | half8/标量 | P4.2 已实现 |
| sample_argmax | 一行一 WG；local 归约 | P4.2 已实现 |
| rms_norm | 一行一 WG；FP32 平方和 + local 归约 | P4.3 |
| linear | NT GEMM，4×4 输出块，FP32 累加 | P4.3 |
| linear_swi_glu | 一输出一 work-item；gate/up 双点积 + SiLU | P4.3 |
| per_head_rmsnorm_rope | 一 (row,head) 一 WG；per-head norm + NeoX RoPE | P4.3 |
| mega_attention | 一 query 一 work-item；causal GQA + online softmax | P4.3 |

主机 Intel Arc 结果：

```bash
cmake --build Runtime/backend/build
ctest --test-dir Runtime/backend/build --output-on-failure
./Runtime/backend/build/ocl_vs_mock_test.exe
```

- CTest 6/6 通过；
- `ocl_contract_test` 48 PASS；
- `ocl_vs_mock_test` 8/8 通过，small 形状下 max_abs=0、cosine=1，
  sample token 完全一致；
- Android arm64 交叉编译通过，但手机已断开，未做 Mali 运行。

P4.3 新增陷阱：

1. **small 形状 max_abs=0 不代表真实形状也零误差**：两条实现恰好同序；
   P4.5 要用真实形状/多 seed 做统计误差与性能。
2. **linear 4×4 tile 的越界**：`tiles_n/tiles_m`、`row<m`/`col<n` 必须保护；
   全局 work-item 可能超过 tile 数。
3. **per_head 的 head 归约**：D≤128；local 数组 128；`lane<head_dim` 才计算，
   但所有 lane 都要参与 barrier。
4. **attention 的 private `acc[128]`**：D≤128；真实 S=256 时该基线很慢，
   P4.5 必须改成分块 online softmax。
5. **rope 的 position 是 token position_ids**，不是行号；必须读 positions buffer。
6. **Sample 的 local size 固定 256**，与 `.cl` 中 `__local[256]` 一致。
7. **Capability 8 个 op** 必须与 factory 分发一致；新增算子同步两处。

## P4.5 有效 token 截断（2026-11-20）

**场景**：模型包是静态 S=256，实际 prompt 只有 1–N 个 token；不截断会浪费
99% 的算力。`KernelContext.packed_tokens` 由 `KernelOpEnv` 在生产执行时置
true，`token_count` 为本步有效 token 数；`OclKernel::EffectiveRows` 返回
`min(token_count, static_rows)`。测试/基准不设置该标志，保持全量。

**改法**：

1. 每个 Kernel 的 `ExecuteImpl` 里计算 `valid_rows = EffectiveRows(...)`；
2. 用 `valid_rows` 设置 kernel 的行数参数与 global size；
3. `Prepare` 仍按静态 shape 校验（positions 长度等），不要改用有效行；
4. attention 必须把“有效长度”和“物理行跨度”分开：`.cl` 新增
   `seq_stride`，`seq` 只作循环上界，q/k/v/out 地址用 `seq_stride`；
5. add 这类扁平 Kernel 按最后一维换算：`valid_elements = valid_rows * last_dim`。

**陷阱**：

- 只改 `rows` 不改 `prepare` 校验没问题；反过来会误报；
- 把 attention 的 `seq` 同时当 stride 会导致 K/V 读错行（packed 行距 ≠ 静态行距）；
- `batch != 1` 时不要截断：packed 布局与 `[B,S]` 行跨度不一致；
- 基准测试不设 `packed_tokens`，否则测到的是 1 行而不是 256 行。

## P4.6 端到端 Token 一致性（2026-11-20）

- driver：`Runtime/backend/ocl/tests/ocl_e2e_test.cc`；在 `Runtime/build` 下：
  `cmake --build Runtime/build --target ocl_e2e_test`；
- 同一进程内先 OCL 后 Mock，同一模型包、同一 prompt，比较 OCL/Mock/PyTorch
  golden 三方 token；`no-mock` 参数用于调试时跳过 Mock；
- 命令与实测输出见 `Runtime/backend/ocl/tests/test_results.md` 第 11 节；
- 复用的既有链路：`Engine::Load/AddRequest/RunOnce/SetResponseCallback/Destroy`，
  与 `Runtime/tests/mock_e2e_test.cc` 完全同构，只替换 `runtime.backend`；
- CTest 目标 `ocl_e2e_qwen3` 只跑 case 0（整个进程仍会加载两次模型）。

P4.6 关键陷阱：

1. **cl_mem 不能做指针加法**：Executor 回读 `sample.out` 时曾把
   `Storage.offset` 加到 `DevicePtr()`（cl_mem 句柄）上，触发
   `CL_INVALID_MEM_OBJECT`。arena 切片传输必须走 `Backend::UploadAt/
   DownloadAt`，OCL 映射为 `clEnqueue*Buffer` 的 `buffer_offset`。
2. **Output/IoOpEnv 同样受影响**：Input/Output op 的存储可能在 arena 非零
   偏移处，`IoOpEnv` 必须传 `Storage.offset`，不能只传 `buffer->DevicePtr()`。
3. **Mock 很慢是预期**：静态图 S=256，Mock 单步 ~473 s（CPU 参考实现）；
   端到端对比时先跑 OCL，再用 `no-mock` 调试，避免每次等待 Mock。
4. **静态图 padding**：Qwen3 模型包输入形状固定 S=256，prompt 只有 1/5/4
   token 时仍按 256 行计算；因此不同 prompt 的单步时间接近。P4.5 可评估
   按 `KernelContext.token_count` 截断有效行。
5. **无设备返回 77**：`ProbeOclBackend` 在加载模型前探测；CTest 记为
   Skipped，不要误报通过。

## 当前状态

- P4.1 基础设施、P4.2 Kernel 框架、P4.3 全部 8 个基线 Kernel 与 vs Mock
  数值对齐、P4.6 端到端 OCL vs Mock Token 一致性（3/3）均已落地；
  实测输出见 `ocl/tests/test_results.md` 第 11 节；
- P4.5 第一轮优化已落地：`packed_tokens` 有效 token 截断、attention 单遍
  online softmax、linear 的 K 维 `vload_half8` 向量化与单行尾部；
  OCL 单步 6.1 s → 51.7/161.4/130.2 ms（1/5/4 token），linear/lm_head
  1.85–2.86x、attention 1.75x；数值回归通过 `ocl_vs_mock_test` 与 packed 用例；
- **当前主测试环境：本机 Intel Arc GPU**；手机 Mali-G925 已断开，历史结果保留；
- 待继续：swiglu/rope 标量 Kernel 优化、Mali 真机性能与端到端；
- 本 skill 不得在未实测时声称算子性能达标。
