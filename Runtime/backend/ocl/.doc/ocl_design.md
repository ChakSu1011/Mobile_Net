# OCL 后端设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.2 |
| 文档状态 | 设计完成；P4.1 基础设施、P4.2 Kernel 框架、P4.3 全部 8 个基线 Kernel 与 vs Mock 数值对齐、P4.6 端到端 Token 一致性（3/3）已落地；P4.5 第一轮优化（有效 token 截断、linear/swiglu/rope half8、单遍 attention）已完成并记录性能 |
| 编写日期 | 2026-11-20 |
| 模块 | `Runtime/backend/ocl/` |
| 权威位置 | `Runtime/backend/ocl/.doc/ocl_design.md` |
| 上游文档 | [`backend_design.md`](../../.doc/backend_design.md)、[`Runtime_design.md`](../../../.doc/Runtime_design.md)、[`MobileNet_design.md`](../../../../.doc/MobileNet_design.md) 4.4 |
| 参考实现 | `Reference/Neat/NEAT_Develop/NEAT_Develop/source/environment/ocl/src`（仅学习架构与技法，不复制专有代码） |
| 子模块文档 | `Runtime/backend/ocl/kernels/<op>/.doc/<op>_design.md`（含 Kernel 文档章节） |
| 强制规则 | `memory.md`：Google Style、中文注释、四件套、目录规范 |

**文档定位**：本文件是 OCL 后端的权威设计文档，定义 OCL 设备/上下文/队列、Buffer、传输、权重转换、Program 缓存、Kernel 接入框架、8 个融合算子的优化策略、NDK 真机测试与性能验收方法。算子级细节在
`kernels/<op>/.doc/<op>_design.md` 展开，本文件不重复。

**当前状态**：Mock 后端（P1）已完成并通过契约/执行链测试。P4.1 已落地
`OclApi/OclContext/OclBuffer/OclProgramCache/OclBackend/register_ocl`、SHA-256、
FP32/BF16→FP16 转换、vendored Khronos CL 头。P4.2 已落地 `OclKernel` 基类、
`.cl` 构建期源码嵌入、`OclKernelFactory` 分发；P4.3 已接入全部 8 个融合算子
基线 Kernel，并在主机 Intel Arc 上通过 `ocl_vs_mock_test` 与 Mock 逐算子
数值对齐（8/8 通过）。`BackendCapability.supported_ops` 已扩展为 8 个。
P4.6 已新增 `Runtime/backend/ocl/tests/ocl_e2e_test.cc`：用 Engine 加载
同一 Qwen3 模型包，分别以 `backend=ocl` 与 `backend=mock` 跑相同 prompt，
比较生成 Token 与 PyTorch 参考值；3 条 prompt 全部一致
（`ocl=12095/21806/17`，Mock 同值），详见 `tests/test_results.md` 第 11 节。
P4.5 已完成第一轮优化：`packed_tokens` 有效 token 截断（8 个 Kernel 接入）、
attention 单遍 online softmax、linear 的 K 维 `vload_half8` 向量化与单行
尾部、swiglu/rope 的 K 维 half8 向量化；OCL 单步从约 6.1 s 降到
52.2/86.6/68.4 ms（1/5/4 token）。交替 5 组基准中位数：linear/lm_head
1.66–2.56x、swiglu 2.50x、rope 2.22x、attention 1.22x（vs 优化前）。
数值回归由 `ocl_vs_mock_test` 与 `ocl_contract_test` 覆盖（64 条 PASS）。
当前主测试环境为本机 Intel Arc；手机 Mali-G925 已断开，Mali 性能待重连补测。
下一轮优化重点：attention 寄存器分块/GQA K/V 复用与多 batch 截断。

---

## 1. 背景与目标

### 1.1 背景

- Runtime 的 Backend/Kernel 契约已冻结：`Backend` 负责设备、Buffer、传输、权重、Kernel 工厂与同步；
  `Kernel` 只执行单节点；`OpEnv` 由 `KernelOpEnv`/`IoOpEnv` 统一实现；`KernelContext` 以
  `TensorView{data=Buffer*, offset, shape, dtype}` 传递设备数据。
- Mock 后端（`Runtime/backend/mock/`）已验证：
  - 8 个融合算子的 CPU 参考实现；
  - `BackendRegistry` 自注册、`OpEnvFactory`、`KernelOpEnv`/`IoOpEnv` 执行链；
  - Buffer/ConvertParam/H2D/D2H/错误路径/生命周期；
  - 契约测试 88 条、OpEnv 测试 34 条、公共头检查 46 条全部通过。
- **当前开发/测试主环境（2026-11-20 起）：本机 Intel Arc GPU**。已实测：
  `Intel(R) Arc(TM) 130T GPU (16GB)`，OpenCL 3.0 NEO，OpenCL C 1.2，112 CU，
  2200 MHz，global 17.6 GB，max alloc 4.29 GB，local 64 KB（专用类型），
  max workgroup 1024，`cl_khr_fp16`、subgroups 可用。后续 P4.2 起的 Kernel
  回归与性能测试以本机 Intel Arc 为主。
- **手机 Mali-G925 为参考目标**：2026-11-20 曾用 vivo V2426A 实测通过 P4.1
  与 P4.2 首批 Kernel；当前手机已断开，Mali 结果保留为历史参考，重连后补测
  性能与端到端 Token 一致性。

| 项目 | 值（历史/参考，2026-11-20） |
| --- | --- |
| 手机 | vivo V2426A（PD2426） |
| SoC | MediaTek MT6991（Dimensity 9400） |
| GPU | **ARM Mali-G925-Immortalis MC12 r0p1** |
| GPU 驱动 | `v1.r49p1-03bet0.bc90a72cfac7e7049eb7f4a59891303a` |
| OpenCL | **OpenCL 3.0**（driver 3.0，OpenCL C 3.0） |
| Android | 16（SDK 36），arm64-v8a |
| OpenCL 库 | `/vendor/lib64/libOpenCL.so`，且列在 `/vendor/etc/public.libraries.txt` |
| 计算单元 | 12 CU，最高 1612 MHz |
| 全局内存 / 单次分配上限 | 11,853,258,752 B（约 11.85 GB） |
| Local Memory | 32 KB，类型 `CL_GLOBAL`（非专用 SRAM） |
| 全局缓存 | 2 MB，cacheline 64 B |
| Max Workgroup | 1024；max work-item sizes `1024 1024 1024` |
| 向量偏好 | half8 / float4 |
| 扩展 | `cl_khr_fp16`、subgroups、integer dot product、`cl_arm_scheduling_controls`、`cl_arm_matrix_multiply`、image 支持 |
| SVM | coarse-grain buffer only |
| Profiling 计时精度 | 1000 ns |

- 本机 NDK：`Reference/Neat/NEAT_Develop/android-ndk-r17b` 可用（已用它编译并运行 OpenCL 探针，
  C++17 `std::variant` 可用）；`android-ndk-r27d-windows.zip` 实际是 HTML 错误页，不可用。
  P4 建议安装正式 r27d；r17b 仅作当前联调。

### 1.2 目标

| 编号 | 目标 |
| --- | --- |
| G-OCL-01 | 实现与 Mock 同构的 `Backend`：Init/Deinit/Capability/CreateBuffer/DestroyBuffer/ConvertParam/Upload/Download/CreateKernel/DestroyKernel/Synchronize |
| G-OCL-02 | 动态加载 `libOpenCL.so`，公共头不出现 `cl_*` 类型；设备/上下文/队列隐藏在 `ocl/` 内 |
| G-OCL-03 | 实现 Buffer、H2D/D2H、权重上传/打包、Program 编译与二进制缓存、Kernel 创建与同步 |
| G-OCL-04 | 为 8 个融合算子各建一个模块目录：`.doc/<op>_design.md`（含 Kernel 文档章节）、`.skill/SKILL.md`、测试计划与结果；OCL Kernel 源码在 `kernels/<op>/` |
| G-OCL-05 | 每个算子先与 Mock 对齐数值，再做 OCL 优化，最后与 Mock 对比性能并记录实测数据 |
| G-OCL-06 | 建立 NDK 真机测试路径：arm64-v8a 可执行文件 → `adb push` → `/data/local/tmp` 运行 → `cl_event` 计时 |
| G-OCL-07 | 明确 Mali-G925 的优化取舍：fp16 向量化、寄存器分块、避免依赖专用 local memory、控制 workgroup 与占用率 |
| G-OCL-08 | 所有算子完成后，用与 Mock 相同的输入 Token 做端到端生成，逐 Token 比较 OCL 与 Mock 输出是否一致 |

### 1.3 非目标

- 不实现 KV Cache 的 Paged Attention（阶段二；首期 `use_kv_cache=false`）。
- 不做量化 Kernel、Image 路径、多队列/异步传输（保留接口与扩展点）。
- 不追求所有形状的通用最优；首期以 Qwen3-0.6B 的 8 个融合算子为准。
- 不复制 NEAT 的专有 Kernel 源码；只借鉴其工程结构、数据布局与调优方法。
- 不在本文件重复 Runtime 的调度/Engine 接口；OCL 只实现 Backend/Kernel 层。

---

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收 |
| --- | --- | --- |
| FR-OCL-01 | `OclBackend` 完整实现 `Backend` 全部纯虚接口，注册名 `"ocl"` | 契约测试通过；`BackendRegistry::Create("ocl")` 成功 |
| FR-OCL-02 | 动态加载 `libOpenCL.so`，不使用链接期 OpenCL 依赖 | 未安装 OpenCL 时返回 `kBackendInitError`；设备上探针可用 |
| FR-OCL-03 | Capability 声明 FP16、8 个融合 op、H2D/D2H、`max_buffer_bytes`、`min_alignment=device base alignment（Mali 128 B）`、`attention_block_size=16`、`kNhd` | Loader 能力校验通过 |
| FR-OCL-04 | Buffer 以 `OclBuffer` 封装 `cl_mem`；`Storage.data_ptr=Buffer*`，`offset` 为字节偏移 | Kernel 按句柄+偏移寻址；Arena 子分配正确 |
| FR-OCL-05 | ConvertParam 支持 FP16 直通、FP32/BF16→FP16 主机转换、后端私有权重打包 | 权重逐元素对齐；Kernel 读到正确布局 |
| FR-OCL-06 | Program 按源码 SHA-256 + 选项 + 驱动版本缓存；支持源码编译与二进制复用 | 二次加载不重复编译；编译失败返回 `kKernelError` 并落 build log |
| FR-OCL-07 | `CreateKernel(kernel_key)` 分发到 8 个算子模块；未知 key 返回 `kUnsupportedOp` | 8 个 key 均可创建；错误路径正确 |
| FR-OCL-08 | KernelContext 的输入/输出/workspace/attention 元数据可被 OCL Kernel 读取；偏移正确 | 与 Mock 输出在阈值内一致 |
| FR-OCL-09 | 每 Step 仅 `Synchronize` 一次；Kernel 内不 `clFinish` | 执行链不出现逐 op 同步 |
| FR-OCL-10 | 错误映射：OOM、编译失败、设备丢失、非法参数均有明确错误码与 backend/stage/key | 错误可定位 |
| FR-OCL-11 | 8 个算子各有 `.doc`、`.skill`、Kernel 源码与性能/精度测试计划 | P4 四件套齐全 |
| FR-OCL-12 | 真机测试可编译、推送、运行并输出 Kernel 时间、Mock 时间、加速比 | 测试结果含实际命令与输出 |
| FR-OCL-13 | 全算子集成后，OCL 与 Mock 使用相同模型包、相同输入 Token、相同贪心采样，比较生成 Token 序列 | 一致率达标；分歧 Step 有 logits 证据 |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| NFR-OCL-01 | 正确性：单算子与 Mock FP32 参考输出在阈值内一致；端到端 Token 一致率 ≥95% |
| NFR-OCL-02 | 性能：以 Mock 为基线，每个算子优化后 Kernel 时间应优于基线；GEMM/Attention 目标为显著加速，elementwise 以带宽效率衡量 |
| NFR-OCL-03 | 内存：权重常驻、激活复用、单缓冲不超 `CL_DEVICE_MAX_MEM_ALLOC_SIZE`；不新增运行时分配热点 |
| NFR-OCL-04 | 可移植：公共头零 `cl_*`；后端目录自包含；新增后端不修改上层 |
| NFR-OCL-05 | 可测试：无 GPU 主机可编译（依赖 vendored CL 头）；有设备时自动跑真机用例，无设备时跳过并记录 |
| NFR-OCL-06 | 工程：Google Style、中文注释、四件套、测试结果不编造 |

### 2.3 约束条件

| 编号 | 约束 |
| --- | --- |
| C-OCL-01 | C++17；使用 `Utils/cpp/types.h` 的 `U8/U16/U32/U64/I32/I64/FP32/FP64/FP16/BF16/Bool/C8/Usize` 别名 |
| C-OCL-02 | 目标设备 Mali-G925 / OpenCL 3.0；fp16 扩展可用，但必须显式 `-DFP16` 并在不同设备上探测 |
| C-OCL-03 | Mali local memory 类型为 `CL_GLOBAL`（32 KB），不能假设专用 SRAM；local memory 优化必须实测 |
| C-OCL-04 | 仅 coarse-grain SVM；首期不使用 SVM，统一 `cl_mem + offset` |
| C-OCL-05 | 权重包为 FP16、row-major；权重布局转换只在 `ConvertParam` 中做，且为该后端私有 |
| C-OCL-06 | 单队列、单推理线程；`Synchronize` 由 Executor 每 Step 调用一次 |
| C-OCL-07 | 不在公共头暴露 `cl_*`、`cl_mem`、`cl_kernel`；错误消息带 native code |
| C-OCL-08 | OpenCL 头不来自 NDK；使用 vendored Khronos `CL/` 头（Apache-2.0）或本地参考头，禁止复制 NEAT 专有代码 |
| C-OCL-09 | 首期只支持 `kFloat16` 计算；BF16/FP32 仅作为 ConvertParam 源 |
| C-OCL-10 | `max_work_group_size=1024`，`max_work_item_sizes=1024³`；local size 必须是 global size 的整除（或使用 non-uniform 扩展） |

### 2.4 验收标准

| 编号 | 通过条件 |
| --- | --- |
| AC-OCL-01 | 公共头可在主机 `-fsyntax-only`；OCL 目录内部头可编译 |
| AC-OCL-02 | 设备探针成功：平台/设备/OpenCL 版本/扩展/内存/workgroup 输出正确 |
| AC-OCL-03 | 8 个算子的 OCL Kernel 与 Mock 输出在阈值内一致（见 6.4） |
| AC-OCL-04 | 8 个算子的 OCL Kernel 在真实形状下测量 kernel 时间、mock 时间与加速比；结果写入各自 `tests/test_results.md` |
| AC-OCL-05 | 每算子 `.doc`（含 Kernel 文档章节）与 `.skill` 齐全；性能数据来自实际运行 |
| AC-OCL-06 | `Backend` 契约测试通过；100 次 Init/Deinit 无句柄泄漏 |
| AC-OCL-07 | 无 OpenCL 设备时测试返回明确 skip，不伪造通过 |
| AC-OCL-08 | 端到端 OCL vs Mock：同一模型包、同一输入 Token、贪心生成的 Token 序列逐 Step 一致率 ≥95%（目标 100%）；分歧 Step 有 logits/概率证据与归因 |

---

## 3. 概要设计

### 3.1 分层与执行链

```text
Engine → IExecutor → OpEnv（KernelOpEnv/IoOpEnv）→ OclKernel::Execute
                                                   → OclBackend（clEnqueueNDRangeKernel）
                                                   → Mali-G925
Loader → OclBackend::Init/Capability/CreateBuffer/ConvertParam
```

与 Mock 完全同构：

| Mock | OCL |
| --- | --- |
| `MockBackend` | `OclBackend` |
| `MockBuffer`（主机内存） | `OclBuffer`（`cl_mem`） |
| `memcpy` Upload/Download | `clEnqueueWriteBuffer/ReadBuffer` |
| CPU Kernel | OCL Kernel（`.cl` + 宿主包装） |
| `CreateMockKernel` | `CreateOclKernel` |
| `"mock"` 注册 | `"ocl"` 注册 |

### 3.2 目录与模块划分

```text
Runtime/backend/ocl/
├── .doc/ocl_design.md                 # 本文件
├── .skill/SKILL.md                    # OCL 后端开发/优化 skill
├── tests/test_results.md              # 设备探针与后端契约测试实际结果
├── ocl_backend.h/.cc                  # Backend 实现
├── ocl_context.h/.cc                  # Platform/Device/Context/Queue
├── ocl_buffer.h/.cc                   # OclBuffer
├── ocl_program_cache.h/.cc            # Program 编译/二进制缓存
├── ocl_kernel.h/.cc                   # OclKernel 基类与公共参数绑定
├── ocl_utils.h/.cc                    # 错误映射、字符串、计时
├── register_ocl.cc                    # 自注册
├── CMakeLists.txt
└── kernels/
    ├── common/
    │   ├── ocl_kernel_common.h/.cc     # 公共 Prepare/Execute/工作尺寸/参数
    │   └── ocl_source_registry.h/.cc   # .cl 源嵌入与 hash
    ├── gather/
    │   ├── .doc/gather_design.md        # 权威设计（含 Kernel 文档章节）
    │   ├── .skill/SKILL.md
    │   ├── gather_kernel.h/.cc
    │   ├── gather.cl
    │   └── tests/test_plan.md           # 性能/精度测试计划；实测后生成 test_results.md
    ├── rms_norm/ …
    ├── linear/ …
    ├── linear_swi_glu/ …
    ├── linear_per_head_rmsnorm_rope/ …
    ├── mega_attention/ …
    ├── add/ …
    └── sample/ …
```

**为什么 Kernel 文档放在 `.doc/<op>_design.md` 内**：`memory.md` 5.1 要求每个模块的 `.doc/` 只保留一份权威设计文档；因此每个算子目录以 `.doc/<op>_design.md` 为唯一权威文档，其中设置“Kernel 文档”章节，描述 kernel 入口、workgroup、向量化、寄存器分块、边界与调优参数，不再单独放第二份设计文档。`.skill/SKILL.md` 是开发/优化方法沉淀；`tests/test_plan.md` 在实测后由 `tests/test_results.md` 承接。

### 3.3 关键流程

**加载**

```text
Loader → BackendRegistry::Create("ocl") → OclBackend::Init
  → dlopen libOpenCL.so → clGetPlatformIDs → clGetDeviceIDs (GPU)
  → clCreateContext → clCreateCommandQueue (单队列)
  → Capability 填充（FP16、8 op、H2D/D2H、max_buffer_bytes、min_alignment=device base alignment（Mali 128 B）、block=16）
  → ConvertParam(权重): 主机 dtype 转换/打包 → clCreateBuffer → clEnqueueWriteBuffer
  → CreateBuffer(arena/I/O) → 返回 OclBuffer
  → OpEnvFactory → KernelOpEnv::Init → OclBackend::CreateKernel(kernel_key)
       → ProgramCache 取/编译 → clCreateKernel → OclKernel::Prepare
```

**单步执行**

```text
Executor → 逐 op OpEnv::Forward
  → KernelOpEnv 组装 KernelContext（TensorView.data=OclBuffer*，offset）
  → OclKernel::Execute
       → 绑定 cl_mem + offset + 形状/attrs
       → clSetKernelArg（动态）
       → clEnqueueNDRangeKernel(global/local)
       →（可选）clFlush
  → Backend::Synchronize → clFinish（每 Step 一次）
```

**释放**

```text
Synchronize → ClearOpEnv（clReleaseKernel）→ DestroyBuffer（clReleaseMemObject）
→ ClearGraphEnv → Deinit（clReleaseCommandQueue/Context，Program 缓存清理）
```

### 3.4 技术选型

| 编号 | 决策 | 选择 | 理由 |
| --- | --- | --- | --- |
| D-OCL-01 | OpenCL 加载 | `dlopen("libOpenCL.so")` + 函数指针 | Android 公共库；无链接期依赖；主机可编译 |
| D-OCL-02 | 内存模型 | `cl_mem + offset`，不使用 SVM | 设备仅 coarse SVM；与 `Storage.data_ptr=Buffer*` 一致 |
| D-OCL-03 | 头文件 | vendored Khronos OpenCL 头 | NDK 不自带 CL 头；避免复制 NEAT 专有代码 |
| D-OCL-04 | Program | 源码编译 + 二进制磁盘缓存 | 包保持后端无关；二次加载快 |
| D-OCL-05 | 数学选项 | `-cl-mad-enable`；禁止 fast-relaxed-math | LLM 精度优先；FP16 存储 + FP32 累加 |
| D-OCL-06 | Kernel 准备 | `CreateKernel` 内编译与静态参数校验 | 加载期失败可定位；Execute 只做动态绑定 |
| D-OCL-07 | 同步 | 每 Step 一次 `clFinish`；Kernel 不 `clFinish` | 避免逐 op 同步 |
| D-OCL-08 | 性能评估 | `cl_event` profiling + Mock 墙钟对比 | 可区分 kernel 时间与传输/启动开销 |
| D-OCL-09 | 真机测试 | NDK 可执行文件 + adb push/run | 不依赖 APK/JNI；快速迭代 |
| D-OCL-10 | 首期内存形态 | Buffer，不用 Image | LLM 以 GEMM/Attention 为主；Image 收益待实测 |

---

## 4. 详细设计

### 4.1 Backend 接口实现

`OclBackend` 与 `MockBackend` 同签名：

```cpp
class OclBackend final : public Backend {
 public:
  Status Init(const BackendConfig& config) override;
  Status Deinit() override;
  Status Capability(BackendCapability* capability) const override;
  Status CreateBuffer(const BufferDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override;
  Status DestroyBuffer(std::unique_ptr<Buffer> buffer) override;
  Status ConvertParam(const ParamDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override;
  Status Upload(void* device_ptr, const void* host_ptr, U64 bytes,
                Stream* stream) override;
  Status Download(void* host_ptr, const void* device_ptr, U64 bytes,
                  Stream* stream) override;
  Status UploadAt(void* device_ptr, U64 offset, const void* host_ptr,
                  U64 bytes, Stream* stream) override;
  Status DownloadAt(void* host_ptr, const void* device_ptr, U64 offset,
                    U64 bytes, Stream* stream) override;
  Status CreateKernel(const KernelDesc& desc,
                      std::unique_ptr<Kernel>* kernel) override;
  Status DestroyKernel(std::unique_ptr<Kernel> kernel) override;
  Status Synchronize(Stream* stream) override;
};
```

语义：

- `Init`：动态加载 `libOpenCL.so`；`clGetPlatformIDs`/`clGetDeviceIDs(CL_DEVICE_TYPE_GPU)`；
  `clCreateContext`/`clCreateCommandQueue(CL_QUEUE_PROFILING_ENABLE)`；全有或全无。
- `Deinit`：释放 Program 缓存、Kernel 残留、Queue、Context；幂等。
- `Capability`：FP16、8 op、H2D/D2H、`max_buffer_bytes=CL_DEVICE_MAX_MEM_ALLOC_SIZE`、
  `min_alignment=device base alignment（Mali 128 B）`、`attention_block_size=16`、`kNhd`。
- `CreateBuffer`：`clCreateBuffer(CL_MEM_READ_WRITE, bytes)`；`bytes=0`/超限/对齐不满足返回错误。
- `ConvertParam`：FP16 直通；FP32/BF16→FP16 主机转换；按 kernel 需要做私有打包；
  创建 `OclBuffer` 并 `clEnqueueWriteBuffer`。
- `Upload/Download`：`clEnqueueWriteBuffer/ReadBuffer`，`device_ptr` 为 `OclBuffer::DevicePtr()`
  （`cl_mem` 的 `void*` 表示），偏移 0；等价于 `UploadAt/DownloadAt(offset=0)`。
- `UploadAt/DownloadAt`：`cl_mem` 是句柄，`offset` 必须作为 `clEnqueueWriteBuffer/
  ReadBuffer` 的 `buffer_offset` 传入，禁止对 `cl_mem` 做指针加法；用于 arena
  切片（`sample.out`、Output op 源张量）的读写。此接口是 P4.6 端到端打通时
  修复 `CL_INVALID_MEM_OBJECT` 的关键。
- `CreateKernel`：按 `kernel_key` 分发到对应算子模块的 `Create<Op>OclKernel`；
  8 个融合算子 key 全部接入，未知 key 返回 `kUnsupportedOp`。
- `Synchronize`：`clFinish`；每 Step 一次。

### 4.2 上下文与设备

```cpp
class OclContext {
 public:
  Status Init(const BackendConfig& config);
  Status Deinit();
  cl_context context() const;
  cl_device_id device() const;
  cl_command_queue queue() const;
  const OclDeviceInfo& info() const;
};
```

- 设备选择：优先 `CL_DEVICE_TYPE_GPU`；`config.device_id` 选择索引；无 GPU 返回
  `kBackendInitError`。
- 保存 `OclDeviceInfo`：name/vendor/version/driver/extensions、CUs、时钟、global/local mem、
  max alloc、max workgroup、max work-item sizes、cacheline、L2、fp16/subgroup/SVM 能力。
- 编译选项：
  - 基础：`-cl-mad-enable -cl-std=CL2.0`（设备支持 3.0 时可用 `CL3.0`）；
  - fp16：`-DFP16_ENABLE -DUSE_HALF`（仅当 `cl_khr_fp16` 存在）；
  - 禁止：`-cl-fast-relaxed-math`、`-cl-unsafe-math-optimizations`；
  - 可选：`-DWORK_GROUP_SIZE_HINT=...`（Mali 无 Qualcomm hint；按需自定义）。

### 4.3 Buffer 与地址

```cpp
class OclBuffer final : public Buffer {
 public:
  void* DevicePtr() const override;  // 返回 cl_mem 的 void* 表示
  U64 Bytes() const override;
  cl_mem mem() const;
  U64 offset() const;                // 首期 0；Arena 子分配由 TensorView.offset 表达
};
```

- `Storage.data_ptr`（`kDevice`）= `OclBuffer*`；`Storage.offset` = 字节偏移；
- `TensorView.data` = `OclBuffer*`；`TensorView.offset` 是字节偏移，由
  `OclKernel::ElementOffset` 转为元素偏移后传入 kernel（`int`）；
- Kernel 通过 `static_cast<OclBuffer*>(view.data)->mem()` 取 `cl_mem`；
- Arena 是一个大 `cl_mem`，激活按其 offset 计算地址；Kernel 参数传 `(cl_mem, offset_bytes)`。

### 4.4 Program 缓存

- 源：`.cl` 文件在构建期转成 C++ raw string（`ocl_source_registry`），避免运行时文件 IO。
- 缓存键：`SHA-256(source + build_options + driver_version + device_name + precision)`。
- 流程：
  1. 查内存缓存（`kernel_key` → `cl_program`）；
  2. 查磁盘缓存（`cache_dir/<hash>.bin`）；有则 `clCreateProgramWithBinary` + `clBuildProgram`；
  3. 无则 `clCreateProgramWithSource` + `clBuildProgram`，成功后 `clGetProgramInfo(CL_PROGRAM_BINARIES)`
     写磁盘；
  4. 失败：保存 build log 到 `cache_dir/<hash>.log`，返回 `kKernelError`。
- Cache 目录：`BackendConfig.options["cache_dir"]`；缺省不落盘（仅内存缓存）。
- 命令队列使用 `CL_QUEUE_PROFILING_ENABLE`，供 `cl_event` 计时；不改变语义。

### 4.5 OclKernel 接入框架

每个算子模块提供：

```cpp
// kernels/<op>/<op>_kernel.h
class <Op>OclKernel final : public OclKernel {
 public:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status Execute(const runtime_graph::KernelContext& context) override;
  Status Release() override;
};

Status Create<Op>OclKernel(std::unique_ptr<Kernel>* kernel);
```

`OclKernel` 基类（`ocl_kernel.h`）提供：

- `cl_kernel` 生命周期；
- 参数绑定辅助：`SetArg(index, size, ptr)`、`SetMemArg(index, cl_mem, offset)`、
  `SetIntArg`/`SetFloatArg`；
- 工作尺寸设置：`SetWorkSize(global_x, global_y, global_z, local_x, local_y, local_z)`；
- `Prepare` 中完成：Program 获取/编译、`clCreateKernel`、静态参数（权重/形状/常量）绑定；
- `Execute` 中完成：动态参数（输入/输出 buffer、offset、seq_len 等）绑定、NDRange、可选 profile；
- `Release`：`clReleaseKernel`；
- 不在 `Execute` 内 `clFinish`；错误统一映射。

**算子目录约定**：

```text
kernels/<op>/
├── .doc/<op>_design.md       # 唯一权威设计；含 Kernel 文档章节
├── .skill/SKILL.md           # 开发/优化方法
├── <op>.cl                   # OpenCL C 源
├── <op>_kernel.h/.cc         # 宿主包装 + Create<Op>OclKernel
└── tests/test_plan.md        # 性能/精度测试计划；实测后生成 test_results.md
```

#### 4.5.1 有效 token 截断（P4.5）

**问题**：Qwen3 模型包为静态形状（S=256）；实际 prompt 只有 1–N 个 token 时，
按静态形状全量计算会把 99% 的算力花在 padding 行上。

**契约**：`KernelContext.packed_tokens`（`kernel_context.h`）由 `KernelOpEnv`
在 Executor 执行时置 true；`token_count` 是本步有效 token 总数。`OclKernel::
EffectiveRows(context, static_rows)` 在 `packed_tokens && batch==1 &&
token_count>0` 时返回 `min(token_count, static_rows)`，否则返回静态行数。
直接调用 Kernel 的测试/基准（`packed_tokens=false`）保持全量计算，保证
单元测试与性能基准语义不变。

**逐算子映射**：

| Kernel | 截断维度 | 说明 |
| --- | --- | --- |
| gather | tokens | 只查有效 token 的 embedding |
| add | elements | 按最后一维把行数换算成元素数 |
| rms_norm | rows | 只归一化有效行 |
| linear | m | 只计算有效行；C 的行跨度仍为 N |
| linear_swi_glu | rows | 只计算有效行的 gate/up/SiLU |
| per_head_rmsnorm_rope | rows | positions 按有效行读取 |
| mega_attention | seq | 新增 `seq_stride` 参数：`seq` 为有效长度，`seq_stride` 为静态行跨度；q/k/v/out 的地址必须用 `seq_stride` |
| sample | rows | 只对有效行 argmax；Executor 的 `output_slot` 落在有效行内 |

**边界**：仅在 `batch==1` 启用；多 batch 的 packed 布局与静态 `[B,S]` 行跨度
不同，沿用静态全量计算，待多 batch 布局契约明确后再优化。

### 4.6 8 个算子的优化策略摘要

| 算子 | 瓶颈 | 核心策略 | 关键参数 |
| --- | --- | --- | --- |
| gather | 带宽 | 每 workgroup 处理多 token；half8 拷贝；int64 id 读取 | WG 64/128；每 token 2 KB |
| rms_norm | 带宽 + 归约 | 一行一 WG；half8 加载；FP32 平方和；subgroup 归约；避免依赖 local | 1024 行宽；WG 128/256 |
| linear | 计算/带宽 | NT GEMM；4×4/8×4 寄存器分块；half8 向量；权重打包；N 方向 grid-stride | M=256,K=1024,N≤151936 |
| linear_swi_glu | 计算 + 双 GEMM | 共享 x，gate/up 双累加；silu 融合 epilogue | I=3072 |
| per_head_rmsnorm_rope | 计算 + 逐头归约 | GEMM 后逐 head RMSNorm + RoPE epilogue；预计算/查表 cos/sin；half8 | H=16/8,D=128 |
| mega_attention | 计算 + 带宽 | 分块 online softmax；Q tile 寄存器；K/V 复用 2 个 Q head；causal 跳过 | S≤256,H=16,KV=8,D=128 |
| add | 带宽 | half8；一 WG 多元素；合并访存 | 1024 元素 |
| sample | 带宽 | 两阶段 argmax；half8 比较；确定性 tie-break | V=151936 |

详细设计、workgroup 选择、寄存器分块与实测目标见各算子 `.doc`。

### 4.7 目标设备优化要点

#### 4.7.1 Intel Arc（当前主环境）

- 设备：Intel Arc 130T（16 GB），112 CU，2200 MHz，local 64 KB（专用），
  max workgroup 1024，OpenCL 3.0 NEO，OpenCL C 1.2 编译器，`cl_khr_fp16`；
- local memory 可用，但 Kernel 仍以寄存器分块 + half8 为主；
- 编译器对 OpenCL C 1.2 语义更严格：不要使用 3.0-only 语法；
- 主机性能不代表 Mali 手机；Intel 用于正确性回归、相对调优与快速迭代；
- 同一 Kernel 源码必须在 Intel 与 Mali 都能编译（避免依赖 `cl_arm_*` 专有扩展）。

#### 4.7.2 Mali-G925（参考目标）

1. **Local memory 不可依赖**：`CL_DEVICE_LOCAL_MEM_TYPE=CL_GLOBAL`，32 KB；local 数组实际落在全局内存。
   - 优先用寄存器分块、subgroup 归约、`cl_khr_subgroup_*`；
   - 如用 local，限制 ≤16 KB 并实测收益。
2. **fp16 向量**：`preferred_vector_width_half=8`；使用 `half8`/`ushort8` 加载，FP32 累加。
3. **workgroup**：max 1024；常用 64/128/256；Mali 喜欢较大的 workgroup 与较少分支；
   避免 1 个 work-item 做过多串行工作。
4. **缓存**：2 MB L2、64 B cacheline；权重/激活按 128 B 对齐；合并访问。
5. **寄存器压力**：GEMM 分块不宜过大；用 `-cl-mad-enable`；避免 register spill。
6. **subgroup**：支持 `cl_khr_subgroup_*`；归约/scan 优先 subgroup。
7. **调度控制**：`cl_arm_scheduling_controls` 可用于 workgroup 调度提示（P4 后评估）。
8. **测量**：`cl_event` profiling（1 us 分辨率）；每 kernel 重复 ≥10 次取中位数；
   避免热节流：测试前静置、记录温度/频率；与 Mock 墙钟对比时注明 CPU 单线程。
9. **精度**：FP16 存储、FP32 累加；禁止 fast-relaxed-math；RoPE/softmax 用 FP32。

### 4.8 错误处理

| 场景 | 错误码 | 处理 |
| --- | --- | --- |
| 无 OpenCL 库/平台/GPU | `kBackendInitError`/`kDeviceError` | Init 失败并清理 |
| `clCreateBuffer` OOM / 超 max alloc | `kDeviceOom`/`kBufferError` | 加载期失败并定位 |
| Program 编译失败 | `kKernelError` | build log 落盘；消息含 kernel_key |
| `clCreateKernel` 失败 | `kKernelError` | 释放 Program 引用 |
| Enqueue/参数错误 | `kKernelError`/`kInvalidArgument` | 消息含 kernel_key、arg index |
| 设备丢失 | `kDeviceError` | 上报 Engine 受控停止 |
| 未知 kernel_key | `kUnsupportedOp` | 加载期失败 |
| 不支持的 attr/形状 | `kUnsupportedOp`/`kInvalidArgument` | 由 OpSpec/Prepare 拒绝 |

消息格式：

```text
[backend=ocl][stage=CreateKernel][kernel_key=linear] native=-54: ...
```

### 4.9 NDK 真机测试

- 工具链：`Reference/Neat/NEAT_Develop/android-ndk-r17b`（当前可用）；推荐安装 r27d（现有 zip 为坏文件）。
- 构建：
  ```bash
  cmake -S Runtime/backend -B Runtime/backend/build-android -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
    -DMOBILENET_BACKEND_OCL=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build Runtime/backend/build-android
  ```
- 部署：
  ```bash
  MSYS_NO_PATHCONV=1 adb push <binary> /data/local/tmp/<binary>
  MSYS_NO_PATHCONV=1 adb shell "chmod 755 /data/local/tmp/<binary> && /data/local/tmp/<binary>"
  ```
- OpenCL 加载：`dlopen("libOpenCL.so")`；本机验证 `/vendor/lib64/libOpenCL.so` 在 public.libraries.txt。
- 探针结果（实际）：

```text
platform[0].name: ARM Platform
platform[0].version: OpenCL 3.0 v1.r49p1-03bet0...
device: Mali-G925-Immortalis MC12 r0p1
max_compute_units: 12
max_clock_mhz: 1612
global_mem_bytes: 11853258752
max_mem_alloc_bytes: 11853258752
local_mem_bytes: 32768
local_mem_type: 2
max_work_group_size: 1024
max_work_item_sizes: 1024 1024 1024
cacheline_bytes: 64
global_cache_bytes: 2097152
preferred_vector_width_half: 8
svm_capabilities: 1
```

（完整输出见 `tests/test_results.md` 与 `TempSpace/ocl_probe_20261120_212351/`。）

**P4.1 契约测试结果**：主机（Intel Arc OpenCL 3.0）与手机（Mali-G925）各
`ocl_contract_test` 38 条 PASS；`ocl_sha256_test` 4 条 PASS。详细命令与输出见
`tests/test_results.md` 第 8 节。主机也具备 OpenCL 设备，因此契约测试在主机
实际执行而非 skip。

**P4.2 结果**：主环境 Intel Arc 上 `ocl_contract_test` 48 条 PASS，覆盖
gather/add 的标量与 half8 路径、sample argmax、Program 编译/二进制缓存；
手机 Mali-G925 断开前同样 48 条 PASS，作为历史参考。详见
`tests/test_results.md` 第 9 节。**当前开发/测试主环境为本机 Intel Arc。**

**P4.3 结果**：8 个融合算子的 OCL 基线 Kernel 已全部接入，主环境 Intel Arc 上
`ocl_vs_mock_test` 8/8 通过（small 形状 max_abs=0、cosine=1；sample token 完全
一致），`ocl_contract_test` 48 条 PASS，主机 CTest 6/6；Android arm64 交叉编译
通过但手机已断开未运行。详见 `tests/test_results.md` 第 10 节。

---

## 5. 实施步骤

### 5.1 阶段划分

| 阶段 | 内容 | 交付 |
| --- | --- | --- |
| P4.0 | 公共头/构建/NDK 路径/skill 骨架 | OCL 设计文档、CMake 骨架、头文件 |
| P4.1 ✅ | 上下文/设备/Buffer/传输/Program 缓存 | OclBackend + 契约测试（已落地） |
| P4.2 ✅ | 通用 OclKernel 基类 + kernel 分发 + 源码嵌入 | 已落地；首批 gather/add/sample_argmax 3 个 key |
| P4.3 ✅ | 8 个算子基线 Kernel + 与 Mock 数值对齐 | 全部 8 个 Kernel 已接入；`ocl_vs_mock_test` 8/8 通过 |
| P4.4 | 逐算子性能优化（寄存器分块/向量化/softmax 等） | 第一轮：linear half8/单行尾部、attention 单遍 online softmax |
| P4.5 | 真机性能测试（vs Mock）与回归 | 主机 Intel Arc 已记录（`test_results.md` 11.5）；Mali 待重连 |
| P4.6 ✅ | 全算子端到端 Token 一致性（OCL vs Mock） | 3 条 prompt 的 OCL/Mock/PyTorch token 全部一致（`test_results.md` 11.3–11.4） |
| P4.7 | OCL 后端四件套收口与审计 | 完整四件套 |

### 5.2 任务拆解

| 编号 | 任务 | 依赖 |
| --- | --- | --- |
| OCL-T01 ✅ | `OclContext` + 设备探针 + 错误映射 | 本文件 |
| OCL-T02 ✅ | `OclBuffer` + CreateBuffer/Upload/Download/ConvertParam | OCL-T01 |
| OCL-T03 ✅ | `ProgramCache` + 源码嵌入 + 构建日志 | OCL-T01 |
| OCL-T04 ✅ | `OclKernel` 基类 + `OclBackend::CreateKernel` 分发 | OCL-T02/03 |
| OCL-T05 ✅ | gather / add / sample 基线 | OCL-T04 |
| OCL-T06 ✅ | rms_norm / linear / linear_swi_glu 基线 | OCL-T04 |
| OCL-T07 ✅ | per_head_rmsnorm_rope / mega_attention 基线 | OCL-T04 |
| OCL-T08 | 逐算子优化 + 与 Mock 性能对比（第一轮：linear/attention 已优化并记录） | OCL-T05/06/07 |
| OCL-T09 | 真机测试脚本/结果/skill 收口（主机完成；Mali 待重连） | OCL-T08 |
| OCL-T10 ✅ | 全算子端到端 Token 一致性（同一输入 Token，OCL vs Mock） | OCL-T08、P3 Loader/op_runner |

### 5.3 里程碑与风险

| 里程碑 | 完成标志 |
| --- | --- |
| M-OCL-1 | OclBackend 契约测试通过；设备探针通过 |
| M-OCL-2 | 8 个基线 Kernel 与 Mock 数值对齐 |
| M-OCL-3 | 8 个算子完成优化并记录 vs Mock 性能 |
| M-OCL-4 | 四件套齐全 |

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| Mali local memory 是 CL_GLOBAL | local memory 优化可能无收益 | 优先寄存器/subgroup；实测后决定 |
| OpenCL 头缺失 | 主机无法编译 | vendored Khronos 头；或仅 NDK 构建 |
| r27d NDK 包损坏 | 工具链版本旧 | 用 r17b 验证；后续安装 r27d |
| Android 16 限制 | dlopen 公共库失败 | 已验证 libOpenCL 在 public.libraries.txt |
| 热节流/频率波动 | 性能数据不稳定 | 静置、重复、中位数、记录温度/频率 |
| Program 编译耗时 | 首次加载慢 | 二进制缓存 + 预热（P4.6） |
| 大 N（151936）GEMM | 单 WG 无法覆盖 | N 方向 grid-stride / 分块；lm_head 单独用例 |

---

## 6. 测试与验证设计

### 6.1 测试层次

| 层次 | 内容 | 环境 |
| --- | --- | --- |
| 设备探针 | Platform/Device/OpenCL 版本/扩展/内存/workgroup | 手机 |
| 后端契约 | Init/Deinit、Buffer 往返、ConvertParam、Kernel 创建、错误路径、泄漏 | 手机 |
| 算子数值 | 每个 OCL Kernel vs Mock CPU 参考 | 手机 |
| 算子性能 | 每个 OCL Kernel 时间 vs Mock 时间；Kernel-only 与端到端 | 手机 |
| 回归 | 8 算子 + 后端契约 | 手机 |

### 6.2 测试数据与形状

- 默认 small：`B=1, S=8, H=64, I=128, heads=4, kv_heads=2, D=16, V=128`；
- 真实形状（性能）：`B=1, S=256, H=1024, I=3072, heads=16, kv_heads=8, D=128, V=151936`；
- 每算子至少：default、`S=1`、zero/边界、不同 seed。

### 6.3 正确性阈值

与 Mock FP32 参考对比：

| 算子 | max_abs | relative_l2 | cosine |
| --- | --- | --- | --- |
| gather | 0 | 0 | 1.0 |
| add | 5e-3 | 1e-3 | 0.99999 |
| rms_norm | 2e-2 | 5e-3 | 0.9999 |
| linear | 2e-2 | 5e-3 | 0.9999 |
| linear_swi_glu | 2e-2 | 5e-3 | 0.9999 |
| per_head_rmsnorm_rope | 2e-2 | 5e-3 | 0.9999 |
| mega_attention | 3e-2 | 1e-2 | 0.999 |
| sample | Token 100% 一致 | — | — |

### 6.4 性能测试协议

1. 预热 ≥3 次；
2. 每算子重复 ≥10 次，记录 `cl_event` kernel 时间（中位数）；
3. 记录：kernel 时间、H2D/D2H 字节与时间、总 Step 时间、Mock 单次执行时间；
4. 加速比 = Mock 时间 / OCL kernel 时间；同时给出 OCL kernel 的有效带宽/FLOPs；
5. 真实形状与 small 形状分别报告；小形状标注“启动开销主导”；
6. 记录设备温度/频率（若可读）、Android 版本、驱动版本；
7. 结果写入每算子 `tests/test_results.md`，不得只写“更快”。

### 6.5 通过标准

- 契约测试全部通过；无句柄泄漏；
- 8 个算子正确性在阈值内；
- 每个算子至少一组真实形状性能数据；与 Mock 对比有明确结论（加速或说明瓶颈）；
- 全算子端到端：同一输入 Token 的 OCL vs Mock 生成 Token 一致率 ≥95%（目标 100%），分歧有证据；
- 每算子 `.doc`/`.skill`/Kernel 源码/测试结果齐全；
- 无设备时明确 skip，不伪造通过。

### 6.6 端到端 Token 一致性（OCL vs Mock）

**目的**：所有算子分别通过 Mock 数值对齐后，验证整图执行路径（Loader → OpEnv → Kernel → Sample）没有累积偏差，且 OCL 与 Mock 对同一输入生成相同 Token。

**方法**（复用 Mock 的测试方式，不另造一套）：

1. 使用同一个 Qwen3 `model.mgf` + `weights.bin`（同一 IR 哈希、同一权重 SHA-256）；
2. 固定输入 Token 序列：至少 3 条 prompt（短/中/长），记录 token ids 与 position ids；
3. 固定贪心采样（`method=argmax`、`temperature=0`、固定 seed；Sample 输出确定性）；
4. 分别以 `backend=mock` 与 `backend=ocl` 运行 P3 的 `op_runner` 或端到端 driver：
   - 同一 `S`、同一 `max_new_tokens`（如 16/32/64）；
   - 每一步记录生成 Token、首 Token、logits（若开启调试导出）；
5. 逐 Step 比较：
   - Token 序列一致率 = 相同位置 Token 数 / 总生成 Token 数；
   - 首个分歧 Step、该 Step 的 top-1/top-2 logits 与 margin；
   - 若允许，导出 `lm_head.out` 最后一行的 logits，计算余弦相似度与 max abs；
6. 结果写入 `Runtime/backend/ocl/tests/test_results.md` 的“端到端 Token 一致性”章节。

**通过标准**：

- Token 一致率 ≥95%（与设计文档 6.4 的整体验收一致）；**目标 100%**；
- 若出现分歧，必须定位到算子/层（逐层 logits 对比）并记录原因；不得用放宽阈值掩盖；
- 同一输入重复运行结果确定（argmax 无随机性；若未来引入采样，固定 seed 并记录）。

**实现映射（P4.6）**：

- driver：`Runtime/backend/ocl/tests/ocl_e2e_test.cc`，在 `Runtime/CMakeLists.txt`
  中构建为 `ocl_e2e_test`，并注册 CTest `ocl_e2e_qwen3`（无设备返回 77）；
- 同一进程内先跑 `backend=ocl`、再跑 `backend=mock`：`RunBackend` 各自
  `Engine::Load` → 逐用例 `AddRequest`/`RunOnce` → 响应回调取 token →
  `Engine::Destroy`；两个后端共享同一模型包与 PyTorch golden；
- 判定：`ocl_token == expected_token` 且 `ocl_token == mock_token`；缺失 token
  或执行失败即失败，不跳过；
- CLI：`ocl_e2e_test <package> [only_case] [no-mock]`，`no-mock` 仅用于
  调试 OCL 路径时省去 Mock 的长时间 CPU 前向；
- 关键修复：`Storage.offset` 位于 arena 切片时，Executor 原先把 offset 加到
  `cl_mem` 句柄上，触发 `CL_INVALID_MEM_OBJECT`；新增
  `Backend::UploadAt/DownloadAt`（见 `backend_design.md` 4.1.1），OCL 映射为
  `clEnqueue*Buffer` 的 `buffer_offset`。

**与单算子测试的关系**：单算子 vs Mock 保证 kernel 级正确；端到端 Token 保证融合图、I/O、Sample、packing/offset 与多算子累积误差可接受。两者都通过才算 OCL 后端可用。

### 6.7 未覆盖项与风险

- 未覆盖 Paged KV / Decode attention（阶段二）；
- 未覆盖 Image 路径、SVM、量化；
- 未覆盖多设备/多驱动矩阵（首期锁定 Mali-G925）；
- 未覆盖 APK/JNI 集成（真机测试用可执行文件）；
- 未覆盖长期稳定性/热节流（P4.6 补）。

---

## 附录 A：设备与 OpenCL 探针实际输出

见 `Runtime/backend/ocl/tests/test_results.md` 第 2 节（完整输出）与
`TempSpace/ocl_probe_20261120_212351/ocl_probe.cc`（探针源码，临时文件，不提交）。

## 附录 B：NEAT OCL 参考与取舍

| NEAT 做法 | 采用/改造 |
| --- | --- |
| 动态加载 OpenCL、函数指针表 | 采用 |
| Buffer/Image 双形态、RGBA 打包 | 首期仅 Buffer；Image 后评估 |
| Program 名称后缀 + 预编译二进制 | 改为源码编译 + 二进制磁盘缓存（包可移植） |
| 权重按 4 元素对齐打包 | 采用；在 ConvertParam 中做 |
| 内存池/首次适配 | Runtime 已有 MemoryPlan；OCL 只按计划 CreateBuffer |
| `-cl-fast-relaxed-math` | **不采用**（LLM 精度） |
| 中心 op×后端宏表 | 不采用；用 kernel_key 分发 + 目录自注册 |
| 逐 kernel `clFlush` | 首期保留 `clFlush`；每 Step 一次 `clFinish` |
| host 后端 | 对应我们的 Mock 后端（已实现） |

## 附录 C：逐算子文档索引

| 算子 | 设计文档 | Kernel 源 | 测试计划 |
| --- | --- | --- | --- |
| gather | `kernels/gather/.doc/gather_design.md` | `kernels/gather/gather.cl` | `kernels/gather/tests/test_plan.md` |
| rms_norm | `kernels/rms_norm/.doc/rms_norm_design.md` | `rms_norm.cl` | `tests/test_plan.md` |
| linear | `kernels/linear/.doc/linear_design.md` | `linear.cl` | `tests/test_plan.md` |
| linear_swi_glu | `kernels/linear_swi_glu/.doc/linear_swi_glu_design.md` | `linear_swi_glu.cl` | `tests/test_plan.md` |
| linear_per_head_rmsnorm_rope | `kernels/linear_per_head_rmsnorm_rope/.doc/linear_per_head_rmsnorm_rope_design.md` | 同名 `.cl` | `tests/test_plan.md` |
| mega_attention | `kernels/mega_attention/.doc/mega_attention_design.md` | `mega_attention.cl` | `tests/test_plan.md` |
| add | `kernels/add/.doc/add_design.md` | `add.cl` | `tests/test_plan.md` |
| sample | `kernels/sample/.doc/sample_design.md` | `sample.cl` | `tests/test_plan.md` |
