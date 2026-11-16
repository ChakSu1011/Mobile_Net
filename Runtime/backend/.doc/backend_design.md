# Backend 模块设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 文档状态 | 设计完成，代码/测试/skill 待 P1–P5 落地 |
| 编写日期 | 2026-11-17 |
| 模块 | `Runtime/backend/` |
| 权威位置 | `Runtime/backend/.doc/backend_design.md` |
| 上游文档 | [`.doc/MobileNet_design.md`](../../../.doc/MobileNet_design.md) 第 4.4 节；[`Runtime/.doc/Runtime_design.md`](../../.doc/Runtime_design.md) 第 4.3、4.4、4.6、4.7 节与附录 B |
| 子模块文档 | `Runtime/backend/ocl/.doc/ocl_design.md`；`Runtime/backend/ops/<op>/.doc/<op>_design.md` |
| 强制规则 | `memory.md`：Google Style、中文注释、四件套、目录规范 |

**文档定位**：本文件是 Backend 模块（含 Kernel 公共契约）的权威设计文档。Runtime 统一架构与跨模块接口以
`Runtime/.doc/Runtime_design.md` 为准；本文件只展开 Backend/Kernel 的职责边界、接口语义、后端插件、
算子模块化与算子精度测试方法，不重复定义 Loader/Engine/Scheduler/KVCache/Destroy 的内部实现。

**当前状态**：Runtime 已完成 I0 接口冻结（v0.3），`Runtime/backend/backend.h`、
`Runtime/backend/kernel.h`、`Runtime/backend/kernel_types.h` 已存在；本模块尚无 `.cc` 实现、测试报告与
skill。本文件完成“设计”四件套中的设计文档部分，模块整体在 P1–P5 完成后才可宣称完成。

---

## 1. 背景与目标

### 1.1 背景

- Runtime 架构已冻结为 `Engine → IExecutor → Backend` 的所有权模型：Backend 由 Loader 创建、
  移交给 Executor 持有；OpEnv 由 Loader 通过 `OpEnvFactory` 创建并挂在 `Op` 上；Executor 只按
  `Graph::GetOrderlyOPList()` 调用 `OpEnv::Forward()`。
- 首个模型包 `Models/Qwen3/artifacts/qwen3_0_6b/` 的图为：2 个输入、310 个常量、312 个计算 op、
  312 个 activation、633 行 IR、FP16 权重约 1.19 GB；计算 op 只有 8 种融合类型：
  `gather`、`rms_norm`、`linear`、`linear_swi_glu`、`linear_per_head_rmsnorm_rope`、
  `mega_attention`、`add`、`sample`。
- 现有 `backend.h` 已给出 Backend/Buffer/Capability/ParamDesc 契约，`kernel.h` 已给出
  `KernelDesc`/`Kernel::Execute` 契约；但尚缺：
  1. `KernelContext`/`TensorView` 的具体定义；
  2. Backend 注册表与目录级插件落地；
  3. 每个融合算子的模块化设计、四件套与单测；
  4. 算子级“Python 组图 → 工具链 → Runtime → 与 torch CPU 参考对比”的精度测试方法。
- `kernel_types.h` 中的 `CopyKind`、`BlockStage`、`RopeLayout` 为历史遗留枚举：`BlockStage` 引用的
  “融合骨架”已不存在，`CopyKind` 可由 `kInput/kOutput` 推导，`RopeLayout` 在 IR 中是字符串 attr
  `"neox"`。本设计将其列为 P0 清理项。

### 1.2 目标

| 编号 | 目标 |
| --- | --- |
| G-BE-01 | 冻结 Backend/Kernel 的功能边界、调用关系与生命周期，完全对齐 `Runtime_design.md` v0.3 |
| G-BE-02 | 定义 `KernelContext`/`TensorView`/`StepContext`，作为 OpEnv 与 Kernel 之间的唯一执行 ABI |
| G-BE-03 | 定义 Backend 插件（`ocl`/`mock`/`template`）的目录、注册、能力声明与构建方式 |
| G-BE-04 | 把 8 个融合算子各自定义为一个模块，规定 `.doc/`、代码、`tests/`、`.skill/` 四件套 |
| G-BE-05 | 定义算子单测流水线：Python 定义融合算子图与 torch 散算子参考 → Optimizer → Serializer → Loader → OpEnv::Forward → Destroy → 精度对比 |
| G-BE-06 | 明确测试流水线中可省略与不可省略的环节，给出快速路径与验收路径的边界 |
| G-BE-07 | 定义错误映射、内存/带宽约束、日志与可观测性，保证加载期失败可定位到 op/常量 |
| G-BE-08 | 给出 P0–P5 实施步骤、依赖、风险与测试通过标准 |

### 1.3 非目标

- 不实现 Executor 的 Packing/plan 适配细节（见 `Runtime/executor` 自身设计文档；本文件只定义
  Executor 与 OpEnv/Kernel 的交接 ABI）。
- 不实现 Loader/Engine/Scheduler/KVCache/Destroy 的内部算法（RuntimeCore 负责）。
- 不定义 Qwen3 之外的新融合算子；新增算子按第 4.4.4 节流程扩展。
- 不做量化 Kernel、投机解码、异步多队列、HiCache（只保留扩展点）。
- 不在本文件展开 OCL 具体 Kernel 的调优参数与真机微基准（见 `ocl_design.md` 与各算子文档）。
- 不修改 Graph/ToolChain 的生产代码；测试所需的合成图构造与 torch 参考放在测试侧。

---

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收要点 | 追溯 |
| --- | --- | --- | --- |
| FR-BE-01 | Backend 实现 `Init/Deinit/Capability/CreateBuffer/DestroyBuffer/ConvertParam/Upload/Download/CreateKernel/DestroyKernel/Synchronize` 全部接口 | 签名与 `backend.h` 完全一致；`Init` 全有或全无；`Deinit` 幂等 | TC-BE-01 |
| FR-BE-02 | 提供 Backend 注册表；后端以目录级插件接入，名称 `"ocl"`、`"mock"` | 未知后端返回 `kUnsupportedBackend`；新增后端只加目录 | TC-BE-02 |
| FR-BE-03 | 定义 `KernelContext`/`TensorView`，由 OpEnv 组装、Kernel 消费 | Kernel 只从 `KernelContext` 取指针/形状；不保存其指针到下一次调用 | TC-BE-03 |
| FR-BE-04 | 提供通用 `KernelOpEnv` 与 `IoOpEnv`，由 `OpEnvFactory` 统一创建 | Input/Output 走 `IoOpEnv`；计算 op 走 `KernelOpEnv`；缺失 op 返回 `kUnsupportedOp` | TC-BE-03 |
| FR-BE-05 | 8 个融合算子各自成为模块，具备 `.doc/`、代码、`tests/`、`.skill/` 四件套 | 每个模块文档齐全；测试报告含实际命令与输出 | TC-BE-04 |
| FR-BE-06 | 每个融合算子提供算子级精度单测：Python 融合算子图 → Optimizer → Serializer → Loader → OpEnv::Forward → Destroy → 与 torch CPU 散算子参考对比 | 每算子至少 3 组用例；精度在阈值内；Hash/IR 校验通过；无泄漏 | TC-BE-05 |
| FR-BE-07 | 能力校验：dtype、op、H2D/D2H、单缓冲上限、最小对齐、KV 块大小 | 加载期失败并定位；禁止运行期静默降级 | TC-BE-06 |
| FR-BE-08 | 权重转换/上传：`ConvertParam` 完成 dtype 转换、后端私有打包与 Buffer 创建 | FP16 直通；不支持 dtype 返回 `kUnsupportedDtype`；上传后主机源可释放 | TC-BE-07 |
| FR-BE-09 | Buffer 地址语义：设备侧 `Storage.data_ptr` 为 `Buffer*`，`offset` 为字节偏移；Kernel 不依赖主机指针 | Arena/权重/I/O/KV 四类缓冲均可被 Kernel 正确寻址 | TC-BE-03、TC-BE-05 |
| FR-BE-10 | OCL 后端：动态加载 OpenCL、设备/上下文/队列、Buffer、传输、Program 缓存、Kernel 创建与同步 | 公共头不出现 `cl_*`；主机无设备时返回明确错误；加载失败可定位 | TC-BE-08 |
| FR-BE-11 | Mock 后端：主机内存 + CPU 参考 Kernel，支持全部 8 个算子与 H2D/D2H | 无 GPU 的主机可完成全部算子单测与契约测试 | TC-BE-09 |
| FR-BE-12 | Template 后端：新后端骨架目录、一个模板 Kernel、自注册与 CMake | 复制目录即可参与构建；上层零改动 | TC-BE-02 |
| FR-BE-13 | 错误映射：设备/编译/缓冲/OOM/不支持算子映射到 `Runtime/common/status.h` 错误码；消息带 backend/stage/op | 错误码可断言；日志可定位 | TC-BE-10 |
| FR-BE-14 | 销毁与回滚：Kernel/Buffer 释放顺序正确；Backend 比 OpEnv/Kernel 活得更久 | 100 次 Load/Destroy 无句柄泄漏；重复 Deinit 安全 | TC-BE-11 |
| FR-BE-15 | 可观测性：缓冲字节统计、Kernel 编译/执行失败信息、Per-op 测试指标（max_abs、relative_l2、cosine） | 测试报告可机器解析 | TC-BE-05 |

### 2.2 非功能需求

| 编号 | 类别 | 要求 |
| --- | --- | --- |
| NFR-BE-01 | 性能 | 量化参数与优化空间由 OCL 设计文档定义；本模块保证每 Step 只一次 `Synchronize`，权重只上传一次 |
| NFR-BE-02 | 内存 | Buffer 数量与峰值可统计；arena 由 Loader 按 MemoryPlan 一次分配；不得在 Kernel 内部分配设备内存 |
| NFR-BE-03 | 可移植 | 公共头文件零厂商类型；新后端只加 `Runtime/backend/<name>/` |
| NFR-BE-04 | 可测试 | 主机无 GPU 可用 mock 后端完成全部 8 算子单测；OCL 无设备时跳过并记录 |
| NFR-BE-05 | 健壮性 | 非法 attr/形状/精度在加载期失败；Kernel 执行失败不崩溃；OOM 返回 `kDeviceOom` |
| NFR-BE-06 | 工程 | Google C++ Style；模块/类/函数中文注释；四件套完整；测试结果不编造 |

### 2.3 约束条件

| 编号 | 约束 |
| --- | --- |
| C-BE-01 | C++17；运行时不链接 LibTorch；Python 只用于工具链与测试 |
| C-BE-02 | Graph 是唯一可执行 IR；Backend/Kernel 不得修改 Graph 拓扑、attr 语义或 Storage 布局 |
| C-BE-03 | Backend 由 Executor 唯一持有；Loader/OpEnv/KVCache 只借用；Backend 必须比所有 OpEnv/Kernel 活得更久 |
| C-BE-04 | 公共头 `backend.h`/`kernel.h` 不出现 OpenCL/Vulkan/CUDA 类型；厂商头只允许出现在 `Runtime/backend/<name>/` 内部 |
| C-BE-05 | Backend 由推理线程串行调用（Engine 保证单推理线程），不要求内部线程安全；`Execute` 只在推理线程调用 |
| C-BE-06 | `Kernel` 公共接口只有 `Execute(const runtime_graph::KernelContext&)`；加载期准备在 `CreateKernel` 内部完成，不扩展公共契约 |
| C-BE-07 | `KernelDesc.op_type`、`BackendCapability.supported_ops` 统一使用 `runtime_graph::OpType`；图侧精度/布局/设备使用 `runtime_graph::{Precision,Layout,DeviceType}` |
| C-BE-08 | 跨终端接口变更必须先改 `Runtime/.doc/Runtime_design.md` 与附录 B，双方确认后再改代码 |

### 2.4 验收标准

| 编号 | 通过条件 |
| --- | --- |
| AC-BE-01 | 全部公共头可被 C++17 编译器 `-fsyntax-only` 通过；公共头不 include 厂商头 |
| AC-BE-02 | mock 后端通过全部契约测试；8 个算子模块的算子精度测试通过 |
| AC-BE-03 | 每个算子模块四件套齐全：设计文档、代码、`tests/test_results.md`、`.skill/SKILL.md` |
| AC-BE-04 | 算子测试包由 Optimizer + Serializer 产出，经 Loader 加载并通过 IR 哈希；测试结束经 Destroy 清理 |
| AC-BE-05 | 误差阈值：见 6.3.6；`sample` 的 argmax Token 100% 一致 |
| AC-BE-06 | 100 次 Load/Destroy 无 Buffer/Kernel 句柄泄漏；重复 `Deinit` 安全 |
| AC-BE-07 | OCL 后端在参考机上跑通 8 个算子测试；主机无设备时明确跳过并记录风险 |

---

## 3. 概要设计

### 3.1 模块边界与分层

```text
Engine
 └─ IExecutor（ExecBackend）
      ├─ 持有 Backend（unique_ptr，Loader 移交）
      ├─ 逐 op 调用 Op::GetOpEnv()->Forward()
      └─ 每步一次 Backend::Synchronize

OpEnv（Graph 侧，Op 持有）
 ├─ KernelOpEnv：读 Op.attr/Tensor.Storage → 组装 KernelContext → Kernel::Execute
 ├─ IoOpEnv：Input 做 H2D，Output 做 D2H（经 Backend::Upload/Download）
 └─ ExecOpEnv：接收 Executor 注入的 StepContext（本步 plan/KV/seq 元数据）

Backend（本模块，Executor 持有）
 ├─ 设备/上下文/队列
 ├─ Buffer 创建/销毁、H2D/D2H
 ├─ ConvertParam 权重转换/打包/上传
 ├─ CreateKernel/DestroyKernel
 └─ Synchronize

BackendRegistry ──▶ ocl / mock / template（目录级插件）
Kernel 实现     ──▶ backend/<name>/kernels/
算子语义模块    ──▶ backend/ops/<op>/（attr/形状校验、KernelDesc、文档与测试）
```

职责边界：

| 本模块负责 | 本模块不负责 |
| --- | --- |
| Backend 设备/上下文/队列/缓冲/传输/权重转换/Kernel 工厂/同步/能力 | 模型包解析、图重建、IR 哈希（Loader） |
| KernelContext/TensorView/StepContext ABI；OpEnv 基类与通用实现 | 调度、组批、KV 分配、停止判定 |
| 8 个算子的 attr/形状/精度契约与 KernelDesc 映射 | 图优化、精度/形状/显存推导（工具链） |
| 每算子四件套与算子级精度测试方法 | Engine 生命周期、线程模型 |
| 后端插件注册与能力声明 | Service/Tokenizer |

### 3.2 目录与子模块划分

```text
Runtime/backend/
├── .doc/backend_design.md          # 本文件（Backend 模块权威设计）
├── .skill/SKILL.md                 # 后端框架开发 skill（P5 落地）
├── tests/test_results.md           # 后端契约测试结果（P1/P4 落地）
├── backend.h                       # Backend/Buffer/BufferDesc/Capability/ParamDesc（已冻结）
├── backend_registry.h/.cc          # 后端注册表 + 自注册宏（P0 新增）
├── kernel.h                        # Kernel/KernelDesc + KernelContext 前置声明（已冻结）
├── kernel_context.h                # KernelContext/TensorView/AttentionContext（P0 新增）
├── exec_op_env.h/.cc               # ExecOpEnv/StepContext/SetStepContext（P0 新增）
├── kernel_types.h/.cc              # 后端边界类型与 DataType↔Precision 映射（P0 清理）
├── ops/                            # 算子语义模块（每个算子一个子目录）
│   ├── attr_utils.h/.cc            # attr 严格读取工具
│   ├── op_spec.h/.cc               # OpSpec 接口与 OpType→Spec 注册
│   ├── kernel_op_env.h/.cc         # 通用计算 OpEnv
│   ├── io_op_env.h/.cc             # Input/Output OpEnv
│   ├── op_env_factory.cc           # CreateDefaultOpEnvFactory 实现
│   ├── tests/                      # 共享测试驱动（Python 组图 + C++ op_runner）
│   ├── gather/
│   │   ├── .doc/gather_design.md
│   │   ├── .skill/SKILL.md
│   │   ├── tests/test_results.md
│   │   ├── gather_spec.h/.cc
│   │   └── tests/test_gather.py
│   ├── rms_norm/ …                 # 同上
│   ├── linear/ …
│   ├── linear_swi_glu/ …
│   ├── linear_per_head_rmsnorm_rope/ …
│   ├── mega_attention/ …
│   ├── add/ …
│   └── sample/ …
├── mock/
│   ├── mock_backend.h/.cc
│   ├── register_mock.cc
│   ├── CMakeLists.txt
│   └── kernels/                    # 8 个 CPU 参考 Kernel
├── ocl/
│   ├── .doc/ocl_design.md
│   ├── .skill/SKILL.md
│   ├── tests/test_results.md
│   ├── ocl_backend.h/.cc
│   ├── ocl_context.h/.cc
│   ├── ocl_buffer.h/.cc
│   ├── ocl_program_cache.h/.cc
│   ├── ocl_kernel.h/.cc
│   ├── register_ocl.cc
│   ├── CMakeLists.txt
│   └── kernels/                    # 8 个 OCL Kernel 与 .cl 源
└── template/
    ├── template_backend.h/.cc
    ├── register_template.cc
    ├── kernels/template_kernel.cc
    └── CMakeLists.txt
```

说明：

- `backend/` 根目录只放公共契约与框架；`ops/` 放算子语义与测试；`mock/`、`ocl/` 放后端实现与 Kernel。
- 每个算子模块的“代码”包括 `ops/<op>/` 的语义代码与测试；设备实现分别在
  `mock/kernels/`、`ocl/kernels/`，由对应后端文档描述。这样既满足“每个融合算子是一个模块”的四件套，
  又保持“后端目录自包含全部 Kernel”的插件规范。
- 每个算子模块的 `.doc/` 与 `.skill/` 只描述该算子的契约、参考实现与测试方法，不重复后端框架内容。
- 为什么 Kernel 不放在 `ops/<op>/`：`.doc/MobileNet_design.md` 4.4.3 要求每个后端目录自包含全部
  Kernel；`ops/<op>/` 只负责语义、契约与测试。若后续希望完全按算子聚合，可在 CMake 中从
  `ops/<op>/<backend>/` 收集源文件，而不改变公共接口。

### 3.3 关键流程

**加载（Backend 视角）**

```text
Loader
  → BackendRegistry::Create(config.backend, backend_config) → Backend::Init
  → Backend::Capability(&cap)
  → 校验 dtype/op/H2D/D2H/min_alignment/max_buffer_bytes/attention_block_size
  → 逐常量 ConvertParam(host Storage bytes) → Buffer（登记释放）
  → 按 MemoryPlan CreateBuffer(arena/KV/workspace) + CreateBuffer(I/O)（登记释放）
  → 回填 Storage.data_ptr = Buffer*，Storage.offset = 计划偏移
  → GraphEnv → OpEnvFactory.Create(op, backend, graph_env) → OpEnv::Init
       → KernelOpEnv: attr 校验 + KernelDesc + Backend::CreateKernel
  → Op::SetOpEnv
```

**单步执行（Backend 视角）**

```text
Executor
  → 构造 StepContext（backend/stream/plan/kv_view/seq/workspace）
  → 对每个 ExecOpEnv 调用 SetStepContext(&ctx)
  → IoBinder::SetInput/SetOutput（Engine 把主机 Storage 写入 Input/Output op 的 attr）
  → for op in GetOrderlyOPList():
       Input   : IoOpEnv::Forward() → Backend::Upload
       Compute : KernelOpEnv::Forward() → Kernel::Execute(KernelContext)
       Output  : IoOpEnv::Forward() → Backend::Download
  → Backend::Synchronize
  → 清空 StepContext
```

**销毁（Backend 视角）**

```text
Destroy
  → Backend::Synchronize
  → 逐 op ClearOpEnv()：OpEnv 析构 → Backend::DestroyKernel(std::move(kernel))
  → ResourceStack.Clear()：逐 Buffer Backend::DestroyBuffer(std::move(buffer))
  → Graph::ClearGraphEnv()
  → Backend::Deinit()（最后由 Executor::Deinit 释放 Backend）
```

### 3.4 算子模块化与四件套

每个融合算子是一个子模块，目录为 `Runtime/backend/ops/<op>/`，四件套固定为：

| 序号 | 交付物 | 位置 | 要求 |
| --- | --- | --- | --- |
| 1 | 设计文档 | `.doc/<op>_design.md` | 六段式；含 attr/形状/数值/错误/测试设计 |
| 2 | 代码 | `ops/<op>/*.h/.cc` + 后端 `kernels/` | Google Style；中文注释；语义与文档一致 |
| 3 | 测试结果 | `tests/test_results.md` | 实际命令、输出、指标、结论、未覆盖项 |
| 4 | skill | `.skill/SKILL.md` | 适用场景、输入、步骤、输出、陷阱 |

算子清单与模块映射：

| OpType | kernel_key | 模块目录 | 参考 torch 散算子 |
| --- | --- | --- | --- |
| `kGather` | `gather` | `ops/gather/` | `weight[input_ids]` |
| `kRmsNorm` | `rms_norm` | `ops/rms_norm/` | `x*rsqrt(mean(x²)+eps)*w` |
| `kLinear` | `linear` | `ops/linear/` | `x @ w.T` |
| `kLinearSwiGlu` | `linear_swi_glu` | `ops/linear_swi_glu/` | `silu(x@gate.T) * (x@up.T)` |
| `kLinearPerHeadRMSNormRope` | `per_head_rmsnorm_rope` | `ops/linear_per_head_rmsnorm_rope/` | `linear + per-head RMSNorm + RoPE(neox)` |
| `kMegaAttention` | `mega_attention` | `ops/mega_attention/` | `GQA + causal softmax` |
| `kAdd` | `add` | `ops/add/` | `a + b` |
| `kSample` | `sample_argmax` | `ops/sample/` | `argmax(logits, -1)` |

### 3.5 算子精度测试流水线总览

```text
Python（每算子一个 test_<op>.py）
  1 定义 fused op 的 ModelGraph（attrs/shapes/constants/inputs/outputs）
  2 用 torch 生成 FP16 权重与输入（固定 seed）
  3 用 torch 散算子组合计算 CPU 参考输出
  4 Optimizer.run(model_graph, weights) → Serialize → model.mgf + weights.bin
  5 写 case.json + inputs/*.bin + expected/*.bin
C++ op_runner（共享）
  6 Loader.Load(package) → Graph/Backend/OpEnv/缓冲
  7 TestIoBinder 绑定主机输入/输出；SetStepContext
  8 Input OpEnv::Forward（H2D）→ 目标 OpEnv::Forward → Output OpEnv::Forward（D2H）
  9 Backend::Synchronize → 写出 actual/*.bin
 10 IDestroyer::Destroy（或测试本地等价清理）
Python
 11 读取 actual，计算 max_abs/mean_abs/relative_l2/cosine
 12 断言阈值；写 tests/test_results.md 与 JSON 指标
```

### 3.6 技术选型与取舍

| 编号 | 决策 | 选择 | 理由 |
| --- | --- | --- | --- |
| D-BE-01 | 首期后端顺序 | mock 先行，OCL 后置 | 主机无 OpenCL 头/设备；mock 先打通 OpEnv/Kernel ABI 与数值 |
| D-BE-02 | Kernel 公共接口 | 只保留 `Execute`；加载期准备放 `CreateKernel` 内部 | 对齐已冻结的 `kernel.h`；避免公共契约反复变更 |
| D-BE-03 | 设备地址表示 | `Storage.data_ptr = Buffer*`，`offset` 为字节偏移 | 非 SVM OpenCL 无法用裸地址表达 `cl_mem`；统一用句柄 + 偏移 |
| D-BE-04 | 动态数据通道 | Executor 通过 `ExecOpEnv::SetStepContext` 注入本步元数据 | `OpEnv::Forward()` 无参已冻结；不引入 Graph 变更 |
| D-BE-05 | OCL 加载方式 | 动态加载 `OpenCL.dll/libOpenCL.so` | 主机无 SDK 也能编译；Android 上按系统库加载 |
| D-BE-06 | Kernel 编译 | 源码编译 + 磁盘二进制缓存（键含源码/选项/驱动） | 包保持后端无关；避免每次启动重编译 |
| D-BE-07 | 数学选项 | 仅 `-cl-mad-enable`；禁止 fast-relaxed-math | LLM 精度不可控；FP16 存储 + FP32 累加 |
| D-BE-08 | 首期内存形态 | Buffer 优先，Image 路径后置 | LLM 以 GEMM/Attention 为主；Image 收益待真机验证 |
| D-BE-09 | 算子测试入口 | 合成 ModelGraph + RecordingGraph（可选）+ Optimizer/Serializer | 不依赖 HF 模型；仍复用真实工具链与模型包 |
| D-BE-10 | 算子测试执行 | Loader + OpEnv 直驱；跳过 Engine/Scheduler/Executor | 单算子无组批/KV 需求；聚焦 Backend+Kernel |

---

## 4. 详细设计

### 4.1 公共接口语义

#### 4.1.1 Backend 生命周期

```cpp
class Backend {
 public:
  virtual Status Init(const BackendConfig& config) = 0;
  virtual Status Deinit() = 0;
  virtual Status Capability(BackendCapability* capability) const = 0;
  virtual Status CreateBuffer(const BufferDesc& desc,
                              std::unique_ptr<Buffer>* buffer) = 0;
  virtual Status DestroyBuffer(std::unique_ptr<Buffer> buffer) = 0;
  virtual Status ConvertParam(const ParamDesc& desc,
                              std::unique_ptr<Buffer>* buffer) = 0;
  virtual Status Upload(void* device_ptr, const void* host_ptr,
                        uint64_t bytes, Stream* stream) = 0;
  virtual Status Download(void* host_ptr, const void* device_ptr,
                          uint64_t bytes, Stream* stream) = 0;
  virtual Status CreateKernel(const KernelDesc& desc,
                              std::unique_ptr<Kernel>* kernel) = 0;
  virtual Status DestroyKernel(std::unique_ptr<Kernel> kernel) = 0;
  virtual Status Synchronize(Stream* stream) = 0;
};
```

语义约定：

| 接口 | 约定 |
| --- | --- |
| `Init` | 全有或全无；失败时清理已创建的平台/设备/上下文/队列/缓存；重复 Init 返回 `kStateError` |
| `Deinit` | 幂等；重复调用返回 `kOk`；释放 Program 缓存、Kernel 残留、Buffer 残留（调试构建报告泄漏） |
| `Capability` | 未 Init 返回 `kStateError`；`supported_ops` 必须真实覆盖已注册 Kernel；`attention_block_size` 为 KV 块大小唯一来源 |
| `CreateBuffer` | `desc.bytes > max_buffer_bytes` 返回 `kBufferError`（或 `kDeviceOom`）；对齐不满足返回 `kBufferError`；不初始化内容 |
| `DestroyBuffer` | 允许空指针；重复释放由 `unique_ptr` 语义保证不会发生 |
| `ConvertParam` | 源主机内存只在调用期间有效；成功后 Buffer 常驻；不支持 dtype 返回 `kUnsupportedDtype` |
| `Upload/Download` | `device_ptr` 必须是 `Buffer::DevicePtr()` 返回值（或空流默认）；`bytes` 不得超过 Buffer 剩余空间；首期只支持 Buffer 内偏移 0 |
| `CreateKernel` | 按 `kernel_key` 选择实现；`op_type` 用于能力校验；失败不泄漏 Program/Kernel |
| `DestroyKernel` | 允许空指针；内部释放 `cl_kernel` 等句柄 |
| `Synchronize` | 等待该流全部任务完成；`stream=nullptr` 表示默认流 |

#### 4.1.2 Buffer 与地址语义

```cpp
class Buffer {
 public:
  virtual ~Buffer() = default;
  virtual void* DevicePtr() const = 0;  // 后端私有设备句柄/地址
  virtual uint64_t Bytes() const = 0;
};
```

地址约定（跨终端契约，见 4.11）：

| 场景 | `Storage.data_ptr`（`kDevice`） | `Storage.offset` | `Buffer::DevicePtr()` | 使用方 |
| --- | --- | --- | --- | --- |
| 权重常量 | 该常量专属 `Buffer*` | 0 | 后端私有句柄 | Kernel |
| Activation | arena `Buffer*` | MemoryPlan 字节偏移 | arena 句柄 | Kernel |
| 图输入/输出 | 专用 `Buffer*` | 0 | 专用句柄 | IoOpEnv/Kernel |
| KV Cache | `Buffer*`（经 `KvCacheView.layers[i].k_base/v_base`） | 由 slot 计算 | KV 块句柄 | Attention Kernel |
| Workspace | workspace `Buffer*` | 0 或计划偏移 | 句柄 | 需要 workspace 的 Kernel |

规则：

1. 上层（Loader/Executor/OpEnv）只把 `Storage.data_ptr` 当作不透明 `Buffer*`，不得解引用。
2. Kernel 通过 `static_cast<const <Backend>Buffer*>(view.data)` 取原生句柄；`view.offset` 为字节偏移。
3. `Backend::Upload/Download` 的 `device_ptr` 取 `buffer->DevicePtr()`，首期只支持偏移 0；I/O 缓冲与权重缓冲都是独立 Buffer，满足该限制。调试导出 `lm_head.out` 的非零偏移下载不在 v1 支持范围（见 4.12）。
4. Buffer 生命周期由 Engine 的 `ResourceStack` 管理；Kernel/OpEnv 只持有非拥有 `Buffer*`；销毁顺序保证 Buffer 晚于 OpEnv。
5. `BufferDesc.alignment` 由 Loader 取 MemoryPlan 对齐与 Capability.min_alignment 的校验结果；不满足时加载失败，不做静默重排。

#### 4.1.3 ConvertParam

```cpp
struct ParamDesc {
  std::string name;
  const void* host_ptr = nullptr;
  uint64_t bytes = 0;
  DataType source_dtype = DataType::kUnknown;
  DataType target_dtype = DataType::kUnknown;
};
```

处理流程：

1. 校验 `host_ptr`/`bytes` 非零、dtype 受支持；否则 `kInvalidArgument`/`kUnsupportedDtype`。
2. 若 `source_dtype != target_dtype`，在主机侧逐元素转换（首期 FP32/BF16→FP16），
   转换使用 round-to-nearest-even；溢出/NaN 按 IEEE 规则处理并在调试日志计数。
3. 按后端私有布局打包（例如 OCL 将 K 维补到 4 的倍数，启用 `vload4`），这一步不改变数学语义。
4. `CreateBuffer` + `Upload` 上传；返回常驻 Buffer。
5. 失败时释放已创建 Buffer，返回 `kBufferError`/`kDeviceError`。

约束：

- 工具链已把权重转成目标精度时，`source_dtype == target_dtype`，第 2 步直通，避免二次舍入。
- 打包布局只属于该后端；Kernel 必须与 `ConvertParam` 的打包规则一致；跨后端不可复用同一 Buffer。
- `ParamDesc` 不含 shape；形状/字节一致性由 Loader 在调用前校验。

#### 4.1.4 能力声明

```cpp
struct BackendCapability {
  std::vector<DataType> supported_dtypes;
  std::vector<runtime_graph::OpType> supported_ops;
  bool h2d = false;
  bool d2h = false;
  uint64_t max_buffer_bytes = 0;
  uint64_t min_alignment = 0;
  int32_t attention_block_size = 0;
  KvLayout kv_layout = KvLayout::kNhd;
};
```

首期建议值：

| 后端 | dtypes | ops | H2D/D2H | max_buffer_bytes | min_alignment | attention_block_size | kv_layout |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mock` | FP16/BF16/FP32 | 8 个融合 op + Input/Output | true/true | `SIZE_MAX/2` | 1 | 16 | `kNhd` |
| `ocl` | FP16 | 8 个融合 op + Input/Output | true/true | `CL_DEVICE_MAX_MEM_ALLOC_SIZE` | 256 | 16（可调） | `kNhd` |

`attention_block_size` 由 Attention Kernel 设计决定；KVCache 据此计算块大小与容量，Backend 与 KVCache
必须一致，否则 Loader/KVCache 初始化失败。

#### 4.1.5 注册表

```cpp
// Runtime/backend/backend_registry.h
using BackendFactory = std::function<Status(
    const BackendConfig&, std::unique_ptr<Backend>*)>;

class BackendRegistry {
 public:
  static BackendRegistry& Instance();
  Status Register(const std::string& name, BackendFactory factory);
  Status Create(const std::string& name, const BackendConfig& config,
                std::unique_ptr<Backend>* backend) const;
  std::vector<std::string> Names() const;
};

#define MOBILENET_REGISTER_BACKEND(name, factory) ...
```

- 注册发生在静态初始化阶段（单线程）；`Create` 在 Load 期调用。
- 名称固定为 `"ocl"`、`"mock"`、`"template"`；未知名称返回 `kUnsupportedBackend`。
- 后端以 object library 编入 `runtime_backend`，避免自注册对象被静态链接器裁剪。
- `Runtime/backend/CMakeLists.txt` 配置期遍历 `backend/*/CMakeLists.txt`，开关为
  `MOBILENET_BACKEND_<NAME>`；首期 `MOCK=ON`、`OCL` 在具备头文件/设备时 `ON`。

### 4.2 Kernel ABI

#### 4.2.1 KernelDesc

```cpp
struct KernelDesc {
  runtime_graph::OpType op_type = runtime_graph::OpType::kUnknown;
  DataType precision = DataType::kUnknown;
  std::string kernel_key;
};
```

- `op_type`：图节点类型，用于能力校验与日志；
- `precision`：本节点计算精度（首期 FP16）；
- `kernel_key`：后端实现键，见 3.4 的映射表；
- Backend 只按 `kernel_key` 分发，`op_type` 不参与实现选择，避免一个 op 多种实现时耦合。

#### 4.2.2 KernelContext 与 TensorView

`Runtime/backend/kernel.h` 已把 `KernelContext` 前置声明在 `runtime_graph` 命名空间，并注明“由
Executor 侧头文件定义（ExecBackend 维护）”。由于 OpEnv 也在 backend 侧组装该结构，为保持依赖方向
（OpEnv → Kernel，而不是 Kernel → Executor），具体定义放在 `Runtime/backend/kernel_context.h`；
Executor 直接包含该头文件。命名空间沿用已冻结的 `runtime_graph`，避免修改 `kernel.h` 公共契约。

```cpp
// Runtime/backend/kernel_context.h
namespace mobilenet {
namespace runtime_graph {

// 单张量的设备/主机视图。
struct TensorView {
  void* data = nullptr;        // kDevice: Buffer*；kHost: 主机地址
  uint64_t offset = 0;         // data 为 Buffer 时的字节偏移
  uint64_t bytes = 0;          // 有效字节数
  Shape shape;                 // 逻辑形状
  DataType dtype = DataType::kUnknown;   // 后端边界精度
  Layout layout = Layout::kRowMajor;     // runtime_graph::Layout
  DeviceType device = DeviceType::kUnknown;
};

// Attention 本步元数据；v1 只用 batch/token/seq，KV 字段为阶段二预留。
struct AttentionContext {
  int32_t batch = 0;
  int32_t token_count = 0;
  const int32_t* seq_lens = nullptr;      // 每 batch 行有效 token 数
  const int32_t* block_table = nullptr;   // 每请求物理块表（预留）
  int32_t block_table_stride = 0;
  const int64_t* slots = nullptr;         // KV 写入 slot（预留）
  int32_t slot_count = 0;
  int32_t kv_block_size = 0;
  const runtime::KvCacheView* kv = nullptr;  // 非拥有，可为空
  bool is_prefill = true;
};

// 单节点执行上下文。
struct KernelContext {
  const TensorView* inputs = nullptr;
  uint32_t input_count = 0;
  TensorView* outputs = nullptr;
  uint32_t output_count = 0;
  const AttrMap* attrs = nullptr;          // Op.attr，非拥有
  const AttentionContext* attention = nullptr;  // 可为空
  const runtime::SamplingParams* sampling = nullptr;  // 可为空
  TensorView workspace;                    // 无 workspace 时 data=nullptr
  Backend* backend = nullptr;              // 非拥有
  Stream* stream = nullptr;                // 可为空，表示默认流
  int32_t batch = 1;
  int32_t token_count = 0;                 // 本步有效 token 数
  int32_t seq_len = 0;                     // 逻辑序列长度（含 padding）
};

}  // namespace runtime_graph
}  // namespace mobilenet
```

规则：

1. `KernelContext` 由 OpEnv 在每次 `Forward()` 内构造于栈上，生命周期仅一次 `Execute`；
   Kernel 不得保存 `context` 指针或其内部指针到下一次调用。
2. `TensorView.data` 对设备张量是 `Buffer*`；Kernel 实现按后端类型转换；不得直接解引用。
3. `TensorView.bytes` 必须等于 `shape.GetShapeSize() * DataTypeSize(dtype)`；OpEnv 在 Init 时校验，
   不匹配返回 `kInvalidArgument`。
4. `attrs` 指向 `Op::Attrs()`，只读；Kernel 不得修改。
5. `attention`/`sampling` 只在对应 op 需要时非空；其余 op 为 `nullptr`。
6. `batch/token_count/seq_len` 由 StepContext 填充；单算子测试直接用测试值。

#### 4.2.3 StepContext 与动态数据注入

`OpEnv::Forward()` 无参，而 Attention/Sample 需要本步 plan/KV/seq 信息。为避免修改 Graph 与
Runtime 统一接口，定义 ExecBackend 内部的 `ExecOpEnv`：

```cpp
// Runtime/backend/exec_op_env.h
namespace mobilenet {

struct StepContext {
  Backend* backend = nullptr;
  Stream* stream = nullptr;
  const runtime::ExecutionPlan* plan = nullptr;      // 可为空
  const runtime::KvCacheView* kv_view = nullptr;     // 可为空
  const int32_t* seq_lens = nullptr;                 // 可为空
  int32_t batch = 1;
  int32_t packed_tokens = 0;
  TensorView workspace;                              // 可为空
  bool is_prefill = true;
};

class ExecOpEnv : public runtime_graph::OpEnv {
 public:
  explicit ExecOpEnv(runtime_graph::Op* op)
      : runtime_graph::OpEnv(op) {}
  virtual bool IsIoNode() const { return false; }
 protected:
  const StepContext* step_context_ = nullptr;
};

// Executor/测试驱动在每步开始对每个 op 调用一次。
Status SetStepContext(runtime_graph::OpEnv* env, const StepContext* context);

}  // namespace mobilenet
```

- `SetStepContext` 内部 `dynamic_cast<ExecOpEnv*>`；失败返回 `kInternal`。所有 OpEnv 均由本模块
  工厂创建，因此该转换在本模块内是封闭的。
- Executor 在 `Execute` 开始时设置、结束时清空；测试驱动同样。
- 该扩展不改变 `Graph::OpEnv` 与 `Runtime_design.md` 的公开契约；属于 ExecBackend 内部约定。

#### 4.2.4 Kernel 生命周期

公共接口：

```cpp
class Kernel {
 public:
  virtual ~Kernel() = default;
  virtual Status Execute(const runtime_graph::KernelContext& context) = 0;
};
```

加载期准备在 `Backend::CreateKernel` 内部完成，不扩展公共接口：

1. 按 `kernel_key` 查询后端 Kernel 表；缺失返回 `kUnsupportedOp`。
2. 创建具体 Kernel 对象；OCL 后端在此完成 Program 获取/编译、`cl_kernel` 创建、静态参数校验。
3. 首次 `Execute` 可完成一次性准备（如绑定权重 Buffer、计算 work size）；之后只绑定动态参数。
4. `OpEnv` 析构时经 `DestroyKernel(std::move(kernel))` 释放；Kernel 不持有 Buffer 所有权。

数值与内存规则：

- FP16 存储、FP32 累加；禁止 FP16 累加；输出只在写回时做一次 round-to-nearest-even。
- Kernel 不得在 `Execute` 中分配/释放设备内存；workspace 由 OpEnv 从 GraphEnv 取得。
- Kernel 不得调用 `clFinish`/`Backend::Synchronize`；同步由 Executor 每 Step 一次。

#### 4.2.5 OpEnv 框架

```cpp
// Runtime/backend/ops/kernel_op_env.h
class KernelOpEnv final : public ExecOpEnv {
 public:
  Status Init(runtime_graph::GraphEnv* graph_env) override;
  Status Forward() override;
  bool NeedTempData() const override;   // 来自 OpSpec
 private:
  const OpSpec* spec_ = nullptr;
  std::unique_ptr<Kernel> kernel_;
  std::vector<runtime_graph::TensorView> inputs_;
  std::vector<runtime_graph::TensorView> outputs_;
  runtime_graph::TensorView workspace_;
};
```

`KernelOpEnv` 由 `OpEnvFactory` 构造时注入非拥有 `Backend*`；`Init(GraphEnv*)` 使用该指针创建 Kernel。
`IoOpEnv` 同样在构造时注入 `Backend*`，用于 `Upload/Download`。

`Init` 流程：

1. 按 `op->Type()` 查找 `OpSpec`；缺失返回 `kUnsupportedOp`。
2. 严格读取 attr（类型缺失/不匹配 → `kInvalidArgument`，消息带 op 名）。
3. 校验输入/输出数量、形状、dtype、layout、`byte_size`；不匹配立即失败。
4. 由 `OpSpec::BuildKernelDesc` 得到 `KernelDesc`；`Backend::CreateKernel` 创建 Kernel。
5. 若 `NeedTempData()`，在 `GraphEnv` 登记 workspace shared storage（键 `"workspace.<op>"`），
   由 Loader 在分配后回填 `data_ptr/offset/byte_size`；首期 8 个算子均不需要 workspace，
   该路径只保留接口与测试桩。

`Forward` 流程：

1. 校验 `step_context_` 非空；未设置返回 `kStateError`。
2. 复制缓存的 `TensorView`（不修改缓存本身）。
3. 从 StepContext/Plan 填充 `AttentionContext`、`SamplingParams`、`batch/token_count/seq_len`。
4. 构造 `KernelContext`，调用 `kernel_->Execute(ctx)`；失败用 `PrefixStatus` 加 op 名。
5. 不 Save/Kernel 不得修改 TensorView。

`IoOpEnv`：

- Input：主机 Storage 来自 `attr["storage"]`（Engine 经 `IoBinder` 写入）；设备 Storage 来自输入 Tensor
  的 Storage（Loader 绑定）。调用 `Backend::Upload(buffer->DevicePtr(), host_ptr, bytes, stream)`。
- Output：设备 Storage 来自源 Tensor；主机 Storage 来自 `attr["storage"]`。调用
  `Backend::Download(host_ptr, buffer->DevicePtr(), bytes, stream)`。
- 校验名称、字节数、dtype、shape；失败返回 `kInvalidArgument` 并带 op 名。

### 4.3 算子模块规范

#### 4.3.1 OpSpec 接口

```cpp
// Runtime/backend/ops/op_spec.h
struct OpSpec {
  runtime_graph::OpType op_type;
  const char* kernel_key;
  bool need_workspace;
  uint64_t workspace_bytes;
  // 校验 attr 与输入输出形状/精度；失败返回带 op 名的 Status。
  Status (*validate)(const runtime_graph::Op& op);
  // 组装 KernelDesc；precision 取 op attr["precision"] 或输出 Storage。
  Status (*build_kernel_desc)(const runtime_graph::Op& op, KernelDesc* desc);
};

const OpSpec* FindOpSpec(runtime_graph::OpType op_type);
```

`op_registry.cc` 集中登记 8 个 spec；新增算子必须同步更新该表与 `BackendCapability.supported_ops`。
`attr_utils.h` 提供严格读取函数：

```cpp
Status GetIntAttr(const runtime_graph::Op& op, const char* name, int64_t* value);
Status GetFloatAttr(const runtime_graph::Op& op, const char* name, double* value);
Status GetBoolAttr(const runtime_graph::Op& op, const char* name, bool* value);
Status GetStringAttr(const runtime_graph::Op& op, const char* name, std::string* value);
```

缺失、类型不符、值为空均返回 `kInvalidArgument`，消息格式 `op=<name> attr=<name> ...`。

#### 4.3.2 算子契约汇总

测试默认形状（small）：`B=1, S=8, H=64, I=128, heads=4, kv_heads=2, D=16, V=128`。
真实形状冒烟测试按 Qwen3 配置（`S=256, H=1024, I=3072, heads=16, kv_heads=8, D=128, V=151936`），
标注 `slow`，默认不在 CI 跑。

| OpType | 输入 | 输出 | 必需 attr | 数值规则 | 参考实现 |
| --- | --- | --- | --- | --- | --- |
| `kGather` | ids `[1,S]` int64；W `[V,H]` fp16 | `[1,S,H]` fp16 | `axis=0` | `out[b,s,:]=W[id[b,s],:]`；id 越界 → `kKernelError` | `W[ids]` |
| `kRmsNorm` | x `[1,S,H]` fp16；w `[H]` fp16 | `[1,S,H]` | `axis=-1, eps, normalized_shape` | `y=x*rsqrt(mean(x²)+eps)*w`，FP32 归约 | 同式 |
| `kLinear` | x `[1,S,K]` fp16；W `[N,K]` fp16 | `[1,S,N]` | `in_features, out_features, transpose_weight=true` | `y=x·Wᵀ`，FP32 累加 | `x @ W.T` |
| `kLinearSwiGlu` | x `[1,S,H]`；gate/up `[I,H]` | `[1,S,I]` | `intermediate_size, activation=silu` | `silu(x·gateᵀ)*(x·upᵀ)` | 同式 |
| `kLinearPerHeadRMSNormRope` | x `[1,S,H]`；W `[N,H]`；norm `[D]`；pos `[1,S]` int64 | `[1,S,N]` | `num_heads, head_dim, norm_eps, rope_theta, rope_layout=neox, transpose_weight` | GEMM → reshape `[B,S,heads,D]` → per-head RMSNorm → RoPE(neox) | 同式（rotate_half） |
| `kMegaAttention` | q `[1,S,heads·D]`；k/v `[1,S,kv_heads·D]` | `[1,S,heads·D]` | `num_heads, num_kv_heads, head_dim, causal, softmax_scale, merge_heads, has_output_projection=false, use_kv_cache=false`；`gqa_ratio` 可选，若提供必须等于 `num_heads/num_kv_heads` | GQA repeat、causal mask、online softmax、FP32 累加 | `softmax(QKᵀ·scale+mask)V` |
| `kAdd` | a/b `[1,S,H]` | `[1,S,H]` | 无 | `a+b`，FP16 舍入 | `a+b` |
| `kSample` | logits `[1,S,V]` fp16 | `[1,S]` int64 | `method=argmax, axis=-1` | 首个最大值；并列取最小下标 | `argmax(logits,-1)` |

精度/布局约束：

- 所有浮点输入/输出 `row_major`、FP16；`input_ids/position_ids` int64；Sample 输出 int64。
- `normalized_shape` 必须等于最后一维；`num_heads % num_kv_heads == 0`；`out_features % head_dim == 0`。
- `transpose_weight=true` 表示权重布局 `[N,K]`；首期不支持 `bias=true`，遇到返回 `kUnsupportedOp`。
- 缺失必需 attr、属性类型不符、形状/字节不一致：加载期失败，禁止静默降级。

#### 4.3.3 每个算子模块的文件清单

```text
ops/<op>/
├── .doc/<op>_design.md
├── .skill/SKILL.md
├── tests/test_results.md
├── tests/test_<op>.py          # 组图、torch 参考、跑包、对比
├── <op>_spec.h                 # spec 声明（可选，简单算子可并入 op_registry）
└── <op>_spec.cc                # validate/build_kernel_desc 实现
```

后端实现：

```text
mock/kernels/<op>_kernel.cc     # CPU 参考 Kernel
ocl/kernels/<op>_kernel.cc      # OCL Kernel 封装
ocl/kernels/<op>.cl             # OCL 源（构建期嵌入）
```

#### 4.3.4 新增算子流程

1. 在 Graph 的 `runtime_graph::OpType` 增加枚举并更新 `OpTypeToStr`（ToolChain 评审）。
2. 在 `Runtime/backend/ops/` 新建算子目录与四件套骨架。
3. 实现 `OpSpec`（attr/形状/精度校验、KernelDesc）。
4. 在 `mock/kernels/` 与 `ocl/kernels/` 实现 Kernel；更新后端 `CreateKernel` 表与 Capability。
5. 在 `op_registry.cc` 登记 spec。
6. 编写 Python 算子测试与 torch 参考；按第 6.3 节跑通。
7. 更新本文档的算子契约表、`ocl_design.md` 与 Runtime 架构文档的 op 清单（如涉及冻结接口）。

### 4.4 Mock 后端

- **Buffer**：`posix_memalign`/`_aligned_malloc` 主机内存；`DevicePtr()` 返回主机地址；`Bytes()` 返回大小。
- **传输**：`Upload/Download` 为 `memcpy`；`Synchronize` 为空操作。
- **ConvertParam**：dtype 直通或主机侧转换；不做后端私有打包（或仅按 64 B 对齐）。
- **Kernel**：CPU 参考实现，内部 FP32 计算，写回时 FP16 舍入；单线程、确定性。
- **Capability**：8 op、FP16/BF16/FP32、H2D/D2H、`max_buffer_bytes=SIZE_MAX/2`、`min_alignment=1`、
  `attention_block_size=16`、`kNhd`。
- **用途**：主机 CI、OpEnv/Kernel ABI 验证、8 算子精度测试、OCL 数值的对照基线。
- **限制**：不作为产品配置；性能不设目标。

### 4.5 OCL 后端（摘要）

详细设计见 `Runtime/backend/ocl/.doc/ocl_design.md`。本文件冻结以下原则：

1. **动态加载**：`OpenCL.dll`/`libOpenCL.so` 通过 `LoadLibrary/dlopen` 加载，函数指针表由本后端维护；
   公共头不出现 `cl_*`。构建期可切换为直接链接（`MOBILENET_OCL_STATIC`）。
2. **设备与队列**：创建 Platform/Device/Context/Queue；`CL_DEVICE_MAX_MEM_ALLOC_SIZE` 映射
   `max_buffer_bytes`；`CL_DEVICE_VENDOR` 用于选择调优参数（不改变语义）。
3. **Buffer**：`clCreateBuffer(CL_MEM_READ_WRITE)`；`DevicePtr()` 返回 `cl_mem` 的不透明值；
   `offset` 由 Kernel 侧以字节偏移参与地址计算。
4. **传输**：`clEnqueueWriteBuffer/ReadBuffer`；首期同步语义（阻塞调用）；`Synchronize` 用 `clFinish`。
5. **Program 缓存**：每个 `kernel_key` 在首次 `CreateKernel` 时编译；源码以构建期生成的 C++ 字符串
   嵌入；编译选项只含 `-cl-mad-enable`（按需加 `-DUSE_HALF`）；二进制缓存键 =
   `SHA-256(源码 + 选项 + 驱动版本 + 设备名)`；缓存目录来自 `BackendConfig.options["cache_dir"]`，
   缺省不落盘。
6. **Kernel**：`clCreateKernel`；静态参数（权重 Buffer、形状、常量）在首次 `Execute` 绑定；动态参数
   （输入/输出 Buffer、offset、seq_len）每次绑定；`clEnqueueNDRangeKernel`；不在 Kernel 内 `clFinish`。
7. **错误映射**：
   - `CL_OUT_OF_RESOURCES`/`CL_MEM_OBJECT_ALLOCATION_FAILURE` → `kDeviceOom`；
   - `CL_BUILD_PROGRAM_FAILURE` → `kKernelError`，消息附 build log 文件路径；
   - `CL_DEVICE_NOT_AVAILABLE` → `kDeviceError`；
   - `CL_INVALID_*` → `kInvalidArgument`/`kKernelError`；
   - 所有错误消息带 backend/stage/kernel_key/原生错误码。
8. **数值**：FP16 存储、FP32 累加；禁止 `-cl-fast-relaxed-math`；`mad` 只允许在 FP32 累加语义下使用。

### 4.6 构建与注册

- `Runtime/CMakeLists.txt` 由 RuntimeCore 维护，链接 Graph 静态库（`GRAPH_BUILD_PYTHON=OFF`）。
- `Runtime/backend/CMakeLists.txt` 由本模块提供：
  1. 公共源码（`backend_registry.cc`、`ops/*.cc`、`exec_op_env.cc`、`kernel_types.cc`）编成
     `runtime_backend` 静态库；
  2. 配置期遍历 `backend/*/CMakeLists.txt`，按 `MOBILENET_BACKEND_<NAME>` 加入 object library；
  3. `runtime_backend` 以 `$<TARGET_OBJECTS:...>` 合并后端对象，防止自注册被裁剪；
  4. 测试 target：`backend_contract_test`、`op_runner`、各算子 Python 测试。
- OCL `.cl` 源：构建期由脚本生成 `ocl_kernel_sources.cc`（raw string 常量），避免运行时文件 IO；
  生成的源码哈希写入 Program 缓存键。
- 公共头检查：`Runtime/backend/tests/header_check.cc` 包含全部公共头，CI 执行 `-fsyntax-only`。

### 4.7 内存与性能设计

| 项目 | 值/策略 |
| --- | --- |
| 权重 Buffer | 每常量一个 Buffer；Qwen3 最大 `embedding.weight` 311,164,928 B，须 ≤ `max_buffer_bytes` |
| Activation | 单个 arena Buffer，78,317,568 B；tensor 用 `offset` 寻址 |
| I/O Buffer | `input_ids` 2 KB、`position_ids` 2 KB、`sample.out` 2 KB |
| KV Buffer | 由 IKVCache 按 `attention_block_size` 与层数分配；Backend 只提供分配/释放 |
| 传输 | 权重一次上传（1.19 GB）；每 Step I/O 约 6 KB；调试 logits 延后 |
| Kernel 数量 | 每 Step 312 个计算 op；每 op 一次 `Execute`；每 Step 一次 `Synchronize` |
| Program | 8 个 kernel_key 对应 8 套 Program；首次加载编译，之后走二进制缓存 |
| 对齐 | MemoryPlan 256 B；`Capability.min_alignment` 校验，不满足加载失败 |

性能优化只允许在不改变数值契约的前提下进行（向量化、workgroup 调优、权重打包、local memory）；
任何改变累加精度或数学语义的优化必须单独立项并重新验收。

### 4.8 异常与错误处理

| 场景 | 错误码 | 处理 |
| --- | --- | --- |
| 后端未注册 | `kUnsupportedBackend` | Loader 拒绝加载 |
| 设备/上下文/队列失败 | `kBackendInitError`/`kDeviceError` | `Init` 清理内部资源后返回 |
| 不支持 dtype | `kUnsupportedDtype` | Capability 校验或 `ConvertParam` 返回 |
| 不支持 op/kernel_key | `kUnsupportedOp` | 加载期定位到 op 名 |
| Buffer 超限/对齐不满足 | `kBufferError` | 加载期定位到 BufferDesc/常量名 |
| OOM | `kDeviceOom` | 加载期返回；不进入推理 |
| Kernel 编译/执行失败 | `kKernelError` | 消息带 kernel_key/op 名/build log |
| H2D/D2H 失败 | `kBufferError`/`kDeviceError` | 本步失败；设备级错误上报 Engine |
| 非法 attr/形状 | `kInvalidArgument` | OpEnv::Init 返回，带 op 名与 attr 名 |
| 重复 Deinit | `kOk` | 幂等 |
| StepContext 未设置 | `kStateError` | OpEnv::Forward 返回，提示 Executor 缺陷 |

错误消息统一格式：

```text
[backend=ocl][stage=CreateKernel][op=layer.0.attn][kernel_key=mega_attention] native=-54: ...
```

### 4.9 与其他模块的契约

| 编号 | 契约 | 对方 |
| --- | --- | --- |
| BE-C-01 | 设备侧 `Storage.data_ptr` 为 `Buffer*`，`offset` 为字节偏移；Loader 分配后回填 | RuntimeCore/Loader |
| BE-C-02 | Loader 在绑定 Input storage 时必须同时写 Input op 的 `attr["storage"]` 与输入 Tensor 的 Storage（或保证计算 op 能经 Tensor 读到设备 Buffer） | RuntimeCore/Loader |
| BE-C-03 | `KvLayerView.k_base/v_base` 为 `Buffer*`；slot/块步长由 KVCache 与 Attention Kernel 约定 | RuntimeCore/KVCache |
| BE-C-04 | Loader 使用 `BackendRegistry::Create`、`Backend::Init/Capability/ConvertParam/CreateBuffer`；`OpEnvFactory` 由 `CreateDefaultOpEnvFactory()` 提供 | RuntimeCore |
| BE-C-05 | Executor 每步对每个 `ExecOpEnv` 调用 `SetStepContext`，并在步末清空；每步只 `Synchronize` 一次 | RuntimeCore/Executor（ExecBackend 实现） |
| BE-C-06 | 销毁顺序：Synchronize → ClearOpEnv（DestroyKernel）→ ResourceStack（DestroyBuffer）→ ClearGraphEnv → Deinit | RuntimeCore/Destroy |
| BE-C-07 | `kernel_types.h` 的 `CopyKind/BlockStage/RopeLayout` 在 P0 删除；`Runtime_design.md` 附录 A 同步 | RuntimeCore |
| BE-C-08 | 合成 ModelGraph 与 torch 参考由测试侧构造，不进入生产工具链；可复用 `ToolChain.testing` 的 `make_tensor/write_safetensors` | ToolChain |
| BE-C-09 | Graph 提供 C++ `DumpIr` 供 Loader 校验 IR 哈希；Kernel 不依赖 Graph 执行接口 | ToolChain |
| BE-C-10 | `DataType` 与 `runtime_graph::Precision` 的互转由本模块 `kernel_types.h` 提供 | RuntimeCore |

### 4.10 已知限制与待确认项

| 编号 | 限制/待确认 | 影响 | 计划 |
| --- | --- | --- | --- |
| L-BE-01 | `Backend::Upload/Download` 无 offset 参数 | 调试导出 arena 中 `lm_head.out` 的末行 logits 暂不可用 | v1 只导出 Sample 输出；调试导出另立契约 |
| L-BE-02 | OCL 主机编译需要 Khronos `CL/cl.h` | 本机只有 `OpenCL.dll`，无头文件 | P0 决策：vendor 头或仅 Android 构建 |
| L-BE-03 | 无参考机 | OCL 数值只能在有设备时验收 | mock 先覆盖全部算子；OCL 用例标注 skip |
| L-BE-04 | `KernelContext` 命名空间被冻结在 `runtime_graph` | 语义上属于 runtime，但类型名固定 | 保持不动；在文档中说明 |
| L-BE-05 | `kernel_types.h` 遗留枚举 | 可能误导实现者 | P0 删除并同步 Runtime 文档 |
| L-BE-06 | Loader/Destroyer 尚未实现（RuntimeCore R1/R3） | 算子测试的 Loader 路径依赖对方 | 先用 mock + 测试本地清理打通 Kernel；Loader 就绪后切验收路径 |
| L-BE-07 | Image 路径、量化、多队列未实现 | 后续性能/量化需求 | 预留 `BufferUsage`/`Capability`/源码结构，不提前实现 |

---

## 5. 实施步骤

### 5.1 阶段划分

| 阶段 | 内容 | 主要交付 | 依赖 |
| --- | --- | --- | --- |
| P0 契约与骨架 | 新增 `kernel_context.h`/`backend_registry.h`/`exec_op_env.h`；清理 `kernel_types`；CMake 骨架；公共头编译检查 | 设计文档 + 头文件 + 构建骨架 | 本文件；Runtime_design v0.3 |
| P1 Mock 与框架 | mock 后端、通用 `KernelOpEnv`/`IoOpEnv`、OpSpec 注册、8 个 CPU Kernel、后端契约测试 | 框架代码 + `tests/test_results.md` | P0 |
| P2 Python 算子图与包 | 每算子合成 ModelGraph、torch 散算子参考、Optimizer/Serializer 打包、case 数据 | 8 套 Python 测试与包生成器 | P1；ToolChain API |
| P3 Runtime 单算子执行 | `op_runner`、TestIoBinder、Loader 集成、Destroy；mock 上 8 算子精度测试 | 8 份算子测试结果 | P2；RuntimeCore Loader/Destroyer |
| P4 OCL 后端 | OCL 封装、传输、Program 缓存、8 个 Kernel、契约与精度测试 | OCL 代码 + 真机测试结果 | P1；OpenCL 头/参考机 |
| P5 四件套收口 | 8 个算子 `.doc`/`.skill`/测试报告；后端框架 skill；文档一致性审计 | 完整四件套 | P1–P4 |

### 5.2 任务拆解

| 编号 | 任务 | 产出 | 依赖 |
| --- | --- | --- | --- |
| BE-T01 | 公共头与 ABI：`kernel_context.h`、`exec_op_env.h`、`kernel_types` 清理 | 头文件、编译检查 | BE-T00（本文件） |
| BE-T02 | 注册表与 CMake：`backend_registry`、目录自动发现、object library | 可构建的 `runtime_backend` | BE-T01 |
| BE-T03 | Mock 后端：Buffer/传输/ConvertParam/Capability | mock 后端与契约测试 | BE-T02 |
| BE-T04 | OpEnv 框架：OpSpec、KernelOpEnv、IoOpEnv、工厂 | `CreateDefaultOpEnvFactory` | BE-T01 |
| BE-T05 | 8 个 CPU Kernel 与算子 spec | mock 算子实现 | BE-T03、BE-T04 |
| BE-T06 | Python 组图与参考：每算子 ModelGraph + torch 参考 + 权重/输入生成 | 8 套测试脚本 | BE-T05、ToolChain |
| BE-T07 | 包生成与 case 数据：Optimizer/Serializer/权重/inputs/expected | 可加载测试包 | BE-T06 |
| BE-T08 | `op_runner` + TestIoBinder + Loader/Destroy 集成 | 单算子执行器 | BE-T07、RuntimeCore |
| BE-T09 | mock 上 8 算子精度测试与报告 | 8 份 `test_results.md` | BE-T08 |
| BE-T10 | OCL 后端基础设施 | OCL 封装与契约测试 | BE-T02、OpenCL 头 |
| BE-T11 | 8 个 OCL Kernel 与真机测试 | OCL 算子实现与报告 | BE-T10、参考机 |
| BE-T12 | 每算子文档/skill/审计 | 四件套 | BE-T09、BE-T11 |

### 5.3 依赖关系

```text
BE-T00 ─▶ BE-T01 ─▶ BE-T02 ─┬─▶ BE-T03 ─▶ BE-T05 ─┐
                            │                      ├─▶ BE-T06 ─▶ BE-T07 ─▶ BE-T08 ─▶ BE-T09 ─▶ BE-T12
                            └─▶ BE-T04 ────────────┘
BE-T02 ─▶ BE-T10 ─▶ BE-T11 ───────────────────────────────────────────────▶ BE-T12
RuntimeCore Loader/Destroyer（R1/R3）──────────────────────▶ BE-T08
ToolChain testing API / 权重读写 ─────────────────────────▶ BE-T06
```

### 5.4 里程碑与风险应对

| 里程碑 | 完成标志 |
| --- | --- |
| M-B1 | 公共头编译通过；mock 后端契约测试通过 |
| M-B2 | 8 个算子在 mock 上通过算子精度测试；每算子测试包可重复生成 |
| M-B3 | OCL 在参考机上通过 8 个算子精度测试与契约测试 |
| M-B4 | 每算子四件套齐全；文档与代码一致；审计通过 |

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| Loader/Destroyer 延期 | BE-T08 阻塞 | 先用测试本地 Graph+Buffer 路径验证 Kernel；Loader 就绪后补验收 |
| OpenCL 头/设备缺失 | OCL 无法编译/验收 | vendor Khronos 头或仅 Android；mock 覆盖全部算子；记录风险 |
| `Storage.data_ptr` 语义与 RuntimeCore 不一致 | Kernel 寻址错误 | P0 冻结 BE-C-01/02/03；Loader 单测断言地址语义 |
| 算子测试包生成慢/占空间 | CI 时间/磁盘 | small 形状默认；真实形状标 `slow`；包目录复用与清理 |
| 阈值过松/过紧 | 漏检或误报 | 初始阈值 + 实测锁定；每算子文档记录依据 |
| 真实形状内存大 | 测试 OOM | small 为主；真实形状只在有足够内存时跑 |
| 工具链 API 变动 | 测试脚本失效 | 只依赖稳定公共 API；变更先协调 ToolChain |
| Kernel 公共接口被要求扩展 | 契约反复 | 加载期准备放 `CreateKernel`；扩展必须先改 Runtime_design |

---

## 6. 测试与验证设计

### 6.1 测试策略

| 层次 | 范围 | 环境 | 目的 |
| --- | --- | --- | --- |
| 接口检查 | 公共头自洽、无厂商类型、无反向依赖 | 主机 | 冻结契约 |
| 后端契约测试 | Init/Deinit、Capability、Buffer 往返、ConvertParam、错误路径、泄漏 | 主机（mock）；设备（ocl） | 后端实现正确 |
| OpEnv/Kernel ABI 测试 | KernelContext 组装、StepContext、IO 绑定、失败传播 | 主机（mock） | OpEnv 框架正确 |
| 算子精度测试 | 8 个融合算子 vs torch CPU 散算子 | 主机（mock）；设备（ocl） | 数值正确 |
| 全图/端到端 | Qwen3 单请求 | 设备 | 后续集成（不属于本模块单算子验收） |

### 6.2 后端契约测试用例

| 用例 | 场景 | 通过标准 |
| --- | --- | --- |
| TC-BE-01 | `Init` 正常/重复/失败；`Deinit` 重复 | 错误码正确；无句柄泄漏；重复 Deinit 安全 |
| TC-BE-02 | 注册表：已注册/未知名称/重复注册 | `Create` 正确；未知返回 `kUnsupportedBackend` |
| TC-BE-03 | Buffer：正常、超 `max_buffer_bytes`、对齐不满足、上传/下载往返（含 1 B、奇数长度、随机数据） | 数据逐字节一致；错误码正确 |
| TC-BE-04 | ConvertParam：FP16 直通、FP32→FP16、未知 dtype、空指针 | 数据/精度正确；错误码正确 |
| TC-BE-05 | CreateKernel：8 个 kernel_key、未知 key、空 desc | 成功/`kUnsupportedOp` 正确；无泄漏 |
| TC-BE-06 | Capability：未 Init、字段完整、op/dtype 覆盖 | 返回 `kStateError` 或完整能力 |
| TC-BE-07 | Synchronize：空流、有在途任务 | 返回后数据可见 |
| TC-BE-08 | 错误路径：设备丢失/编译失败/超大分配（可注入） | 错误码与消息可定位；不崩溃 |
| TC-BE-09 | 生命周期：100 次 Init/Deinit；Buffer/Kernel 计数归零 | 无泄漏 |
| TC-BE-10 | 公共头：`-fsyntax-only`；扫描 `cl_`/`Vk`/`cuda` 类型 | 零错误；公共头零厂商类型 |

### 6.3 算子精度测试

#### 6.3.1 测试目标

对 8 个融合算子分别验证：
1. Python 端能把融合算子表达为单 op 图；
2. 该图能经 Optimizer/Serializer 生成可加载模型包；
3. Runtime 经 Loader 加载、OpEnv::Forward 执行、Destroy 释放；
4. 输出与 torch CPU 散算子组合的参考结果在阈值内一致。

#### 6.3.2 流水线阶段取舍

| 阶段 | 是否执行 | 说明 |
| --- | --- | --- |
| Converter（HF 模型 Adapter） | **省略** | 单算子图不是模型；模型 Adapter 无意义。用 Python 直接构造 `ModelGraph`（可复用 `ToolChain.testing.make_tensor`），必要时用 `RecordingGraph` 代理 |
| RecordingGraph | **可选** | 仅在需要模拟 Converter 记录行为时使用；直接构造 ModelGraph 更简单 |
| Optimizer | **不可省略** | 产生精度/形状/显存计划；Serializer 与 Loader 依赖其产物；同时校验算子形状契约 |
| Memory Manage（Optimizer 内） | **不可省略** | 提供 Activation offset/arena 峰值；Loader 按计划分配与绑定 |
| Serializer | **不可省略** | 生成 `model.mgf` + `weights.bin`；Loader 消费真实包并校验 IR 哈希/权重 |
| Loader | **不可省略** | 测试包校验、图重建、后端初始化、权重上传、缓冲分配、OpEnv 创建 |
| Engine | **省略** | 单算子无生命周期/线程/IO 绑定需求 |
| Scheduler | **省略** | 无组批/停止判定 |
| KVCache | **省略** | `use_kv_cache=false`；Attention 单算子用完整 S |
| Executor | **省略** | 单算子由测试驱动直接调用 OpEnv；Executor 的 Packing/plan 适配不在本测试范围 |
| OpEnv::Forward | **不可省略** | 被测目标 |
| Destroy | **不可省略** | 验证 Kernel/Buffer/GraphEnv 释放与泄漏 |
| Tokenizer/Service | **省略** | 阶段二 |

快速路径（仅调试，不作为验收）：

```text
Python/C++ 直接构图 → mock Backend → CreateBuffer/Upload → OpEnv::Forward → 对比
```

该路径跳过 Optimizer/Serializer/Loader，只用于 Kernel 早期调试；不得写入 `tests/test_results.md`
作为算子验收结论。

Loader 内部环节的取舍：

| Loader 环节 | 测试中 | 说明 |
| --- | --- | --- |
| 包头 magic/version/manifest SHA 校验 | 保留 | 成本低，验证真实包 |
| manifest JSON 解析与图重建 | 保留 | 被测路径 |
| `DumpIr` + IR 哈希 | **必须保留** | 防止合成图与 C++ 构图语义漂移 |
| Backend Init + Capability 校验 | 保留 | 验证能力声明与算子覆盖 |
| 权重流式上传 | 保留 | 算子测试的核心依赖 |
| 权重逐项 SHA-256 校验 | 可关（`verify_weights_sha256=false`） | 仅影响速度；验收默认开启 |
| arena/I/O 缓冲分配与 storage 回填 | 保留 | Kernel 寻址依赖 |
| GraphEnv + OpEnvFactory + OpEnv::Init | 保留 | 被测路径 |
| KVCache 初始化 | 省略 | 单算子测试不建 KV；`use_kv_cache=false` |
| Engine/Scheduler/Executor | 省略 | 由测试驱动替代 |
| Warmup | 不存在（v1 无） | — |

Destroy 的取舍：算子验收必须经过真实 Destroy 路径（Synchronize → ClearOpEnv → ResourceStack →
ClearGraphEnv → Deinit），以获得无泄漏证据；只有快速路径允许测试本地等价清理，且不得作为验收结论。

#### 6.3.3 Python 侧构造

每算子 `tests/test_<op>.py` 负责：

1. **算子规格**：`op_type`、attrs、输入/常量/输出名称、dtype、shape。
2. **数据生成**：
   - 固定 `torch.manual_seed`；
   - 权重 `torch.randn(shape, dtype=torch.float16)`（或按算子需要缩放）；
   - 输入 `torch.randn`/`randint`（gather 的 id 在 `[0,V)`）；
   - 权重与输入先转 `float16`，作为唯一数值真源。
3. **参考计算**：
   - 用 torch 散算子组合（见 4.3.2 表）；
   - 计算前把 FP16 张量转 `float32`，按算子数学式计算；
   - 对 `sample` 用同一 FP16 logits 求 argmax，避免并列；
   - 参考输出保留 FP32，另存一份 FP16 便于对比舍入。
4. **ModelGraph**：用 `ToolChain.common.model_ir` 构造张量/Op/IO 绑定；常量 `weight_source`
   与 safetensors key 一致；`metadata` 写入必要的 `model_config`（如 `tie_word_embeddings=False`）。
5. **权重文件**（算子有常量时）：`ToolChain.testing.write_safetensors` 写 FP16 权重，
   key = `weight_source`；`add`/`mega_attention`/`sample` 无常量，跳过该步并允许 metadata-only 包。
6. **优化/序列化**：
   ```python
   optimizer = Optimizer(OptimizeConfig(
       target_precision="float16", accumulate_precision="float32",
       batch_size=1, max_num_tokens=S, context_length=64,
       alignment=256, enable_reuse=True))
   result = optimizer.run(model_graph, weights_path=weights_path)
   serial = serialize_package(result, package_dir, weights_path=weights_path,
                              overwrite=True)
   ```
7. **case 数据**：写 `case.json`、`inputs/<name>.bin`、`expected/<name>.bin`；
   `case.json` 记录 op 名、backend、包目录、输入/输出文件与 dtype/shape。
8. **调用 op_runner**：
   ```bash
   PYTHONUTF8=1 python Runtime/backend/ops/tests/run_case.py \
     --package <pkg> --case <case.json> --backend mock \
     --runner <build>/op_runner
   ```
9. **对比与报告**：读取 `actual/*.bin`，计算指标，断言阈值，输出 JSON 与 Markdown 片段。

算子声明示例（核心思想：同一组权重/输入，左边是 Python 单融合算子图，右边是 torch 散算子组合）：

```python
# Runtime/backend/ops/linear_swi_glu/tests/test_linear_swi_glu.py（示意）
def torch_reference(x, gate, up):
    xf = x.float()
    return (torch.nn.functional.silu(xf @ gate.float().T)
            * (xf @ up.float().T)).half()

case = OpCase(
    name="mlp_gate_up",
    op_type="linear_swi_glu",
    attrs={"intermediate_size": I, "activation": "silu",
           "transpose_weight": True, "precision": "float16"},
    inputs=[TensorSpec("input.x", "float16", (1, S, H))],
    constants=[
        TensorSpec("mlp.gate.weight", "float16", (I, H), "test.gate"),
        TensorSpec("mlp.up.weight", "float16", (I, H), "test.up"),
    ],
    outputs=[TensorSpec("output.y", "float16", (1, S, I))],
    reference=torch_reference,
)
```

- 左图：`linear_swi_glu` 一个 op；右图：`linear(gate) + silu + linear(up) + mul` 四个 torch 散算子；
- 两侧使用**同一份 FP16 权重与输入**，因此差异只来自 Kernel 数学/累加顺序/舍入；
- 每个算子按同样模板声明，`OpCase`/`TensorSpec` 由 `ops/tests/op_case.py` 提供。

#### 6.3.4 C++ op_runner

职责：

1. `CreateDefaultLoader()->Load(package, resource_stack, &result)`；记录 Graph/Backend/ModelConfig。
2. 构造 `TestIoBinder`：按 `case.json` 把主机 `Storage` 写入 Input/Output op 的 `attr["storage"]`；
   名称/大小不匹配返回错误。
3. 找到目标计算 op（按 `case.json.op`），校验其 `OpEnv` 是 `ExecOpEnv`。
4. 构造 `StepContext`：`backend=result.backend.get()`、`batch=1`、`packed_tokens=S`、
   `seq_lens` 指向 `[S]`、`is_prefill=true`、`plan=null`、`kv_view=null`。
5. 对 `GetOrderlyOPList()` 中每个 op 调用 `SetStepContext`，再逐个 `Forward()`：
   Input（Upload）→ 目标计算 op（Kernel::Execute）→ Output（Download）。
6. `Backend::Synchronize(nullptr)`。
7. 把每个输出 op 的主机缓冲区写 `actual/<name>.bin`。
8. `CreateDefaultDestroyer()->Destroy(ctx)`；context 中 `executor=null`、`backend_initialized=true`、
   `executor_initialized=false`；随后释放 `result.backend`/`result.graph` 与 ResourceStack。
9. 返回非 0 表示失败，并打印带 op 名的错误。

约束：`op_runner` 只依赖公共接口（Loader/Backend/Graph/OpEnv/Destroyer），不得包含 OCL 头；
backend 通过名称参数选择。它属于测试代码，不进入产品库。

#### 6.3.5 用例设计

每个算子至少 3 组：

| 用例 | 内容 |
| --- | --- |
| `small/default` | 默认 small 形状、随机 FP16、固定 seed |
| `small/boundary` | `S=1`、全零输入、极大/极小值（不溢出）、单 head/单 KV head（合法比例） |
| `small/seed2` | 不同 seed，验证非偶然一致 |
| `real_shape`（slow） | Qwen3 真实维度的一次冒烟（如 `linear` N=151936 可单独标注） |
| 错误路径（每算子） | 缺 attr、错误 dtype、形状不匹配、缺少权重 → 加载期失败并定位 |

`linear_per_head_rmsnorm_rope` 额外用例：分别覆盖 q 路径（`num_heads=4`）与 k 路径
（`num_heads=2`），以及 `S=1` 的位置编码边界；`mega_attention` 额外用例：
`heads/kv_heads = 4/2`、`S=1`、causal 边界；`sample` 额外验证 argmax 并列时取最小下标
（构造并列 logits）。

#### 6.3.6 精度阈值（初始，实测后锁定）

| 算子 | max_abs | relative_l2 | cosine | 其他 |
| --- | --- | --- | --- | --- |
| `gather` | 0 | 0 | 1.0 | 逐字节一致 |
| `add` | 5e-3 | 1e-3 | 0.99999 | — |
| `rms_norm` | 2e-2 | 5e-3 | 0.9999 | 无 NaN/Inf |
| `linear` | 2e-2 | 5e-3 | 0.9999 | 无 NaN/Inf |
| `linear_swi_glu` | 2e-2 | 5e-3 | 0.9999 | 无 NaN/Inf |
| `linear_per_head_rmsnorm_rope` | 2e-2 | 5e-3 | 0.9999 | 无 NaN/Inf |
| `mega_attention` | 3e-2 | 1e-2 | 0.999 | 无 NaN/Inf |
| `sample` | — | — | — | Token 100% 一致（无并列数据） |

说明：参考值使用与运行时相同的 FP16 输入/权重转 FP32 计算，因此主要差异来自累加顺序与舍入；
阈值先宽松，采集真机/多组数据后在各算子文档中锁定并记录依据。

#### 6.3.7 测试结果记录

每算子 `tests/test_results.md` 必须包含：

1. 环境：日期、OS、编译器、Python/torch 版本、backend、设备（mock/OCL 型号/驱动）；
2. 命令：包生成、op_runner、对比脚本的完整命令；
3. 用例：seed、形状、输入范围；
4. 实际输出：指标数值、通过/失败、错误消息（如有）；
5. 结论：是否达到阈值；未覆盖项与风险；
6. 可复现信息：模型包/权重/IR 的 SHA-256（由 Serializer 报告提供）。

不得只写“测试通过”。失败时保留产物与日志路径。

### 6.4 测试环境

| 环境 | 配置 | 用途 |
| --- | --- | --- |
| 主机 | Windows 11 / Linux；C++17；CMake 3.20+；Python 3.10+；torch CPU | 接口、mock 后端、8 算子精度、包生成 |
| 参考机 | Snapdragon 8 系 + Adreno + OpenCL 1.2+ | OCL 契约与算子精度 |
| 工具 | `PYTHONUTF8=1 python`、C++ 测试二进制、SHA-256 工具 | 复现与审计 |

当前主机已确认：Python 3.13、torch 2.12.0+cpu、`OpenCL.dll` 存在但无 Khronos 头文件；
OCL 主机编译需先解决头文件问题（L-BE-02）。

### 6.5 通过标准

- 接口：公共头 `-fsyntax-only` 通过；公共头零厂商类型。
- 后端：mock 契约测试全部通过；OCL 契约测试在参考机全部通过。
- 算子：8 个算子各至少 3 组用例通过；`sample` Token 100%；无 NaN/Inf；无泄漏。
- 四件套：每算子 `.doc`/代码/`tests/test_results.md`/`.skill` 齐全且与实现一致。
- 回归：任何算子阈值收紧/放宽必须记录原因；性能回退不适用本模块单算子验收。

### 6.6 回归与验收

- 每次 Kernel/OpEnv 改动运行对应算子测试与 mock 契约测试；
- OCL 改动按参考机每日/里程碑回归；
- 新增算子按 4.3.4 流程提交四件套并纳入本文档清单；
- 验收材料：本文件、算子文档、测试结果、skill、模型包哈希。

### 6.7 未覆盖项与风险

- 未覆盖真实设备多驱动/多 SoC 差异（阶段一参考机锁定后补）；
- 未覆盖量化、Image 路径、多队列、异步传输；
- 未覆盖多算子组合图与端到端 Qwen3（属于集成测试）；
- 未覆盖 Arena 复用对算子数值的影响（由 MemoryPlan 保证不重叠）；
- 真实形状 `lm_head` 权重 311 MB，CI 默认不跑；
- Loader/Destroyer 未就绪前，算子测试只能跑快速路径，不能宣称算子验收完成。

---

## 附录 A：算子测试命令模板

```bash
# 1. 生成测试包与数据（每算子脚本内部完成）
cd d:/Mobile_Net
PYTHONUTF8=1 python Runtime/backend/ops/linear/tests/test_linear.py \
  --work-dir TempSpace/backend_op_linear_20261117_120000 \
  --backend mock

# 2. 上述脚本会调用 op_runner：
Runtime/build/backend/op_runner \
  --package TempSpace/backend_op_linear_20261117_120000/package \
  --case TempSpace/backend_op_linear_20261117_120000/case.json \
  --backend mock

# 3. 对比与报告
PYTHONUTF8=1 python Runtime/backend/ops/tests/compare.py \
  --case TempSpace/backend_op_linear_20261117_120000/case.json \
  --actual TempSpace/backend_op_linear_20261117_120000/actual \
  --expected TempSpace/backend_op_linear_20261117_120000/expected
```

## 附录 B：算子模块四件套检查清单

- [ ] `.doc/<op>_design.md` 六段式齐全，含 attr/形状/数值/错误/测试；
- [ ] `ops/<op>/` 语义代码与文档一致；Google Style；中文注释；
- [ ] `mock/kernels/` 与 `ocl/kernels/` 实现已接入并登记 kernel_key；
- [ ] `tests/test_<op>.py` 覆盖 default/boundary/seed2/错误路径；
- [ ] `tests/test_results.md` 记录环境、命令、指标、结论、未覆盖项；
- [ ] `.skill/SKILL.md` 说明适用场景、输入、步骤、输出与陷阱；
- [ ] 更新本文档 3.4/4.3.2 清单与 Runtime 架构文档（如涉及冻结接口）。

## 附录 C：术语

| 术语 | 说明 |
| --- | --- |
| OpSpec | 单个融合算子的 attr/形状校验与 KernelDesc 组装规格 |
| KernelContext | OpEnv 传给 Kernel 的单节点执行上下文 |
| StepContext | Executor 注入 OpEnv 的本步 plan/KV/seq 元数据 |
| KernelOpEnv | 通用计算 op 的执行环境 |
| IoOpEnv | Input/Output 边界 op 的 H2D/D2H 执行环境 |
| Program 缓存 | OCL 编译产物缓存，键含源码/选项/驱动/设备 |
| 算子精度测试 | 单融合算子 vs torch 散算子组合的 CPU 参考对比 |
