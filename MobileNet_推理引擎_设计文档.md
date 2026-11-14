# MobileNet 推理引擎设计文档（大语言模型引擎）

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.2 |
| 文档状态 | 架构设计（Graph 对齐稿） |
| 编写日期 | 2026-11-13 |
| 适用范围 | 阶段一：大语言模型推理；阶段二：大语言模型服务框架 |
| 后续文档 | 各模块设计文档（拆分计划见 3.7） |
| 强制规则 | 项目 memory.md：Google Style、中文注释、四件套、六段式设计文档 |

**文档约定**

- “首期”指阶段一与阶段二；“后续阶段”指多模态服务框架（阶段三）与对话 Agent（阶段四）。
- 本文是总体架构文档，描述模块边界、接口契约、关键流程与验收方式，不展开单模块的算法与实现细节。
- 标注“待校准”的性能、内存数字为初始目标，须在阶段一启动时以真机实测锁定，并在模块设计文档中记录测量方法。
- 每个模块的详细设计、代码、测试结果与 skill 按四件套另行交付，本文不替代模块设计文档。

## 目录

1. [背景与目标](#1-背景与目标)
2. [需求分析](#2-需求分析)
3. [概要设计](#3-概要设计)
4. [详细设计](#4-详细设计)
5. [实施步骤](#5-实施步骤)
6. [测试与验证设计](#6-测试与验证设计)
7. [附录](#附录)

# 1. 背景与目标

## 1.1 背景

Agent 智能体在服务侧的应用部署已经非常广泛，手机端或者其他边缘端设备能够高效应用智能体，会是一个值得开拓的方向。

本项目旨在开发一套全自研的端侧大语言模型、多模态模型、生成类模型的推理引擎，再扩展为端侧可离线应用的智能体服务框架。

整体项目按四个阶段演进：

1. 大语言模型推理：手机端 GPU 部署 4B 以下大语言模型，实现动态调度场景下文本输入、文本自回归输出的模型推理。
2. 大语言模型服务框架：手机端 CPU 部署 Tokenizer 服务，支持 PrefixCache 的调度，并结合第一阶段的 GPU 模型推理，形成文本生成服务框架。
3. 多模态服务框架：在第二阶段服务框架基础上，实现图片或语音输入、文本自回归输出的原生多模态模型推理服务。
4. 对话 Agent：完成手机端 APP 功能开发，后端调用第一、二、三阶段的推理服务，实现可视化离线 Agent 对话功能。

本文覆盖阶段一与阶段二的架构设计；阶段三、阶段四只在扩展性设计中定义接口与依赖关系，不展开详细设计。

## 1.2 目标

- **阶段一目标**：在手机端 GPU 上完成 Qwen3-0.6B（FP16）的端到端自回归文本生成；支持多请求动态调度（Continuous Batching）、Paged Attention KV Cache；实现离线工具链到在线运行时的完整闭环。
- **阶段二目标**：在阶段一之上增加 CPU Tokenizer 服务、PrefixCache 调度、会话与流式输出，形成可被上层 APP 调用的大语言模型服务框架。
- **长期目标**：同一套图、工具链与运行时架构可扩展到多模态推理与端侧 Agent 服务，新增模型与新增后端不推翻既有架构。
- **工程目标**：每个模块以设计文档、代码、测试结果、skill 四件套交付；所有模块可独立开发、独立测试、可替换。

## 1.3 非目标

- 不做模型训练、微调、评测平台；不实现训练框架。
- 不做云端推理服务、多租户、分布式部署；首期是单机、单进程、离线运行。
- 首期不做多模态输入输出（阶段三）与 APP 界面（阶段四）。
- 首期只交付高通 Adreno GPU 的 OpenCL 后端；Vulkan、CUDA、CPU 等后端不在首期实现，但按“目录级插件”规范预留接入能力。
- 不做模型权重分发、加密、版权与商业授权管理。
- 不做投机解码、MoE、多模型流水线等高级特性；架构预留，不在首期验收范围。

## 1.4 设计原则

1. **插件化、模块化**：所有可插拔模块组成完整服务框架，模块间只依赖接口与数据契约。
2. **搭积木、分阶段**：严格分阶段、分模块演进，每完成一块即可独立测试与验收，逐步累加功能。
3. **文档先行、四件套**：严格按“人机交互设计文档 → Agent 开发代码 → 测试验证 → 沉淀 skill”的流程执行；设计文档、代码、测试结果、skill 四件套缺一不可。
4. **离线重、在线轻**：复杂转换、融合、精度/形状推导、显存编排尽量前置到工具链；运行时只做加载、调度、执行与释放，保持轻量、确定、可预测。
5. **数值可对齐、可验证**：以 PyTorch 参考实现为数值金标准；工具链输出结构化日志，便于 Agent 自动校验与人工审查。
6. **接口先行、并行开发**：模块间先冻结接口与数据契约，允许以 mock 并行开发；接口变更必须先改设计文档。

# 2. 需求分析

## 2.1 阶段规划与本文范围

| 阶段 | 内容 | 本文范围 |
| --- | --- | --- |
| 阶段一：大语言模型推理 | 手机端 GPU 部署 4B 以下大语言模型；动态调度；文本自回归生成 | 核心设计范围 |
| 阶段二：大语言模型服务框架 | CPU Tokenizer 服务、PrefixCache 调度、会话与流式输出 | 设计范围 |
| 阶段三：多模态服务框架 | 图片/语音输入、文本输出 | 仅定义扩展点 |
| 阶段四：对话 Agent | 手机 APP、可视化离线对话 | 仅定义依赖接口 |

## 2.2 功能需求

### 2.2.1 阶段一：大语言模型推理

| 编号 | 需求描述 | 验收要点 | 关联用例 |
| --- | --- | --- | --- |
| FR-1.1 | 提供基于 Graph 的 runtime 静态图对象模型：Graph 持有 Op、Tensor、constant tensor、图输入/输出与拓扑序列；Op 以 op_type、attr、src_tensors、dst_tensors 描述计算/输入/输出节点，并挂载 OpEnv；Tensor 表示数据依赖边并内嵌 Storage（shape、precision、layout、device_type、offset、byte_size、data_ptr、reusable）；constant tensor 作为依赖叶子独立持有，不进入执行序列；Graph 提供 create_input、create_constant、create_op、create_output、SortOp 与 GetAllConstantTensor/GetAllInputOp/GetAllOutputOp/GetOrderlyOPList 查询；SortOp 从图输出做后序 DFS，保证生产者先于消费者；Graph 不执行 Forward、不分配显存、不推导 shape/precision/layout，执行由 Engine 为每个 op 创建 OpEnv 并调用 OpEnv::Forward 完成。 | 合法图可构造并通过校验；非法图（重名、成环、悬空依赖、缺少输出生产者、I/O 注册非法）被拒绝并定位；SortOp 后生产者全部先于消费者；constant 不出现在执行序列；Graph 与 OpEnv 生命周期可正确释放。 | TC-GRAPH-01、TC-EXEC-01 |
| FR-1.2 | Graph 对象模型通过 pybind11 暴露给 Python，不使用 LibTorch type caster：Python 侧只暴露 Graph、Tensor（name、is_constant、reusable、set_reusable）与 OpType/Precision/Layout/DeviceType 枚举，以及 create_input、create_constant、create_op、create_output、sort_op、dump_ir；不暴露 Op、Storage、Shape 访问器，不提供序列化与 Forward。Build 阶段由 Converter 的 RecordingGraph 透明代理构图 API，并旁路记录 ModelGraph 元数据 sidecar，供 Optimizer 与 Serializer 使用。 | Python 构造图与 C++ 构造图的 dump_ir 逐字节一致；ModelGraph 的 op/tensor 计数与 dump_ir 一致；Python 侧不存在 torch.Tensor/torch.dtype 属性；C++ 运行时不链接 LibTorch。 | TC-GRAPH-02 |
| FR-1.3 | Converter 将模型厂商发布的模型源码（首期 Qwen3-0.6B）转换为 Graph 静态图，并同步旁路记录 ModelGraph；完整保留原图结构、算子属性、常量权重来源与图输出。 | 节点/常量/权重映射与参考模型一致；RecordingGraph 与原生 Graph 的 dump_ir 一致；转换报告完整。 | TC-CONV-01 |
| FR-1.4 | Precision Infer 将 BF16 权重与激活转换为 FP16，并为所有 op 的输入、输出、参数、权重确定唯一精度。后续模型量化（INT8/INT4 等）统一在 Precision Infer 阶段扩展，不新增独立优化阶段。 | 所有 Tensor 均有明确 dtype；精度策略可审查；量化扩展点已在精度表中预留。 | TC-OPT-01 |
| FR-1.5 | Shape Infer 以模型 Config、MaxNumToken、ContextLength 为输入，推导图中所有 Tensor 与 op 的形状，并校验合法性。 | 形状与配置及运行边界一致；无未推导维度。 | TC-OPT-01 |
| FR-1.6 | Memory Manage 基于 ModelGraph 的静态依赖与生命周期分析推导 Activation 显存复用方案，生成显存排版，降低峰值占用；不执行图，也不分配真实显存。 | 活跃区间无冲突；开启/关闭复用图结构与输出绑定一致；arena 不小于峰值。 | TC-OPT-02 |
| FR-1.7 | Optimizer 以结构化日志输出所有算子的精度、形状、显存排版与生命周期，供 Agent 自动校验与人工审查。 | 日志可被脚本解析；关键不变量自动检查通过。 | TC-OPT-01 |
| FR-1.8 | Serializer 把优化后的 ModelGraph 固化为自描述的 `model.mgf` 模型包（112 字节固定头 + 规范化 JSON manifest），可选写出独立 `weights.bin`；manifest 包含完整图、显存计划、权重表、格式版本与 SHA-256 校验；Loader 可据此重建 Graph。 | 序列化往返图一致；重建图 dump_ir 的 SHA-256 与 manifest 记录一致；权重校验和一致；无权重时可生成 metadata-only 包。 | TC-SER-01 |
| FR-1.9 | Loader 读取并校验 `model.mgf`，按 manifest 重建 Graph，按推理配置从后端注册表选择并初始化后端（首期 OpenCL），完成 GraphEnv/OpEnv、Kernel 准备、权重转换/上传、输入/输出存储绑定与运行时资源创建。 | 正常包加载成功；后端选择正确；未知后端、损坏文件或缺少 I/O 传输能力返回错误码且不崩溃。 | TC-LOAD-01 |
| FR-1.10 | Scheduler 实现 Continuous Batching；维护 Waiting/Running 队列；对上层提供 Merge（添加请求）、Update（回写生成 Token 与状态）、Stop（结束请求）接口。 | 多请求并发正确、无串扰；取消与超时生效。 | TC-SCHED-01、TC-SCHED-02 |
| FR-1.11 | KVCache Manager 实现 Paged Attention 的 KV Block 分配、追加、回收与 Block Table 管理，支持多请求并发与显存不足时的抢占。 | 块分配不变量成立；OOM 时触发抢占并可恢复。 | TC-KV-01 |
| FR-1.12 | Executor 按调度计划构建主机输入/输出绑定与 ForwardContext；通过 Engine 的 I/O 绑定接口写入 Input/Output op 的 storage attr，随后按 Graph::GetOrderlyOPList() 逐 op 调用 OpEnv::Forward()，完成 Input H2D、计算与 Output D2H；经 Backend/Kernel 公共接口提交节点执行；优先对 Waiting 请求做 Prefill，再对 Running 请求做 Decode；同一 Step 内按 Token 维度 Packing 多请求；采样 op 输出 Token。 | 单请求/多请求结果与参考一致；H2D/D2H 往返正确；Kernel 独立对齐通过；Executor 无后端专有代码；不直接修改 Graph 拓扑。 | TC-EXEC-01、TC-KERNEL-01 |
| FR-1.13 | Engine 聚合 Loader、Scheduler、KVCache Manager、Executor、Destroy，持有 Backend、Graph 与各 op 的 OpEnv，对上层提供统一的 Load、Schedule、Execute、Destroy 接口与生命周期管理。 | 生命周期状态正确；Backend、Graph 与 OpEnv 生命周期正确；接口可重复调用且错误可诊断。 | TC-ENG-01 |
| FR-1.14 | Destroy 按依赖顺序释放各 op 的 OpEnv、Kernel、Buffer（含输入/输出设备缓冲）、GraphEnv、Graph、Backend 与主机资源；释放接口可重复调用，不产生内存、设备或线程资源泄漏。 | 反复 Load/Destroy 无泄漏；后端句柄计数归零；Destroy 幂等。 | TC-ENG-01 |

### 2.2.2 阶段二：大语言模型服务框架

| 编号 | 需求描述 | 验收要点 | 关联用例 |
| --- | --- | --- | --- |
| FR-2.1 | Tokenizer 服务在 CPU 上提供文本与 Token 的编码/解码及流式 Detokenize，支持中英文、Emoji、特殊 Token。 | 编解码往返一致；流式输出 UTF-8 边界正确。 | TC-TOK-01 |
| FR-2.2 | PrefixCache 调度对请求 Token 前缀建立索引；命中时复用已缓存 KV Block，仅对未命中后缀执行 Prefill；支持引用计数、写时复制与 LRU 淘汰。 | 命中后输出与无缓存一致；Prefill 开销下降。 | TC-PREFIX-01 |
| FR-2.3 | Session 服务提供会话创建、对话请求、流式返回、取消与释放接口，隔离不同会话的上下文与采样参数。 | 多会话隔离；流式顺序正确；取消生效。 | TC-SVC-01 |
| FR-2.4 | Tokenizer/Detokenizer 与 GPU Prefill/Decode 并行执行，减少串行等待。 | 时间线可观测重叠；吞吐不低于串行基线。 | TC-PIPE-01 |

## 2.3 非功能需求

| 编号 | 类别 | 要求 | 验证方式 |
| --- | --- | --- | --- |
| NFR-1 | 性能 | 参考机（高通骁龙 8 系 Adreno GPU，机型待锁定）上，Qwen3-0.6B、FP16、上下文 2048：Decode ≥ 10 token/s；128 Token Prompt 的 Prefill ≥ 100 token/s，首 Token 延迟 ≤ 2 s；4 并发总吞吐 ≥ 单请求的 1.5 倍。以上为初始目标，待真机基准校准后锁定。 | TC-PERF-01 |
| NFR-2 | 内存 | Qwen3-0.6B、FP16、上下文 2048、单会话：进程峰值内存 ≤ 3 GB（含权重、KV Cache、Activation 与运行时开销，不含 Android 系统与 APP）。初始目标，待真机测量确认。 | TC-MEM-01 |
| NFR-3 | 可移植性 | 设备后端以 Backend/Kernel 公共接口 + 目录级插件接入；新增后端仅新增 `Runtime/backend/<name>/` 目录（接口实现、Kernel、自注册、构建脚本），不修改图、调度、Executor、Engine 与上层接口；缺少算子或精度能力时在加载期明确报错。 | TC-COMPAT-01 |
| NFR-4 | 可测试性与可观测性 | 工具链输出 JSONL 结构化日志（每算子精度、形状、显存偏移、生命周期）；运行时输出请求级与 Step 级统计；关键模块可在主机侧单测。 | TC-OPT-01、TC-ENG-01 |
| NFR-5 | 稳定性与健壮性 | 非法模型文件、版本不兼容、设备异常、OOM、请求取消均返回明确错误码或受控状态，不崩溃；1 小时混合负载压测内存增长 ≤ 5%，无资源泄漏。 | TC-ROBUST-01、TC-STAB-01 |
| NFR-6 | 工程规范 | 代码遵循 Google C++/Python Style；模块级、类级、函数级中文注释覆盖功能边界、输入输出与主要逻辑；每模块交付四件套。 | TC-DOC-01 |
| NFR-7 | 模型支持范围 | 架构支持 4B 以下 Dense Decoder-only 模型；首期以 Qwen3-0.6B 完成验收；4B 级模型需结合量化能力，后续量化统一在 Precision Infer 阶段扩展，不在阶段一验收范围。 | TC-CONV-01、TC-EXEC-01 |

## 2.4 约束条件

- **工具链环境**：Python 3.10+（当前验证环境为 Python 3.13）、CMake 3.20+；运行于 Windows 11 或 Linux 开发机。Build 阶段不依赖 PyTorch、LibTorch 或 Protobuf；Graph 原生扩展由仓库内 CMake 构建，产物位于 `Build/graph/python/`。
- **运行时环境**：C++17、CMake 3.20+、OpenCL 1.2+（优先使用设备支持的更高版本）；首期目标平台为 Android arm64-v8a（Android NDK），主机侧编译用于单元测试。Graph 与运行时均不链接 LibTorch。
- **图表示约束**：全项目只有一套可执行图表示，即 Graph 的 C++17 runtime IR（Graph/Op/Tensor/Storage/GraphEnv/OpEnv）；Python 侧仅通过 pybind11 最小构图接口访问，不暴露 Op/Storage/Shape 访问器。Build 阶段的优化与序列化元数据由 Converter 旁路记录的 `ModelGraph` sidecar 承载；不存在 Graph IR/RuntimeGraph 双表示，也不存在 `TensorDesc` 转换层。
- **设备目标**：首期只实现高通 Adreno GPU 的 OpenCL 后端；其他 GPU/CPU 后端不在首期实现，按目录级插件规范预留接入能力。
- **离线运行**：运行时不访问网络；模型、权重、Tokenizer 文件全部本地加载。
- **格式自主**：模型包 `model.mgf`（固定头 + 规范化 JSON manifest + 可选 `weights.bin`）由本项目定义并版本化，不兼容第三方推理格式。
- **资源受限**：手机内存与显存受限，必须支持 Activation 复用与 KV 分页，且具备 OOM 抢占能力。
- **流程约束**：所有开发遵循项目 memory.md 的编码规范、中文注释、四件套与六段式设计文档要求。
- **后端插件约束**：所有后端实现必须位于 `Runtime/backend/<name>/` 独立目录；公共头文件不得暴露后端专有类型；首期只验收 OCL，其他后端只验证插件机制。

## 2.5 验收标准

| 编号 | 验收项 | 通过条件 |
| --- | --- | --- |
| AC-1 | 阶段一端到端功能 | Qwen3-0.6B 在参考机 GPU 上完成文本输入到自回归文本输出；贪心 64 Token 与 PyTorch 参考一致率 ≥ 95%。 |
| AC-2 | 工具链完整性 | 图、权重、优化日志完整且可复现；所有 Tensor 的精度、形状、显存排版可由脚本校验；Python 构建的 Graph `dump_ir` 与重建图逐字节一致，IR SHA-256 与 manifest 一致。 |
| AC-3 | 动态调度 | ≥ 4 并发请求连续批处理；输出无串扰；请求可取消；KV 不足时可抢占并恢复。 |
| AC-4 | 资源与稳定性 | 峰值内存 ≤ 3 GB；1 小时混合负载无崩溃、无泄漏。 |
| AC-5 | 阶段二服务框架 | Tokenizer 编解码往返正确；PrefixCache 命中后输出一致且 Prefill 开销下降；流式会话可用。 |
| AC-6 | 工程交付 | 每个模块四件套齐全；Google Style 与中文注释检查通过。 |
| AC-7 | 性能 | 达到 2.3 节锁定后的性能目标（初始目标见 NFR-1）。 |

# 3. 概要设计

## 3.1 总体架构

整个推理引擎分为两个阶段：**离线工具链编译阶段**（以下简称工具链）与**在线推理运行时阶段**（以下简称运行时）。

- **工具链**：主要由 Python 实现，在个人主机或服务器上执行，负责将模型厂商发布的模型经过转换、优化、编译并序列化，产出运行时可直接加载的模型图文件与权重文件。
- **运行时**：主要由 C++ 实现，在手机端等边缘设备 GPU 上执行，负责加载工具链产物，完成请求调度、模型前向（按 `GetOrderlyOPList()` 依次调用 `OpEnv::Forward()`）与 Token 生成。

```text
┌────────────────────────────── 离线：工具链（PC / 服务器） ──────────────────────────────┐
│                                                                                         │
│   HF 模型源码（config / tokenizer / weights）                                            │
│       │                                                                                 │
│       ▼                                                                                 │
│   Converter ──────▶ Graph.Graph（C++17 runtime IR）                               │
│       │                    │  旁路记录 ModelGraph（元数据 sidecar）                      │
│       │                    ▼                                                            │
│       │              Optimizer（Precision → Shape → Memory，作用于 ModelGraph）          │
│       │                    │                                                            │
│       ▼                    ▼                                                            │
│   Serializer ──────▶ model.mgf + weights.bin（可选） + compile_report.json               │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │ 模型包部署
┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
│                          在线：运行时（手机端，C++ / OpenCL）                            │
│                                                                                         │
│   APP / Agent（阶段四）                                                                 │
│       │                                                                                 │
│       ▼                                                                                 │
│   Service Framework（阶段二）                                                           │
│   Tokenizer(CPU) ──▶ Session ──▶ PrefixCache ──▶ Engine                                 │
│                                                      │                                  │
│   ┌──────────────────────────────────────────────────┘                                  │
│   │  Engine（阶段一）                                                                    │
│   │    Loader / Scheduler / KVCache Manager / Executor / Destroy                        │
│   │         │                                                                          │
│   │         ▼                                                                          │
│   │    Graph（GetOrderlyOPList）──▶ OpEnv / Kernel ──▶ Backend Plugin ──▶ Adreno GPU     │
│   └────────────────────────────────────────────────────────────────────────────────────┘
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

架构要点：

1. **单一图表示、元数据旁路**：Graph 是全项目唯一可执行图表示，采用 C++17 runtime IR（Graph/Op/Tensor/Storage/GraphEnv/OpEnv），不依赖 torch/LibTorch；Python 侧只有 pybind11 暴露的最小构图接口。Build 阶段所需的 shape、precision、权重来源等元数据由 Converter 在构图时旁路记录为 `ModelGraph` sidecar，既不改动 Graph C++ 接口，也不维护第二套可执行图。
2. **离线优化、在线执行**：融合、精度转换、形状推导、显存复用全部在工具链完成；运行时只负责加载、调度、执行与释放。
3. **引擎分层**：Service（阶段二）→ Engine（阶段一）→ OpEnv/Kernel/Backend → GPU，上层不感知设备细节。
4. **可替换模块**：Loader、Scheduler、Executor、KVCache、Backend 均以接口注入 Engine，支持 mock、单测与后续后端扩展；Graph 作为冻结的图契约不随后端或模型替换。

## 3.2 分层与模块划分

| 层次 | 模块 | 主要职责 | 实现与目录 |
| --- | --- | --- | --- |
| 图与契约层 | 融合算子图（Graph） | C++17 runtime IR：Graph/Op/Tensor/Storage/GraphEnv/OpEnv；模型无关的 op_type 与 attr、Input/Output op、后序 DFS 拓扑排序、常量依赖叶子；pybind11 最小构图接口与 `dump_ir()` 调试输出 | C++/pybind11，`Graph/` |
| 离线工具链层 | Converter | 模型源码转换为 Graph，并旁路记录 ModelGraph | Python，`Build/converter/` |
| 离线工具链层 | Optimizer | 在 ModelGraph 上做精度推导、形状推导、显存编排、优化日志 | Python，`Build/optimizer/` |
| 离线工具链层 | Serializer | 把优化后的 ModelGraph 与权重序列化为 `model.mgf` 模型包 | Python，`Build/serializer/` |
| 运行时核心层 | Loader | 模型包加载、校验、Graph重建、后端选择与能力校验、GraphEnv/OpEnv 创建、权重转换/上传 | C++，`Runtime/loader/` |
| 运行时核心层 | Scheduler | 请求队列、Continuous Batching、调度计划 | C++，`Runtime/scheduler/` |
| 运行时核心层 | KVCache Manager | Paged KV Block 分配、共享与回收 | C++，`Runtime/kvcache/` |
| 运行时核心层 | Executor | Packing、主机 I/O 绑定、按 `GetOrderlyOPList()` 驱动 OpEnv（H2D/计算/D2H）、采样、结果回写 | C++，`Runtime/executor/` |
| 运行时核心层 | Engine | 统一门面、生命周期、推理循环、线程模型，持有 Graph 与 OpEnv | C++，`Runtime/engine/` |
| 运行时核心层 | Destroy | 资源有序释放（OpEnv、Kernel、Buffer、GraphEnv、Graph、Backend） | C++，`Runtime/destroy/` |
| 执行后端层 | Kernel 公共契约 | Kernel/KernelContext/KernelDesc、OpType/DataType 能力描述 | C++，`Runtime/kernel/` |
| 执行后端层 | Backend 框架 | Backend 抽象、后端注册表、运行时选择、能力校验 | C++，`Runtime/backend/`（公共部分） |
| 执行后端层 | OCL 后端（含 Kernel） | OCL 设备/内存/队列、Kernel 编译缓存、Qwen3 算子 Kernel、后端自注册 | C++/OCL，`Runtime/backend/ocl/` |
| 执行后端层 | 后端模板与契约测试 | 新后端骨架目录、统一一致性测试 | C++，`Runtime/backend/template/`、`Tests/backend_conformance/` |
| 服务层（阶段二） | Tokenizer 服务 | CPU 编解码、流式 Detokenize | C++，`Service/tokenizer/` |
| 服务层（阶段二） | PrefixCache | 前缀索引、块共享、引用计数与淘汰 | C++，`Service/prefix_cache/` |
| 服务层（阶段二） | Session | 会话、流式 API、取消与隔离 | C++，`Service/session/` |
| 支撑 | 日志/错误码/配置/测试 | 横切能力与测试框架 | C++/Python，`Runtime/common/`、`Tests/` |

### 3.2.1 仓库目录规划

```text
MobileNet/
├── Graph/           # C++17 runtime 静态图 IR 与 pybind11 构图绑定（权威图实现）
│   ├── .doc/              # Graph 唯一设计文档
│   ├── .skill/            # Graph 独立 skill
│   ├── include/            # Graph/Op/Tensor/Storage/Shape/GraphEnv/OpEnv
│   ├── src/               # 构图、排序、属性/存储/环境实现
│   ├── python/            # pybind11 绑定与最小 Python 构图 API
│   └── tests/             # Graph C++/Python 测试与 test_results.md
├── Build/                 # 离线工具链（Python）
│   ├── .skill/            # 阶段级 skill
│   ├── tests/test_results.md  # 阶段级测试结果
│   ├── graph/             # Graph 构建产物（_graph 原生扩展）
│   ├── common/            # ModelGraph、safetensors、JSON、错误与哈希工具
│   ├── converter/         # RecordingGraph、Qwen3 Adapter、转换报告
│   │   ├── .doc/          # converter_design.md
│   │   ├── .skill/        # converter skill
│   │   └── tests/         # 模块测试结果
│   ├── optimizer/         # Precision / Shape / Memory Pass
│   │   ├── .doc/          # optimizer_design.md
│   │   ├── .skill/        # optimizer skill
│   │   └── tests/         # 模块测试结果
│   ├── serializer/        # model.mgf 格式、序列化与重建 Loader
│   │   ├── .doc/          # serializer_design.md
│   │   ├── .skill/        # serializer skill
│   │   └── tests/         # 模块测试结果
│   ├── artifacts/         # Qwen3 模型包产物
│   ├── pipeline.py        # convert→optimize→serialize 串联入口
│   └── testing.py         # 测试用小型图与合成权重
├── Models/                # 模型组图设计与 Qwen3 配置/权重映射
│   ├── .doc/Models_design.md
│   ├── .skill/model-graph/SKILL.md
│   └── Qwen3/
│       └── tests/test_results.md
├── Runtime/               # 在线运行时（C++）
│   ├── loader/
│   ├── scheduler/
│   ├── kvcache/
│   ├── executor/
│   ├── engine/
│   ├── destroy/
│   ├── kernel/            # Kernel 公共契约：Kernel/KernelContext/KernelDesc、OpType/DataType
│   ├── backend/           # Backend 框架与后端插件目录
│   │   ├── backend.h/.cc  # Backend 抽象、配置、能力
│   │   ├── backend_registry.h/.cc  # 注册表与自注册宏
│   │   ├── ocl/           # OpenCL 后端：Backend 实现、kernels、自注册、CMakeLists
│   │   └── template/      # 新后端模板目录（复制后改名接入）
│   └── common/            # 错误码、日志、配置、工具
├── Service/               # 阶段二服务框架
│   ├── tokenizer/
│   ├── prefix_cache/
│   └── session/
├── Tests/                 # 单元、集成、基准、金标准数据、后端契约测试
└── Docs/                  # 通用文档、知识库与历史资料
```

## 3.3 离线工具链流程

工具链主要实现以下功能流程：

1. **Converter**：读取模型目录（`config.json`、Tokenizer 配置、可选 safetensors 权重），按模型族 Adapter 构建 Graph；构图 API 由 `RecordingGraph` 透明代理，在创建输入、常量与 op 的同时旁路记录 `ModelGraph`；常量记录 HF 权重名与可用性，权重字节不做数值变换；输出 Graph 对象、ModelGraph 与转换报告。
2. **Optimizer**：在 ModelGraph 深拷贝上按顺序执行 Precision Infer（含后续量化扩展）、Shape Infer、Memory Manage 三个 Pass；每个 Pass 输出结构化报告；任一 Pass 失败即终止并定位到张量/op。
3. **Serializer**：校验 ModelGraph 精度与形状完整；按 manifest 在内存中重建 Graph并计算 `dump_ir()` 的 SHA-256 与行数；写出 `model.mgf`、可选 `weights.bin` 与 `compile_report.json`；记录格式版本与校验和。
4. **产物归档**：记录模型 ID、配置哈希、工具链版本与生成信息；manifest 保持确定性，时间戳等字段只写入 `compile_report.json`，不参与运行时校验。

工具链产物：

| 文件 | 内容 | 消费方 |
| --- | --- | --- |
| `model.mgf` | 112 B 固定头 + 规范化 JSON manifest：Graph/Op/Tensor 元数据、显存计划、权重表、模型元信息、格式版本与 SHA-256 | Loader |
| `weights.bin`（可选） | 可用权重二进制，按 256 B 对齐，含偏移、dtype/shape 与逐项 SHA-256 | Loader |
| `compile_report.json` | 生成时间、优化摘要、结构统计、显存计划、权重可用性与文件哈希 | 人工/Agent 审查 |
| Tokenizer 文件 | 模型自带的 tokenizer 配置与词表 | 阶段二 Tokenizer 服务 |
| 工作目录中间产物 | `converter/model_graph.json`、`converter/conversion_report.json`、`optimizer/model_graph.json`、`optimizer/optimize_report.json` | 调试、审查与回归 |

## 3.4 在线推理流程

### 3.4.1 请求生命周期

1. 文本经 Tokenizer 编码为 Token ID（阶段一可由测试程序直接提供 Token ID）。
2. Session 或调用方构造 Request，通过 Engine 转发 Scheduler.Merge 进入 Waiting 队列。
3. 推理循环中，Scheduler.Schedule 生成 ExecutionPlan；Executor 绑定主机输入输出，通过 Engine 把 host/device storage 写入 Input/Output op 的 attr，再按 `Graph::GetOrderlyOPList()` 逐 op 调用 `OpEnv::Forward()` 完成 H2D、计算与 D2H，随后采样；Scheduler.Update 回写新 Token 与状态。
4. Prefill 完成后请求进入 Running；此后每个 Step 执行 Decode 生成一个 Token；满足 EOS、长度上限或 Stop 条件后结束并释放 KV。
5. 输出 Token 经流式 Detokenize 转为文本片段，回调给 Session/APP。
6. 取消、超时或异常时，请求进入 CANCELLED/FAILED，KV 与相关资源被回收。

```text
APP          Tokenizer        Scheduler        Executor          GPU
 │─ text ──────▶│                │                │                │
 │             │── token ids ──▶│                │                │
 │             │                │── plan ───────▶│                │
 │             │                │              │─ Graph/OpEnv 执行 ─────▶│
 │             │                │◀── Update ─────│◀── logits ─────│
 │             │◀── token ──────│                │                │
 │◀── stream ──│                │                │                │
```

### 3.4.2 连续批处理与 Paged Attention

- **迭代级组批**：每个 Step 重新决定本批请求；Waiting 中新请求与 Running 中解码请求可同批执行，避免长请求阻塞短请求。
- **双重预算**：Token 预算（MaxNumBatchedTokens）限制单 Step 打包 Token 数；KV 预算限制同时驻留的块数。
- **Prefill 优先**：优先满足 Waiting 请求的 Prefill，再为 Running 请求分配 Decode 名额；超长 Prompt 支持分块 Prefill（可选）。
- **Paged KV**：KV Cache 按固定大小 Block 管理，Block Table 维护逻辑块到物理块的映射；Attention Kernel 按块读取，降低碎片与扩容成本。
- **Token 维度 Packing**：本 Step 所有请求的 Token 拼接为统一 Batch，通过 attention metadata（序列长度、Block Table、Slot Mapping、阶段标记）区分请求。
- **抢占**：KV 不足时按策略选择牺牲者（首版采用重计算抢占：释放 KV、请求回到 Waiting、重新 Prefill），保证高优先级或先到请求推进。

### 3.4.3 PrefixCache（阶段二）

- **前缀索引**：按 Block 粒度的 Token 前缀哈希建立 Trie/哈希索引；节点记录物理块、引用计数与最近访问时间。
- **命中复用**：请求到达时查找最长公共前缀，复用对应 KV Block；仅对未命中后缀执行 Prefill。命中前缀的位置编码必须与缓存一致。
- **写时复制**：共享块只读；当最后一个不完整块需要追加时，先复制再写，避免污染其他请求。
- **淘汰**：引用计数为 0 的块进入 LRU 缓存池；显存压力时优先回收；模型、Tokenizer 或上下文配置不同则不共享。
- **可观测**：统计命中率、复用块数、节省的 Prefill Token 数，作为阶段二验收指标。

## 3.5 关键技术决策

| 决策点 | 选择 | 理由 | 代价与风险 |
| --- | --- | --- | --- |
| 执行形态 | 融合算子静态图 | 性能可预测、离线优化空间大、运行时轻量 | 自研图与 Kernel 成本高 |
| 语言分工 | 工具链 Python + 运行时 C++ | Build 迭代与文本处理效率高；端侧性能与包体可控 | 双语言一致性需 IR 哈希与契约测试保障 |
| 图表示与属性载体 | Graph C++17 runtime IR 单一表示；Build 旁路 `ModelGraph` 元数据 sidecar | 图语义与运行时代码同源，避免双表示漂移；Python 绑定最小化，C++ 运行时不链接 LibTorch | 需保证 RecordingGraph 记录与原生图同步；Python 侧元数据访问受限 |
| 序列化 | 自研 `model.mgf`（固定头 + canonical JSON manifest + 可选 `weights.bin`） | 无 protoc 依赖、可审查、可确定性生成、支持 Graph 重建校验 | 需自行维护格式版本与迁移；manifest 体积较大 |
| KV 管理 | Paged Attention | 低碎片、支持变长与并发、便于前缀共享 | 实现与调试复杂度高 |
| 调度 | Continuous Batching | 提高 GPU 利用率、降低尾延迟 | 组批与抢占逻辑复杂 |
| 精度 | FP16 主路径，关键累加 FP32 | 手机 GPU 吞吐与带宽友好 | 数值偏差需金标准对齐 |
| 量化位置 | 后续所有量化能力统一在 Precision Infer 阶段实现 | 精度转换与量化同属精度决策，集中管理精度表、量化参数与 Kernel 精度选择，避免分散到多个 Pass | Precision Infer 复杂度上升，需按模型/层配置策略并单独制定验收标准 |
| 设备后端 | OpenCL + 高通 Adreno 优先；其他后端按插件接入 | 目标设备支持成熟；插件机制保证后续扩展 | 厂商/驱动差异，需要设备矩阵 |
| 后端接入方式 | 静态目录级插件：统一 Backend/Kernel 接口 + 自注册 + 构建自动发现 | 新增后端只新增一个目录，图、调度、Executor、Engine 与 Service 零改动 | 注册对象可能被链接器裁剪，需 object library/whole-archive 与契约测试；Kernel 实现工作量不减少 |
| 后端资源封装 | 上下文、队列、显存、权重转换、Kernel 全部经 Backend 接口创建与释放；Loader/Destroy 只编排 | 上层零后端代码；释放顺序统一；插件边界清晰 | 需定义所有权、回滚与幂等规则 |
| 推理 I/O | Graph 的 Input/Output op + 对应 OpEnv；Engine 在执行前绑定 storage attr | 输入输出进入图契约，I/O 与计算统一调度；Graph 不感知设备 API | I/O storage 绑定与主机缓冲生命周期需严格管理；拷贝可能成为瓶颈 |
| 服务形态 | 进程内 C++ 服务 | 低延迟、离线、易集成 | 跨进程隔离与多语言绑定后续补充 |
| 模型格式 | 自研 `model.mgf` 模型包 | 可控、可版本化、可优化 | 需自行维护格式与迁移 |

## 3.6 设计假设

- 首期模型为 Decoder-only Dense Transformer（Qwen3 系列），不含 MoE、状态空间等结构。
- 模型权重与 Tokenizer 文件可合法获取，并以本地模型包形式随 APP 或测试工具分发。
- 目标设备支持 FP16 与 OpenCL，驱动行为稳定；具体机型在阶段一启动时锁定并记录。
- 引擎以单进程、单设备方式运行；多进程、多设备、多模型并行不在首期范围。
- 阶段一请求可以 Token ID 形式进入引擎；文本编码由阶段二 Tokenizer 服务负责。
- 推理输入/输出张量经 Graph 的 Input/Output op 与对应 OpEnv 完成 H2D/D2H；Engine 在执行前绑定 storage attr，主机缓冲由 Executor 提供并在执行期间有效。
- 引擎不处理模型版权、加密、用户鉴权与内容安全策略。

## 3.7 模块清单与设计文档索引

| 序号 | 设计文档 | 覆盖模块 | 主要依赖 | 建议顺序 |
| --- | --- | --- | --- | --- |
| 1 | [Graph 设计文档](Graph/.doc/Graph_design.md) | C++17 runtime IR（Graph/Op/Tensor/Storage/GraphEnv/OpEnv）、Input/Output op、SortOp、pybind11 最小构图接口、dump_ir | 无 | 1 |
| 2 | [Converter 设计文档](Build/converter/.doc/converter_design.md) | Qwen3 Adapter、RecordingGraph、ModelGraph sidecar、权重映射 | 1 | 2 |
| 3 | [Optimizer 设计文档](Build/optimizer/.doc/optimizer_design.md) | Precision/Shape/Memory Pass、显存复用、优化报告 | 1、2 | 3 |
| 4 | [Serializer 设计文档](Build/serializer/.doc/serializer_design.md) | model.mgf 格式、manifest、权重打包、图重建校验 | 1、3 | 4 |
| 5 | Loader 与 Backend 框架设计文档 | 模型包校验、Graph 重建、GraphEnv/OpEnv 创建、后端注册与选择、I/O 绑定、能力校验、权重转换与上传、资源所有权 | 4 | 5 |
| 6 | OCL 后端与 Kernel 设计文档 | Backend 接口实现、Kernel 清单、编译缓存、精度策略、契约测试 | 1、5 | 6 |
| 7 | Executor 设计文档 | Packing、主机 I/O 绑定、attention metadata、按 GetOrderlyOPList 驱动 OpEnv（H2D/计算/D2H）、采样、回写 | 5、6 | 7 |
| 8 | Scheduler 与 KVCache 设计文档 | 连续批处理、分页、抢占、公平性 | 5 | 7（可与 7 并行） |
| 9 | Engine 与 Destroy 设计文档 | 门面、线程模型、生命周期、资源释放 | 7、8 | 8 |
| 10 | Service 设计文档 | Tokenizer、PrefixCache、Session、流式输出 | 9 | 9 |
| 11 | 测试与基准设计文档 | 金标准、用例、性能与稳定性基准 | 全部 | 随模块迭代 |

# 4. 详细设计

本章给出各模块的架构级详细设计，覆盖职责边界、输入输出、关键接口、设计要点与异常处理。模块内部的算法、数据结构细节与代码级接口，由对应的模块设计文档展开。

## 4.1 融合算子图（Graph）

**职责**：Graph 提供全项目唯一的 C++17 runtime 静态图 IR，承载模型拓扑、Op、Tensor、constant、图输入输出与索引；提供构图、图边界注册、`SortOp()` 拓扑排序与查询接口。Graph 只表达“图是什么”和“对象如何连接”，不直接执行 kernel、不申请显存、不推导 shape/precision/layout，也不提供序列化与 Forward。运行时由 Engine 为每个计算 op 创建 `OpEnv`，Forward 阶段按 `GetOrderlyOPList()` 逐 op 调用 `OpEnv::Forward()`。

**不负责**：算子设备实现（Kernel/Backend）、显存实际分配、请求调度与批处理决策、采样策略参数生成、shape/precision/layout 推导策略、模型包序列化。

| 对象 | 职责 | 关键字段 / 接口 |
| --- | --- | --- |
| Graph | 整图容器与图边界注册 | `name_`、`op_list_`、`constant_tensor_list_`、`input_op_list_`、`output_op_list_`、`op_map_`、`tensor_map_`、`constant_tensor_map_`、`graph_env_`；`CreateOp/CreateConstant/CreateInput/CreateOutput/SortOp/GetAllConstantTensor/GetAllInputOp/GetAllOutputOp/GetOrderlyOPList` |
| Op | 计算、输入、输出或 runtime 边界节点 | `name_`、`op_type_`（`OpType`）、`attr_`（`map<string, variant>`）、`src_tensors_`、`dst_tensors_`、`op_env_`；`SetOpEnv/GetOpEnv/ClearOpEnv` |
| Tensor | op 之间的数据依赖边 | `name_`、`src_ops_`、`dst_ops_`、`storage_`、`is_constant_` |
| Storage | 张量/常量/临时数据的物理存储描述 | `Shape shape`、`Precision precision`、`Layout layout`、`DeviceType device_type`、`offset`、`byte_size`、`data_ptr`、`reusable` |
| Shape | 逻辑维度与物理 pitch | `dims[kMaxShapeDim]`、`pitch[kMaxShapeDim]`、`max_dim`；`GetShapeSize/GetByteSize` |
| GraphEnv | graph 级共享环境资源 | 由 Engine 创建并 `SetGraphEnv()` 移交 Graph 持有，供各 `OpEnv` 共享 workspace storage |
| OpEnv | 挂在 Op 上的运行时执行环境 | `Init(GraphEnv*)`、`Forward()`、`NeedTempData()`；由 Engine 按 `OpType` 创建并 `Op::SetOpEnv()` 绑定 |

设计要点：

- **单一 runtime IR**：全项目只有 Graph 这一套可执行图表示，不存在 Graph IR/RuntimeGraph 双表示，也不存在 `TensorDesc`。Graph 用 `OpType`、`Storage`、`Shape` 与 `attr` 表达图语义；C++ 运行时不链接 LibTorch，Python 侧不出现 `torch.Tensor`。
- **模型无关**：Graph 是全部模型族共用的 runtime 图表达，模型差异由 `OpType` 与 `Op.attr` 表达；Qwen3-0.6B 是阶段一首个验证模型，不是 Graph 的适用范围边界。
- **构图接口**：Python 侧仅暴露 `Graph(name)`、`create_input`、`create_constant`、`create_op`、`create_output`、`sort_op`、`dump_ir`。`create_op(name, src_tensors, out_cnt, op_type, attr)` 创建普通计算 op 与输出 tensor；输出命名固定为 `out_cnt==1` 时 `<op>.out`，否则 `<op>.out.<i>`。
- **常量是依赖叶子**：`create_constant(name, dims, pitch, precision, layout, byte_size, data_ptr)` 创建 constant tensor，独立存入 `constant_tensor_map_`，不创建 producer op，不进入 `SortOp()` 的执行序列；Engine 初始化阶段通过 `GetAllConstantTensor()` 统一加载权重。
- **拓扑排序**：`SortOp()` 从 `output_op_list_` 出发，沿 tensor 的生产者依赖做后序 DFS，将 graph op 按拓扑顺序写入 `op_list_`；生产者先于消费者，constant 遇到即停，不要求 producer op。
- **I/O 边界**：`CreateInput(name)` 创建 Input op 并登记图输入；`CreateOutput(name, input)` 创建 Output op 并登记图输出，Output op 不产生新 tensor。Input/Output op 预置 `attr["storage"] = Storage{}`，Engine 在 Init/Forward 时通过 `SetInput()`/`SetOutput()` 覆盖该值，解绑时恢复默认值。
- **执行环境**：Graph 本身没有 `Forward()`。Engine 初始化阶段创建 `GraphEnv`，按 `OpType` 为每个计算 op 创建 `OpEnv` 并调用 `Op::SetOpEnv()`；Forward 阶段按 `GetOrderlyOPList()` 逐 op 调用 `OpEnv::Forward()`。`OpEnv` 从所属 op 的输入/输出 Tensor 的 `Storage` 读取 shape、precision、layout 与 data_ptr，不拥有 Op/Tensor，也不申请显存。
- **I/O 执行**：Input/Output op 不产生计算 Kernel；对应的 I/O `OpEnv` 读取 `attr["storage"]` 完成 H2D/D2H。首版 Graph 只提供 `OpEnv` 抽象与 `MegaAttentionOpEnv` 示例，I/O OpEnv 与真实算子 OpEnv 由后续引擎/后端实现接入。
- **融合表达**：融合体现为 `OpType` 与 `attr`（例如 `kLinearRmsNormRope` 等融合类型）；具体融合规则由模型组图文档与 Converter 决定，Graph 只承载融合结果，不自动识别或改写算子。
- **Python 绑定最小化**：`Tensor` 仅暴露 `name`、`is_constant`、`reusable`、`set_reusable`；不暴露 `Op`、`Storage`、`Shape` 访问器，也不提供 op 遍历接口。`dump_ir()` 是只读调试接口，输出 graph/inputs/constants/ops/outputs 文本，不作为 runtime 数据交换边界。
- **Build 元数据旁路**：由于 Python 绑定不暴露 shape/precision/storage，Converter 使用 `RecordingGraph` 代理构图 API，在调用原生 Graph 成功后同步记录 `ModelGraph`（输入、常量、激活、op 属性、权重来源、输出绑定）。`ModelGraph` 是可序列化的 sidecar，不是第二套可执行图；Optimizer 与 Serializer 只消费它，运行时仍以 Graph 重建的图为准。
- **图校验**：唯一命名、输出 tensor 有生产者、Input/Output 注册合法、constant 不参与执行序列、`SortOp()` 覆盖所有从输出可达的计算 op；非法构图由 Graph 返回错误，Python 侧以异常形式抛出。
- **版本化**：Graph IR 本身不携带 schema_version；模型包版本由 `model.mgf` 的 `format_version` 与 manifest 管理，运行时拒绝不兼容版本。
- **无 Dummy Run**：Memory Manage 直接基于 ModelGraph 的静态依赖与拓扑序做生命周期分析，不执行图；Graph 首版也不提供执行模拟。

```text
Host input ──▶ Input op（Engine 绑定 storage，I/O OpEnv 做 H2D）──▶ 设备张量
           ──▶ GetOrderlyOPList() 逐 op 调用 OpEnv::Forward() ──▶ 设备张量
           ──▶ Output op（I/O OpEnv 做 D2H）──▶ Host output
```

**异常**：`CreateOp/CreateConstant/CreateInput/CreateOutput/SortOp` 失败返回 `Status`，并携带 graph/op/tensor 名称与原因；重复名称、输出缺少生产者、非法 `OpType`、I/O 注册冲突均在构图或排序阶段拒绝；Python 侧以异常形式抛出。

## 4.2 工具链（Build）

工具链由 Converter、Optimizer、Serializer 三个模块组成，全部产出均为可审查、可复现的中间结果。

### 4.2.1 Converter

**职责**：把厂商模型源码转换为 Graph 静态图，并同步旁路记录 `ModelGraph`；保持计算语义、结构、算子属性与常量权重映射不变。

**输入**：模型目录（`config.json`、Tokenizer 配置、可选 safetensors 权重）、转换配置（模型族、上下文上限、权重目录）。

**输出**：`Graph.Graph`、`ModelGraph`（输入、常量、激活、op 属性、权重来源、输出绑定）、转换报告（结构映射、算子/常量统计、权重可用性）。

设计要点：

- **模型族 Adapter**：首期实现 Qwen3 Dense Decoder-only Adapter；模型结构差异隔离在 Adapter，公共构图逻辑复用。
- **构图方式**：以模型配置与层结构描述为基础，由 `Qwen3GraphBuilder` 通过注入的 `graph_factory` 显式调用 Graph 构图接口；不执行 PyTorch 模型、不做 TorchScript/图捕获。
- **RecordingGraph**：对 `create_input/create_constant/create_op/create_output/sort_op/dump_ir` 做透明代理；每个方法先转发原生 Graph，成功后再记录元数据，防止“图已创建但元数据缺失”。`ModelGraph` 不是第二套可执行图。
- **融合算子表达**：使用 Graph 的 `OpType`（如 `gather`、`linear`、`linear_swi_glu`、`rms_norm`、`rope`、`mega_attention`、`add`）与 `attr` 表达算子；Qwen3 首版以基础 op 组合为主，融合规则在模型组图文档中定义，Graph 只承载结果。
- **权重映射**：逐常量记录图内名称、HF/Transformers 参数名、shape、dtype、byte_size 与是否可用；转换期不读取或变换权重数值。
- **I/O op**：在模型子图外层统一生成 Input/Output op；Input 记录 `int64`、`row_major` 与符号形状 `[B,S]`，Output 绑定最终 logits 张量；主机指针与设备 storage 由运行时绑定，不进入转换产物。
- **不静默降级**：不支持的算子、缺失权重映射或非法属性直接报错并列出位置，禁止悄悄换成低效路径。

**异常**：模型结构不支持、常量缺少 HF 映射、权重文件损坏、属性非 JSON 安全、Graph 构图失败。

### 4.2.2 Optimizer

**职责**：在不改变数学语义的前提下，在 `ModelGraph` 深拷贝上完成精度、形状、显存三类优化，产出可由 Serializer 固化、可由 Loader 重建的图元数据。

**输入**：`ModelGraph`、`OptimizeConfig`（目标精度、累加精度、batch_size、max_num_tokens、context_length、alignment、复用开关）、模型配置元数据。

**输出**：优化后的 `ModelGraph`、`MemoryPlan`、JSON 优化报告（精度表、形状表、显存计划、不变量检查结果）。

优化 Pass：

1. **Precision Infer**：BF16 转 FP16；为每个激活与常量确定唯一精度，int64 输入保持不变；对 `rms_norm`、`linear`、`linear_swi_glu`、`mega_attention`、`rope` 写入 `accumulate_precision=float32`。拒绝改变元素大小的转换（如 FP32→FP16），因为本阶段不重排权重字节。后续量化能力统一在本阶段扩展，不新增独立 Pass。
2. **Shape Infer**：以 `B=batch_size`、`S=max_num_tokens` 为规划上界，按拓扑序逐 op 推导输出形状；校验 gather/linear/linear_swi_glu/rms_norm/rope/mega_attention/add 的形状契约与 GQA 比例；未知 op 或未解析符号直接报错。
3. **Memory Manage**：基于静态依赖计算每个 Activation 的 `birth` 与 `last_use`，按 256 B 对齐做 first-fit 复用；输出每个张量的 offset、size、生命周期与峰值；图输入不进入 arena，图输出标记为不可复用；不执行 Dummy Run，也不分配真实显存。
4. **Report**：输出精度、形状、显存偏移与生命周期摘要，并自动检查精度完整、形状完整、生命周期重叠区间不相交、偏移对齐、arena 覆盖峰值、常量字节一致等不变量；报告为 JSON，可被脚本稳定解析。

设计要点：

- Pass 可组合、可单独执行；失败时保留中间产物与定位信息。
- 所有优化变换必须可验证；涉及数值变化的 Pass 需提供关闭开关（如 `target_precision="keep"`）与对比结果。
- 报告字段稳定、可版本化，便于 Agent 脚本长期校验。
- **I/O 处理**：`ModelGraph.inputs` 是外部输入，不参与 Activation arena；`ModelGraph.outputs` 的绑定张量把末次使用下标延长到虚拟消费者 `N`，并强制 `reusable=false`。Memory Pass 要求所有 Activation 已完成静态形状推导，未解析时直接报错，不生成不完整计划。

**异常**：形状契约不满足、显存区间冲突、精度不支持、形状未解析（Memory Pass 独立运行时）、配置非法。

#### 量化扩展规划（后续版本）

- **统一入口**：量化策略配置、权重转换/校准、scale/zero_point/granularity 计算、量化后 dtype 表与校验日志全部由 Precision Infer 负责；Shape Infer 与 Memory Manage 只消费量化结果。
- **量化路线**：规划权重 Only 与权重+激活两条路线，支持逐 Tensor、逐通道与分组粒度；具体算法与校准数据在量化模块设计文档中定义。
- **Kernel 契约**：Precision Infer 输出每个 Tensor 的量化描述；Kernel 按 dtype/量化描述选择反量化或低精度计算路径；不支持的组合在工具链阶段报错，不推迟到运行时。
- **序列化预留**：`ModelGraph` 的 `quantization` 字段与权重表预留量化描述；Serializer 写入 scale/zero_point 与量化布局；运行时不理解量化算法，只按 `Storage` 精度与量化描述执行。
- **独立验收**：量化不改变 FP16 首期验收基线；量化模型单独制定数值对齐、性能与内存指标，并单独出具测试报告。
- **实施归属**：后续量化需求统一并入 Precision Infer 的设计文档与代码模块，不单列优化 Pass 或独立任务。

### 4.2.3 Serializer

**职责**：把优化后的 `ModelGraph`、显存计划与权重固化为运行时可直接加载的 `model.mgf` 模型包；Loader 按 manifest 重建 Graph并校验 IR 哈希。

**输入**：优化后的 `ModelGraph`、`MemoryPlan`、`OptimizeConfig`、可选 safetensors 权重路径、工具链版本。

**输出**：

- `model.mgf`：112 B 固定头 + 规范化 JSON manifest（Graph 元数据、显存计划、权重表、模型信息、格式版本、SHA-256）。
- `weights.bin`（可选）：可用权重原始字节，按 256 B 对齐，逐项记录 offset/size/SHA-256。
- `compile_report.json`：生成时间、优化摘要、结构统计、显存摘要、权重可用性与文件哈希，不参与运行时加载。

设计要点：

- **容器格式**：magic `MNETMGF\0`、容器 version、flags（bit0 表示有 `weights.bin`）、manifest offset/length/SHA-256、weights offset/length/SHA-256；Loader 先校验头部与边界，再解析 manifest。
- **manifest 内容**：直接嵌入 `ModelGraph.to_dict()`，附加 `ir_sha256` 与 `ir_line_count`；`ir_sha256` 由“按该 manifest 重建后的 Graph `dump_ir()`”计算，形成 Loader 自校验闭环。
- **权重布局**：只打包 `weight_available=true` 且 shape/dtype/byte_size 匹配的常量；不转换 dtype，BF16 字节原样写入；每个权重按 256 B 对齐并计算 SHA-256；不可用权重写入 `reason`。
- **完整性校验**：manifest 整体 SHA-256 + 权重文件 SHA-256 + 逐权重 SHA-256；任何截断、篡改或版本不兼容均由 Loader 拒绝。
- **I/O 绑定序列化**：manifest 记录 `inputs` 名称与 `outputs`（输出名、张量名）绑定；主机指针、设备 storage 不序列化，由 Engine 在运行时通过 `SetInput/SetOutput` 绑定。
- **图重建**：按常量创建顺序、输入顺序、op 创建顺序、输出绑定顺序重放 Graph 构图 API，最后 `sort_op()`；不能以拓扑序替代 op 创建顺序。
- **确定性**：canonical JSON（键排序、紧凑分隔符、UTF-8）保证同一输入与配置产生逐字节一致的 `model.mgf`；时间戳只写入 `compile_report.json`。
- **metadata-only**：无权重或权重不可用时仍生成可加载的包，不写 `weights.bin`，manifest 中逐项标记 `available=false`。

**异常**：模型图未优化、输出目录非空、权重路径不存在/损坏、manifest 或权重校验失败、图重建与 IR 哈希不一致、写出失败。

## 4.3 运行时（Runtime）

运行时由 Loader、Scheduler、KVCache Manager、Executor、Engine、Destroy 组成；Loader 按 `model.mgf` 重建 Graph，Engine 持有 Graph、GraphEnv 与各 op 的 OpEnv，Executor 按 `GetOrderlyOPList()` 驱动 OpEnv 执行；Engine 聚合其余模块并提供统一门面。Graph 不链接 LibTorch。

### 4.3.1 Loader

**职责**：运行时加载入口，读取并校验模型包，重建 Graph与执行环境，并完成所选后端的初始化。

**输入**：`model.mgf`、可选 `weights.bin`、运行时配置（backend、device_id、queue_count、memory_budget、Kernel 编译选项）。

**输出**：Graph `Graph`、`GraphEnv`、各计算 op 的 `OpEnv`、Backend 实例、权重常驻缓冲、输入/输出设备缓冲、Activation 池与 KV 池。

流程：

1. 读取 `model.mgf` 固定头与 manifest，校验 magic、`format_version`、边界与 SHA-256；有 `weights.bin` 时校验其长度与 SHA-256。
2. 解析 manifest 中的 `ModelGraph`，按常量、输入、op、输出的记录顺序调用 Graph 构图 API，最后 `sort_op()`；计算 `dump_ir()` 的 SHA-256 并与 manifest 的 `graph.ir_sha256` 比对。
3. 按 `RuntimeConfig.backend` 从 BackendRegistry 创建后端，调用 `Backend::Init`；平台、设备、Context、Queue 由后端内部创建。
4. 读取 `BackendCapability`，校验 dtype、算子、最小对齐、单缓冲上限、传输方向（H2D/D2H）与 MemoryPlan。
5. 逐权重调用 `Backend::ConvertParam`，由后端完成转换并创建权重 Buffer；`GetAllConstantTensor()` 提供全部常量。
6. 按 MemoryPlan 通过 `Backend::CreateBuffer` 创建 Activation、KV、Workspace 与输入/输出设备缓冲。
7. 为 Graph 创建 `GraphEnv` 并调用 `Graph::SetGraphEnv()`；按 `OpType` 为每个计算 op 创建 `OpEnv`，调用 `Op::SetOpEnv()`；Input/Output op 不创建计算 Kernel，只创建 I/O OpEnv 或由 Engine 直接绑定 storage。
8. 可选 Warmup：执行空跑或小批量，完成 Kernel 预热与缓存。

设计要点：

- **零设备 API**：Loader 不直接调用 OpenCL/Vulkan/CUDA API；上下文、显存、权重转换、Kernel/OpEnv 创建全部经 Backend 或 Graph 接口。
- **显存计划与对齐**：Memory Manage 以统一对齐（默认 256 B，待真机校准）排版；Loader 用 `BackendCapability.min_alignment` 校验，不满足则加载失败，不做运行时静默重排。
- **后端解耦**：同一模型包可加载到不同后端；后端选择由 `RuntimeConfig.backend` 决定；未知后端、缺失算子或不支持精度在加载期返回明确错误码。
- **I/O 缓冲**：Input/Output op 不创建计算 Kernel；Loader 为 I/O storage 分配设备缓冲，并通过 `BackendCapability.copy_directions` 校验 H2D/D2H 能力；主机缓冲由 Executor 持有，不属于 Loader。
- **IR 闭环**：重建图后必须通过 `dump_ir()` 哈希校验；失败说明 manifest 与 Graph 构图语义不一致，拒绝加载。
- **失败回滚**：任一步失败按逆序调用 `Op::ClearOpEnv()`、`Backend::DestroyBuffer`、`Graph::ClearGraphEnv()`、`Backend::Deinit`；`Backend::Init` 自身保证失败时清理内部资源。

**异常**：文件损坏、版本不兼容、IR 哈希不一致、不支持的后端、设备初始化失败、Kernel 编译失败、显存不足、缺少 H2D/D2H 传输能力。

### 4.3.2 Scheduler

**职责**：请求级调度，决定每个 Step 执行哪些请求、哪些 Token，并维护请求状态。

**输入**：Request（prompt token ids、max_new_tokens、采样参数、优先级、到达时间）、KV 预算、Token 预算。

**输出**：ExecutionPlan。

接口：

- `Merge(request) -> request_id`：加入 Waiting 队列。
- `Schedule() -> ExecutionPlan`：组批与阶段划分。
- `Update(request_id, tokens, status)`：回写 Decode 结果与状态。
- `Stop(request_id)`：取消请求并释放资源。

策略：

- **Continuous Batching**：每个 Step 重新组批；Waiting 与 Running 可同批；Token 预算与 KV 预算双重约束。
- **Prefill 优先**：Waiting 请求优先 Prefill；Decode 按轮转或到达顺序推进，避免饿死。
- **分块 Prefill（可选）**：超长 Prompt 分块执行，避免单步阻塞已有请求的 Decode。
- **抢占**：KV 不足时选择牺牲者（低优先级/最新请求优先），首版采用重计算抢占，请求回到 Waiting。
- **停止条件**：EOS、max_new_tokens、Stop Token、调用方取消、超时。

**异常**：请求 ID 非法、状态非法、预算配置非法、KV 分配失败。

### 4.3.3 KVCache Manager

**职责**：管理 Paged KV Cache 的物理块、逻辑块表与共享。

设计要点：

- **物理块**：固定 Token 容量（首期建议 16 Token/块，可配置）；各层使用统一的块号空间、每层独立存储区，Block Table 统一管理。
- **逻辑块表**：`request -> [physical_block_id]`；随序列增长追加。
- **接口**：Allocate、Append、Free、Share（前缀共享）、RefCount/COW、统计（已用/空闲/碎片）。
- **共享与 COW**：PrefixCache 命中时增加引用计数；最后不完整块写入前复制。
- **回收**：引用计数为 0 的块进入空闲池或 LRU 缓存；OOM 时通知 Scheduler 抢占。
- **容量规划**：由显存预算、层数、KV heads、head_dim、dtype、block size 计算；启动时校验并锁定。

**异常**：块越界、重复释放、块表损坏、OOM、共享块非法写入。

### 4.3.4 Executor

**职责**：执行 ExecutionPlan，构建主机输入/输出绑定与 ForwardContext，按 Graph 拓扑序驱动 OpEnv，完成 H2D、图执行、D2H 与采样。

**输入**：ExecutionPlan、Graph `Graph`（非拥有）、Backend（非拥有）、KVCache 视图、主机输入/输出缓冲、采样参数。

**输出**：每个请求的新 Token（调试模式可输出 logits）、执行统计。

流程：

1. 构建主机输入：Token ID、Position ID、attention metadata（序列长度、Block Table、Slot Mapping、阶段标记）、逐请求采样参数；按输入名映射到 Graph 的 Input op。
2. 通过 Engine 的 `SetInput()`/`SetOutput()` 把 host/device storage 写入 Input/Output op 的 `attr["storage"]`；解码阶段的 KV 视图与 attention metadata 写入对应计算 op 的 attr 或 `OpEnv` 可访问的上下文字段。
3. 按 `Graph::GetOrderlyOPList()` 顺序逐 op 调用 `op->GetOpEnv()->Forward()`：Input op 的 I/O OpEnv 执行 H2D，计算 op 的 OpEnv 调用对应 Kernel，Output op 的 I/O OpEnv 执行 D2H；Prefill 与 Decode 使用对应 Attention Kernel。
4. 在 `Backend::Synchronize` 后读取主机输出，收集 Token、logits 与停止标记；调用采样逻辑生成 Token。
5. 回写 Scheduler.Update；释放 Step 临时资源；记录耗时、Token 数与显存占用。

设计要点：

- **Token 维度 Packing**：多请求 Token 拼接为统一 Batch，减少 Kernel Launch 与调度开销。
- **预分配缓冲**：主机输入/输出与设备缓冲均来自预分配池，避免每 Step 动态分配；同步点尽量少。
- **可观测**：调试模式可导出 logits 或中间张量，用于与 PyTorch 金标准对齐。
- **错误隔离**：单请求失败标记 FAILED，不阻塞其他请求；设备级错误上报 Engine 并触发受控停止。
- **Graph 驱动**：Executor 只依赖 `Graph::GetOrderlyOPList()`、`Op::GetOpEnv()` 与 `OpEnv::Forward()`，不包含后端专有头文件或条件分支；Graph 拓扑在 Load 后冻结，Executor 不增删 op。
- **执行环境所有权**：OpEnv 由 Op 通过 `unique_ptr` 持有，Executor 只借用；Kernel/Buffer 由 Backend 创建，OpEnv 通过 Backend 接口访问。
- **主机 I/O 所有权**：`host_inputs`/`host_outputs` 由 Executor 持有并在执行期间保持有效；Backend 不得在同步前释放或复用；异步传输时生命周期延续到 `Backend::Synchronize`。

**异常**：OpEnv::Forward 失败、OpEnv 缺失、I/O 绑定缺失、尺寸/对齐不匹配、H2D/D2H 传输失败、Kernel 执行失败、越界、采样参数非法、设备错误、Packing 元数据不一致。

### 4.3.5 Engine

**职责**：统一门面与生命周期管理，聚合 Loader、Scheduler、KVCache Manager、Executor、Destroy，并持有 Backend、Graph、GraphEnv 与各 op 的 OpEnv。

接口：

- `Load(config) -> Status`：加载模型，初始化后端、Graph 与运行时资源。
- `Schedule() -> ExecutionPlan`：驱动一次调度。
- `Execute(plan) -> Results`：执行计划并回写结果。
- `Destroy() -> Status`：释放全部资源。
- `SetInput(name, storage)` / `SetOutput(name, storage)`：绑定 Input/Output op 的 host/device storage。
- 转发 Scheduler 的 `Merge/Update/Stop` 给 Session 或调用方。

运行模型：

- **推理线程**：Engine 持有一个推理线程，循环 Schedule → Execute → Update；无活跃请求时休眠或退出（可配置）。
- **调用方线程**：Merge/Stop 线程安全；结果通过回调、事件或队列返回。
- **状态机**：CREATED → LOADED → RUNNING → DESTROYED；非法状态调用返回错误码。
- **模块注入**：各模块通过接口构造，支持 mock 与替换；Engine 不实现具体算法。
- **后端与图所有权**：Engine 持有 Backend 与 Graph；Loader 在 Load 时创建并移交；Op 通过 `unique_ptr<OpEnv>` 持有执行环境，Executor 仅持有非拥有指针；Destroy 时由 Engine 统一释放。
- **主机 I/O 缓冲所有权**：`host_inputs`/`host_outputs` 由 Executor 持有并在 Step 内复用；Engine、Graph 与 OpEnv 不持有主机缓冲。

**异常**：状态错误、线程启动/停止失败、模块初始化失败、推理循环异常退出。

### 4.3.6 Destroy

**职责**：按依赖顺序释放后端与主机资源，保证进程可安全退出。

释放顺序：

1. 停止调度循环，拒绝新请求。
2. 调用 `Backend::Synchronize` 等待在途 Kernel 完成。
3. 按 `GetOrderlyOPList()` 遍历计算 op，调用 `Op::ClearOpEnv()` 释放 OpEnv 与 Kernel/Program 资源（内部经 `Backend::DestroyKernel`）。
4. 逐缓冲调用 `Backend::DestroyBuffer` 释放 KV、Activation、权重、输入/输出与暂存缓冲。
5. 调用 `Graph::ClearGraphEnv()` 释放 GraphEnv 及其共享 workspace storage。
6. 调用 `Backend::Deinit` 释放 Program 缓存、Context、Queue 等后端内部资源。
7. 销毁 Graph 与 Backend 实例；关闭文件与线程，重置状态。

设计要点：

- **零设备 API**：Destroy 不直接调用任何设备释放函数；所有设备资源经 Backend 接口释放。
- **所有权与顺序**：先停调度、再同步、先 OpEnv 后 Buffer、再 GraphEnv，最后 Deinit；`Backend::Deinit` 作为最后兜底，释放残留资源并在调试构建中报告泄漏。
- **I/O op**：Input/Output op 不创建计算 Kernel；若存在 I/O OpEnv，随 `ClearOpEnv()` 释放；其设备缓冲随 Buffer 统一释放；主机缓冲由 Executor 释放。
- **幂等与异常安全**：重复 Destroy 不重复释放；单个释放失败记录错误并继续；最终返回首个错误码。
- **部分加载失败**：与正常 Destroy 使用同一逆序回滚路径，保证 Engine 可重试。

## 4.4 执行后端（Backend 插件框架）

### 4.4.1 设计目标与边界

**目标**：设备后端以目录级插件接入。新增 Vulkan、CUDA、CPU 等后端时，只新增 `Runtime/backend/<name>/` 一个目录，目录内包含 Backend 接口实现、H2D/D2H 传输实现、该后端全部 Kernel、自注册入口与 CMakeLists；推理配置指定后端名即可生效。Graph、Loader、Scheduler、Executor、Engine、Service 源码不做修改。

**边界**：

- 后端只负责设备、内存、Kernel 与权重预处理；不参与图优化、调度、KV 策略与采样。
- 上层只依赖 `Runtime/backend/` 与 `Runtime/kernel/` 的公共契约，不得包含后端专有头文件。
- 首期只实现 OCL；其他后端只验证插件机制，不在阶段一验收。
- 后端在编译期按目标平台选择并链接，不支持运行时热插拔。
- 公共契约变更需架构评审并更新本文档；单个后端目录不得修改公共接口。
- 插件机制降低接入成本，不减少 Kernel 实现工作量。

### 4.4.2 公共契约

| 层 | 内容 | 位置 |
| --- | --- | --- |
| Kernel 公共契约 | Kernel、KernelContext、KernelDesc、OpType/DataType | `Runtime/kernel/` |
| Backend 框架 | Backend、BackendConfig、BackendCapability、BackendRegistry | `Runtime/backend/` |
| 后端实现 | Backend 派生类（含 H2D/D2H）+ 该后端全部 Kernel + 自注册 | `Runtime/backend/<name>/` |

Backend 负责：初始化/释放设备与上下文、查询能力、创建/销毁 Buffer、主机-设备数据传输（Input H2D / Output D2H）、权重转换与上传、创建/销毁 Kernel、同步。`BackendCapability` 声明 supported_dtypes、supported_ops、copy_directions、max_buffer_bytes、min_alignment。Kernel 负责：单节点的 Prepare/Execute/Release。公共头文件不出现厂商类型；设备句柄均为不透明类型。接口签名由《Loader 与 Backend 框架设计文档》《OCL 后端与 Kernel 设计文档》冻结。

### 4.4.3 目录级插件规范

```text
Runtime/backend/
├── backend.h/.cc                   # Backend 抽象、配置、能力
├── backend_registry.h/.cc          # 注册表与自注册宏
├── ocl/                            # OpenCL 后端（首期）
│   ├── ocl_backend.h/.cc           # Backend 接口实现
│   ├── kernels/                    # 该后端全部算子实现与 Kernel 源文件
│   ├── register_ocl.cc             # 自注册入口
│   └── CMakeLists.txt              # 目录自描述构建
├── vulkan/                         # 后续新增：只加本目录
├── cuda/                           # 后续新增
└── template/                       # 新后端骨架目录
```

每个后端目录必须提供：Backend 派生类、H2D/D2H 传输实现、全部 Kernel、自注册入口、CMakeLists。构建系统自动遍历 `Runtime/backend/*/CMakeLists.txt`，新增目录无需修改根构建；每个后端以 object library 链接，或用 whole-archive 防止自注册对象被裁剪。后端开关为 `MOBILENET_BACKEND_<NAME>`，首期仅 OCL 默认开启。

### 4.4.4 注册、选择与执行

```text
Engine.Load(config)
  → BackendRegistry.Create(config.backend)
  → Backend::Init
  → Loader: 读取 model.mgf → 重建 Graph → ConvertParam / CreateKernel / CreateBuffer
  → Engine: 创建 GraphEnv 与各 op 的 OpEnv → Op::SetOpEnv
  → Engine 注入 Backend 与 Graph 到 Executor
  → Executor: for op in Graph.GetOrderlyOPList(): op.GetOpEnv()->Forward()
  → Backend::Synchronize
```

- **选择时机**：加载期；`RuntimeConfig.backend` 缺省 `"ocl"`；未知后端返回 `UNSUPPORTED_BACKEND`，不进入推理。
- **注册与调用路径**：BackendRegistry 只负责名称→工厂；Loader 创建后端并重建 Graph；Engine 创建 GraphEnv/OpEnv 并注入 Executor。Executor 通过 `GetOrderlyOPList()` 与 `OpEnv::Forward()` 触发后端执行，不持有注册表、不感知后端类型。
- **能力校验**：逐计算 op 创建 OpEnv/Kernel；I/O op 校验传输方向；缺算子、缺精度、缺传输方向、对齐不满足在加载期失败并定位 op，不做静默降级。
- **多实例**：每个 Engine 一个 Backend 与 Graph；`device_id` 预留多设备选择。

### 4.4.5 OCL 后端（首期）

- 封装 Platform、Device、Context、Queue、Buffer、Program、Kernel；全部隐藏在 `Runtime/backend/ocl/`。
- Kernel 优先使用预编译二进制或缓存，缓存键包含源码哈希、编译选项与驱动版本；运行时编译作为回退。
- 设备缓冲分配、对齐、上传/下载；权重常驻，Activation 复用；workgroup/local memory/向量化按 Adreno 调优。
- 公共头不暴露 OpenCL 类型。

### 4.4.6 新增后端接入流程

1. 复制 `Runtime/backend/template/` 为 `Runtime/backend/<name>/`。
2. 实现 Backend 接口（含 `Upload`/`Download` H2D/D2H）并注册后端名。
3. 在 `kernels/` 实现目标模型所需算子并注册 Kernel。
4. 打开 `MOBILENET_BACKEND_<NAME>`，构建系统自动发现并编译。
5. 运行后端契约测试（缓冲与 H2D/D2H 往返、逐算子数值、错误路径、生命周期）。
6. 推理配置指定 `backend=<name>`，加载并执行；检查上层源码零改动。

`Runtime/backend/mock/` 仅用于 TC-COMPAT-01 验证插件机制，不进入产品配置。

### 4.4.7 资源封装与释放

- Loader/Destroy 只做编排，不直接调用设备 API；上下文、显存、H2D/D2H、权重转换、Kernel、同步全部经 Backend。
- Backend 实例与 Graph 由 Loader 创建并移交 Engine；Op 持有 `OpEnv`，OpEnv 持有 Kernel 执行环境；Executor 持有 Backend/Graph 非拥有指针；Buffer/Kernel 句柄由 Backend 创建，运行时对象通过 Backend 释放。
- Input/Output op 的设备缓冲由 Loader 创建、Destroy 释放；主机输入/输出缓冲由 Executor 持有，Backend 只在执行期间使用，不保留指针。
- 释放顺序：停止调度 → Synchronize → 逐 op `ClearOpEnv()` → DestroyBuffer → `Graph::ClearGraphEnv()` → Deinit → 销毁 Graph 与 Backend。
- `Backend::Init` 全有或全无；`Deinit` 幂等；部分加载失败与正常 Destroy 使用同一逆序回滚路径。
- Backend 调用由 Engine 串行化，不要求内部线程安全。

### 4.4.8 异常与错误处理

- 错误码：`UNSUPPORTED_BACKEND`、`UNSUPPORTED_OP`、`UNSUPPORTED_DTYPE`、`BACKEND_INIT_ERROR`、`BUFFER_ERROR`、`DEVICE_ERROR`、`KERNEL_ERROR`。
- 后端错误附带 backend 名、原生错误码与阶段（Init/CreateBuffer/Upload/Download/CreateKernel/Execute/Synchronize）。
- 缺少 H2D/D2H 传输方向在加载期返回 `UNSUPPORTED_OP`；传输执行失败返回 `BUFFER_ERROR` 或 `DEVICE_ERROR`。
- 设备丢失触发 Engine 受控停止；Destroy 仍保证幂等与安全释放。

## 4.5 服务框架（Service，阶段二）

### 4.5.1 Tokenizer 服务

- **实现位置**：CPU、C++ 实现，加载模型自带的 Tokenizer 配置（首期支持 BPE/`tokenizer.json`）。
- **接口**：`Encode(text) -> ids`、`Decode(ids) -> text`、`StreamingDecode(token) -> text_piece`。
- **流式处理**：缓冲不完整 UTF-8 多字节序列，保证输出片段不截断字符；处理特殊 Token。
- **线程模型**：词表只读、常驻内存；接口线程安全；支持批量与增量调用。
- **部署形态**：阶段二以进程内 CPU 服务形式提供；独立进程或跨语言服务作为扩展。

**异常**：词表缺失、非法 Token ID、非法 UTF-8 输入、配置版本不兼容。

### 4.5.2 PrefixCache 调度

- **索引结构**：Block 粒度前缀哈希 + Trie/哈希索引；节点记录 Token 序列哈希、物理块、引用计数、最近访问时间。
- **流程**：请求到达 → 最长前缀匹配 → 共享块引用计数加一 → 仅 Prefill 后缀 → 新块在产生后登记。
- **一致性条件**：仅同模型、同 Tokenizer、同上下文配置的请求可共享；位置编码必须与缓存前缀一致。
- **写时复制**：共享块只读；最后一个不完整块追加前复制。
- **淘汰**：LRU；引用计数为 0 且显存紧张时优先回收；统计命中率与节省的 Prefill Token 数。

**异常**：索引损坏、引用计数错误、共享块写入冲突、缓存容量不足。

### 4.5.3 Session 与流式输出

- **Session 对象**：会话 ID、历史 Token、采样参数、状态、输出回调。
- **接口**：CreateSession、Chat（同步/流式）、Cancel、DestroySession。
- **多轮对话**：历史 Token 作为新请求前缀，结合 PrefixCache 复用已有 KV。
- **流式输出**：每 Step 的 Token 经 StreamingDecoder 转为文本片段，通过回调或队列返回；提供背压策略与顺序保证。
- **并发与隔离**：多 Session 共享 Engine；请求级隔离；取消传播到 Scheduler；错误映射为会话错误。
- **上层对接**：阶段四 APP 通过 JNI/FFI 调用；本文只定义 C++ 接口，绑定层在阶段四设计。

**异常**：会话不存在、会话忙、取消竞态、回调异常、输出队列满。

## 4.6 核心数据结构与接口汇总

核心数据结构：

| 结构 | 关键字段 | 说明 |
| --- | --- | --- |
| Request | id、prompt_ids、max_new_tokens、sampling_params、priority、state、arrival_time | 一次生成请求 |
| ExecutionPlan | step_id、entries[]、token_budget_used | 单 Step 调度结果 |
| PlanEntry | request_id、phase（PREFILL/DECODE）、token_range、block_table、sampling_params | 请求在本 Step 的执行片段 |
| Graph | op_list、constant_tensor_list、input_op_list、output_op_list、op_map、tensor_map、constant_tensor_map、graph_env | 全项目唯一可执行图；提供构图、SortOp 与查询接口 |
| Op | name、op_type、attr、src_tensors、dst_tensors、op_env | 计算/输入/输出节点；OpEnv 由 Engine 创建并挂载 |
| Tensor / Storage | name、src_ops、dst_ops、is_constant；Storage（Shape、precision、layout、device_type、offset、byte_size、data_ptr、reusable） | 数据依赖边与物理存储描述 |
| ModelGraph | name、adapter、inputs、constants、outputs、tensors、ops、metadata | Build 阶段旁路记录的元数据 sidecar；Optimizer 输入、Serializer 序列化对象 |
| ForwardContext | host_inputs/host_outputs、kv_cache_view、attention_metadata、sampling_params、workspace、backend、stream | Executor 的 Step 上下文；host I/O 通过 Engine SetInput/SetOutput 绑定，backend/stream 为非拥有不透明句柄 |
| I/O Binding | input/output op 名、host_ptr、bytes、dtype、shape、stride | Executor 绑定 Input/Output op 的主机缓冲描述 |
| SamplingParams | temperature、top_k、top_p、repetition_penalty、seed、stop_ids | 逐请求采样参数 |
| ModelConfig | 模型族、层数、hidden、heads、kv_heads、head_dim、vocab、context_length、rope、dtype | 模型元信息，存放于 ModelGraph.metadata |
| ModelPackage | path、manifest、manifest_sha256、weights_path、weights_sha256 | `model.mgf` 加载结果；manifest 含图、显存计划与权重表 |
| BlockTable | request_id、logical_to_physical、length | KV 块映射 |
| RuntimeConfig | backend、device_id、queue_count、memory_budget、options | 运行时配置；backend 决定后端选择 |
| BackendCapability | supported_dtypes、supported_ops、copy_directions、max_buffer_bytes、min_alignment | 加载期能力校验；copy_directions 声明 H2D/D2H |
| BufferDesc | bytes、alignment、usage、memory_type | 后端内存分配描述 |
| MemoryPlan | alignment、arena_bytes、peak_bytes、constants_bytes、entries[] | Memory Manage 产物，覆盖 Activation 复用；Loader 按后端能力校验并分配 |

核心接口：

| 接口 | 所属模块 | 说明 |
| --- | --- | --- |
| `Load(config)` / `Destroy()` | Engine、Loader | 模型加载与资源释放 |
| `Merge(req)` / `Stop(id)` | Scheduler | 请求加入与取消 |
| `Schedule() -> ExecutionPlan` | Scheduler、Engine | 生成本 Step 调度计划 |
| `Update(id, tokens, status)` | Scheduler | 回写生成结果与状态 |
| `CreateOp/CreateConstant/CreateInput/CreateOutput/SortOp` | Graph | 构图、图边界注册与拓扑排序 |
| `GetOrderlyOPList/GetAllConstantTensor/GetAllInputOp/GetAllOutputOp` | Graph | 拓扑序列、常量与 I/O 查询 |
| `SetInput/SetOutput` | Engine | 绑定 Input/Output op 的 host/device storage |
| `SetOpEnv/ClearOpEnv/GetOpEnv` | Op | 挂载、释放与访问执行环境 |
| `Forward() -> Status` | OpEnv | 单 op 执行入口；Executor 按拓扑序逐个调用 |
| `Execute(plan) -> Results` | Executor、Engine | 执行一个 Step；内部按 `GetOrderlyOPList()` 驱动 OpEnv |
| `Register/Create/List` | BackendRegistry | 后端自注册、按名称创建与枚举 |
| `Init/Deinit/Capability/CreateBuffer/Upload/Download/ConvertParam/CreateKernel/Synchronize` | Backend | 设备、内存、H2D/D2H、权重与 Kernel 生命周期 |
| `Prepare/Execute/Release` | Kernel | 单节点 Kernel 执行 |
| `Allocate/Append/Free/Share` | KVCache Manager | KV 块生命周期与共享 |
| `Encode/Decode/StreamingDecode` | Tokenizer 服务 | 文本与 Token 转换 |
| `CreateSession/Chat/Cancel/DestroySession` | Session 服务 | 会话与流式生成 |

## 4.7 状态机与错误处理

### 4.7.1 请求状态

| 状态 | 含义 | 迁移 |
| --- | --- | --- |
| WAITING | 等待调度与 Prefill | → PREFILLING、CANCELLED |
| PREFILLING | 正在执行 Prefill（可分多 Step） | → RUNNING、PREEMPTED、FAILED、CANCELLED |
| RUNNING | 已进入 Decode | → FINISHED、PREEMPTED、FAILED、CANCELLED |
| PREEMPTED | 被抢占，释放 KV 后回到等待 | → WAITING |
| FINISHED | 正常结束（EOS/长度/Stop） | 终态 |
| CANCELLED | 调用方取消或超时 | 终态 |
| FAILED | 执行异常 | 终态 |

```text
WAITING ──Schedule──▶ PREFILLING ──Prefill 完成──▶ RUNNING ──停止条件──▶ FINISHED
   ▲                       │                          │
   │                       │ 抢占                     │ 抢占
   └───────────────────────┴──────── PREEMPTED ◀──────┘

任意非终态 ──取消/超时──▶ CANCELLED
任意非终态 ──异常──────▶ FAILED
```

### 4.7.2 Engine 生命周期

```text
CREATED ──Load──▶ LOADED ──Run──▶ RUNNING ──Destroy──▶ DESTROYED
```

`LOADED` 表示 Backend 已初始化、Graph 已重建、GraphEnv/OpEnv、Kernel 与 Buffer（含 I/O 缓冲）已创建；Destroy 可从 `LOADED`/`RUNNING` 进入，按 4.3.6 顺序释放；Load 失败必须完整回滚，不得留下半初始化后端。

### 4.7.3 错误码

| 错误码 | 含义 | 典型场景 |
| --- | --- | --- |
| OK | 成功 | 正常返回 |
| INVALID_ARGUMENT | 参数非法 | 请求字段缺失、采样参数越界 |
| INVALID_MODEL_FILE | 模型文件损坏 | 文件截断、SHA-256 不匹配、IR 哈希不一致 |
| VERSION_MISMATCH | 版本不兼容 | `model.mgf` 的 `format_version` 或容器 version 不支持 |
| DEVICE_ERROR | 设备异常 | 设备 API 调用失败、设备丢失 |
| DEVICE_OOM | 设备内存不足 | KV/Activation/权重分配失败 |
| KERNEL_ERROR | Kernel 失败 | 编译失败、执行错误 |
| UNSUPPORTED_BACKEND | 后端未注册或未编译 | 配置名错误、目标平台未启用 |
| UNSUPPORTED_OP | 后端缺少算子实现或 I/O 传输方向 | CreateKernel 不支持 op_type；缺少 H2D/D2H |
| UNSUPPORTED_DTYPE | 后端不支持该精度 | FP16/量化 dtype 不在能力集合 |
| BACKEND_INIT_ERROR | 后端初始化失败 | Context/Queue/Program 创建失败 |
| BUFFER_ERROR | 后端缓冲操作失败 | 分配、对齐、H2D/D2H 上传/下载、释放失败 |
| REQUEST_NOT_FOUND | 请求不存在 | Update/Stop 非法 ID |
| STATE_ERROR | 状态非法 | 未 Load 就 Schedule、Destroy 后调用 |
| TIMEOUT | 超时 | 请求超时、设备等待超时 |
| CANCELLED | 已取消 | 调用方主动取消 |
| INTERNAL | 内部错误 | 不变量破坏、未分类异常 |

错误处理原则：

- 对外接口统一返回错误码/Status；C++ 异常不得穿越 JNI/FFI 等外部边界。
- 错误信息携带模块、请求 ID、节点名或设备信息，便于定位。
- 资源分配失败必须回滚；单请求失败不影响引擎其他请求；设备级错误触发受控停止。
- 日志分级（DEBUG/INFO/WARN/ERROR）；工具链日志采用 JSONL，运行时输出 Step 级统计与请求级轨迹。
- 错误码与日志字段在模块设计文档中冻结，作为测试与 Agent 开发的契约。

## 4.8 性能与内存设计

### 4.8.1 性能目标（初始，待真机校准）

| 指标 | 场景 | 初始目标 | 说明 |
| --- | --- | --- | --- |
| Decode | Qwen3-0.6B FP16，上下文 2048，单请求 | ≥ 10 token/s | 带宽主导，待真机校准 |
| Prefill | 128 Token Prompt | ≥ 100 token/s | 计算主导，待真机校准 |
| 首 Token 延迟 | 128 Token Prompt | ≤ 2 s | 含 Prefill 与首步采样 |
| 并发吞吐 | 4 并发 | ≥ 单请求的 1.5 倍 | 连续批处理收益 |
| 峰值内存 | 上下文 2048，单会话 | ≤ 3 GB | 含权重、KV、Activation、I/O 缓冲、运行时 |

> 以上数字来源为项目一期需求估算，不是实测值；阶段一启动时在参考机上实测并锁定，锁定值与测量方法写入《测试与基准设计文档》。

### 4.8.2 内存预算

| 项目 | 估算 | 说明 |
| --- | --- | --- |
| 权重 | 约 1.2 GB | 0.6B 参数 × 2 Byte（FP16），以实际参数量为准 |
| KV Cache | 可配置预算 | 块数 × 块大小 × 层数 × KV 维度 × dtype |
| Activation | 以 Optimizer 报告为准 | 复用后峰值 |
| 输入/输出缓冲 | 少量 | 主机与设备 I/O 缓冲，按 MaxNumToken 与输出形状估算 |
| Kernel/临时缓冲 | 少量 | 预分配池，避免运行时碎片 |
| 目标峰值 | ≤ 3 GB | 单会话、上下文 2048，不含系统与 APP |

### 4.8.3 优化手段

- **图级**：算子融合、常量折叠、死代码消除、权重/激活精度优化（后续量化统一由 Precision Infer 实施）。
- **显存级**：Activation 生命周期复用、KV 分页、预分配缓冲池、权重常驻。
- **调度级**：Continuous Batching、Token 维度 Packing、Prefill 优先、分块 Prefill、公平调度。
- **执行级**：减少 Kernel Launch、异步提交、CPU-GPU 流水、按 Adreno 调优 workgroup 与向量化。
- **I/O 级**：Input H2D、Output D2H 与计算异步重叠；主机缓冲复用；减少同步点；按后端能力选择批量传输。
- **服务级**：Tokenizer/Detokenizer 与推理重叠、PrefixCache 复用、流式输出。

## 4.9 扩展性设计

- **新增模型**：实现新的模型族 Adapter 与组图逻辑；如模型引入新算子，则同步扩展 `OpType`、`OpEnv` 与 Kernel；Graph、Loader、Scheduler、Executor、Engine 接口不变；Converter 以插件方式注册。
- **新增后端（目录级插件）**：仅新增 `Runtime/backend/<name>/` 目录，包含 Backend 接口实现、H2D/D2H 传输实现、该后端全部 Kernel、自注册入口与 CMakeLists；构建自动发现，推理配置指定后端名即可。Graph、Loader、Scheduler、Executor、Engine、Service 无需修改；加载期通过能力校验发现缺失算子/精度/传输方向并报错；所有后端必须通过统一契约测试。后端在编译期选择并链接，不支持运行时热插拔。
- **量化能力**：后续所有量化能力统一在 Precision Infer 阶段实现（INT8/INT4、权重/激活量化、分组量化等），不新增独立 Pass；`ModelGraph.quantization`、`model.mgf` 权重表与 Graph `Precision` 枚举预留量化描述；Kernel 增加低精度与反量化路径。
- **多模态（阶段三）**：在 Graph 中增加编码器子图（新增 OpType/Op 与对应 OpEnv），向 LLM 注入 embedding；Service 层增加图片/语音输入通道；Engine 的请求结构预留多模态输入字段。
- **多模型/多实例**：每个 Engine 实例独立资源与线程；通过资源配额与调度策略隔离。
- **跨语言/跨进程**：阶段四通过 JNI/FFI 封装 C++ 接口；服务框架可扩展为独立进程或本地服务。
- **高级推理特性**：投机解码、MoE、动态批大小等以新增 OpType/OpEnv/Pass/调度策略的方式接入，不破坏既有接口。

# 5. 实施步骤

## 5.1 里程碑

| 里程碑 | 内容 | 完成标志 |
| --- | --- | --- |
| M0 基础设施 | 仓库骨架、Graph 构建、日志/错误码、测试框架 | 主机侧可编译、可跑单元测试 |
| M1 图与工具链 | T1-1 ~ T1-6 | Python 构图与 Graph IR 一致；Qwen3-0.6B 产出 model.mgf 与 compile_report.json；优化报告可校验 |
| M2 单请求推理 | T1-7 ~ T1-9 | 参考机加载模型，单请求 Prefill/Decode 跑通并与 PyTorch 对齐 |
| M3 多请求调度 | T1-10 ~ T1-11 | 连续批处理、Paged KV、抢占、Engine 生命周期可用 |
| M4 阶段一验收 | T1-12 ~ T1-13 | AC-1 ~ AC-4、AC-7 通过 |
| M5 阶段二验收 | T2-1 ~ T2-4 | AC-5 通过，服务框架可被 APP 调用 |

## 5.2 任务拆解

每个任务均按四件套交付（设计文档、代码、测试结果、skill）；下表“验收”列给出主要验收项，具体用例见第 6 章。

| 编号 | 任务 | 主要内容 | 依赖 | 验收 |
| --- | --- | --- | --- | --- |
| T1-1 | Graph 静态图 | C++17 runtime IR（Graph/Op/Tensor/Storage/GraphEnv/OpEnv）、Input/Output op、SortOp、pybind11 构图绑定、dump_ir、Python 侧构图工具 | M0 | TC-GRAPH-01、TC-GRAPH-02 |
| T1-2 | Converter | Qwen3-0.6B 模型 Adapter、RecordingGraph、ModelGraph sidecar、权重映射、转换报告 | T1-1 | TC-CONV-01 |
| T1-3 | Precision Infer | BF16→FP16、逐 Tensor 精度推导、敏感路径 FP32 累加策略、预留量化扩展接口 | T1-2 | TC-OPT-01 |
| T1-4 | Shape Infer | Config/MaxNumToken/ContextLength 驱动的全图形状推导与校验 | T1-2 | TC-OPT-01 |
| T1-5 | Memory Manage | Activation 静态生命周期、显存复用与排版、结构化日志 | T1-2、T1-4 | TC-OPT-02 |
| T1-6 | Serializer | model.mgf 格式、图/权重序列化、Graph 重建校验、版本与确定性产出 | T1-3、T1-4、T1-5 | TC-SER-01 |
| T1-7 | Loader | 模型包校验、Graph 重建、GraphEnv/OpEnv 创建、后端选择与能力校验、Kernel 准备、I/O 缓冲分配、权重转换/上传、资源池创建 | T1-6 | TC-LOAD-01 |
| T1-8 | Backend 插件框架与 OCL 后端 | Backend/Kernel 公共接口、Upload/Download 传输接口、注册表与自注册、目录自动发现构建、能力校验、OCL 设备/内存/队列、Kernel 编译缓存、Qwen3 算子 Kernel、模板目录、mock 后端与后端契约测试 | T1-1、T1-7 | TC-KERNEL-01、TC-COMPAT-01、TC-LOAD-01、TC-ENG-01 |
| T1-9 | Executor | Packing、attention metadata、主机 I/O 绑定、按 GetOrderlyOPList 驱动 OpEnv（H2D/计算/D2H）、采样、结果回写、调试导出 | T1-7、T1-8 | TC-EXEC-01 |
| T1-10 | Scheduler 与 KVCache | 连续批处理、分页 KV、Block Table、抢占、取消与超时 | T1-7 | TC-SCHED-01、TC-SCHED-02、TC-KV-01 |
| T1-11 | Engine 与 Destroy | 门面、推理线程、生命周期、模块注入、资源释放与泄漏检测 | T1-9、T1-10 | TC-ENG-01 |
| T1-12 | 端到端集成与性能调优 | Android 打包、真机部署、金标准对齐、性能/内存优化 | T1-11 | TC-PERF-01、TC-MEM-01、TC-STAB-01 |
| T1-13 | 阶段一验收 | 全量用例、长稳压测、四件套审计、验收报告 | T1-12 | AC-1 ~ AC-4、AC-7 |
| T2-1 | Tokenizer 服务 | CPU 编解码、流式 Detokenize、UTF-8 边界、特殊 Token | T1-13 | TC-TOK-01 |
| T2-2 | PrefixCache | 前缀索引、块共享、引用计数、COW、LRU 淘汰 | T1-10、T2-1 | TC-PREFIX-01 |
| T2-3 | Session 与流式输出 | 会话管理、多轮对话、流式回调、取消、CPU-GPU 流水 | T2-1、T2-2 | TC-SVC-01、TC-PIPE-01 |
| T2-4 | 阶段二验收 | 集成测试、性能与稳定性验证、四件套审计 | T2-3 | AC-5 |

> 规划说明：后续模型量化不单列任务或优化 Pass，统一并入 T1-3（Precision Infer）的设计文档与代码模块迭代；阶段一首期只交付 BF16→FP16 能力与量化扩展点，量化模型的验收单独立项。
>
> 后端插件说明：T1-8 先冻结 Backend/Kernel 公共接口、注册表与目录规范，再实现 OCL。后续 Vulkan/CUDA 单独立项，只新增 `Runtime/backend/<name>/` 目录；契约测试与模板目录复用，不修改上层。

## 5.3 依赖关系

- T1-1 是所有任务的前置。
- T1-2 依赖 T1-1；T1-3、T1-4、T1-5 依赖 T1-2；T1-6 依赖 T1-3、T1-4、T1-5。
- T1-7 依赖 T1-6；T1-8 依赖 T1-1、T1-7；T1-9 依赖 T1-7、T1-8；T1-1 冻结 Graph IR、SortOp、OpEnv 与 I/O 契约，T1-9 完成与 Backend/Kernel 的集成。
- T1-10 依赖 T1-7 与调度/KV 接口契约，可与 T1-8、T1-9 并行开发（使用 mock）。
- T1-11 依赖 T1-9、T1-10；T1-12、T1-13 顺序推进。
- T2-1 ~ T2-3 依赖阶段一 Engine 接口稳定；T2-2 依赖 T1-10 的 KV 共享接口。

```text
Graph ──▶ Converter ──▶ Optimizer ──▶ Serializer ──▶ Loader ──┬──▶ Backend/Kernel ──▶ Executor ──┐
                                                               └──────────────────────────────────┤
                                                                                                  ▼
                                                             Scheduler + KVCache ──▶ Engine + Destroy ──▶ Service
```

## 5.4 风险与应对

| 风险 | 影响 | 概率 | 应对 |
| --- | --- | --- | --- |
| Adreno OpenCL 性能或兼容性不足 | 性能不达标、设备差异 | 高 | 阶段一早期做 Kernel micro-benchmark；锁定参考机与驱动；准备回退实现；建立设备矩阵 |
| FP16 数值偏差导致 Token 不一致 | 输出质量与验收风险 | 中 | 关键累加 FP32；逐层对齐；金标准回归；阈值用真机数据校准 |
| 后续量化引入数值误差与 Kernel 适配复杂度 | 量化模型质量或性能不达标 | 中 | 量化统一在 Precision Infer 中逐层校准与误差评估；量化描述与 Kernel 精度路径解耦；量化验收单独立项 |
| 显存不足或碎片 | OOM、频繁抢占 | 中 | Paged KV、Activation 复用、预算控制、抢占策略、内存 profiling |
| Python 构图与 Graph IR 漂移 | 工具链与运行时不一致 | 中 | Graph C++ 单一事实源；ModelGraph 与 dump_ir 计数/IR 哈希一致性测试；CI 回归 |
| Kernel 开发量大 | 进度风险 | 高 | 按模型图优先级实现；先跑通再优化；复用融合规则；与调度并行开发 |
| 后端操作封装不彻底，Loader/Destroy 出现专有 API 或资源泄漏 | 新后端接入仍需改上层；Destroy 泄漏 | 中 | 公共头隔离检查；所有权与释放顺序规范；TC-LOAD-01/TC-ENG-01 覆盖 Init 失败与重复 Destroy；CI 禁止上层 include 厂商头 |
| H2D/D2H 拷贝或同步成为延迟瓶颈 | 首 Token 延迟与 Decode 吞吐不达标 | 中 | 主机缓冲复用；异步传输与计算重叠；减少同步点；按后端能力批量传输；性能基线覆盖 I/O 占比 |
| model.mgf 格式演进 | 旧模型无法加载 | 低 | format_version 与容器 version 分离；向后兼容；迁移工具与文档 |
| Qwen3 真实权重未入库 | 只能生成 metadata-only 包，无法验证权重字节与数值 | 高 | Serializer 支持 `weights_path`；权重到位后补打包与 SHA-256 测试；当前交付明确标注 metadata-only |
| Agent 误解接口或跳过测试 | 返工、质量风险 | 中 | 模块设计文档先行；接口冻结；契约测试；四件套评审门禁 |
| 真机/驱动资源不到位 | 测试阻塞 | 中 | 提前锁定参考机；准备备用设备；主机侧单测覆盖非设备逻辑 |
| 性能目标定得过高 | 验收延期 | 中 | 初始目标可校准；阶段一启动时锁定；区分必须项与优化项 |

## 5.5 Agent 开发协作方式

为支持“设计文档 → Agent 开发 → 测试验证 → skill”的流程，每个模块任务按以下方式交付：

1. **任务包**：模块设计文档、接口契约（头文件/数据结构/错误码）、测试设计、相关已有代码与 memory.md。
2. **开发顺序**：先实现接口与桩 → 单元测试 → 具体实现 → 集成测试 → 测试报告 → skill 沉淀。
3. **接口冻结**：模块对外接口在开发前冻结；变更必须先更新架构文档与模块设计文档，再改代码。
4. **目录边界**：Agent 只修改所属模块目录与对应测试；跨模块变更需架构评审。
5. **门禁**：测试失败不得宣称完成；不得删减用例或放宽断言；四件套不完整视为未完成。
6. **可观测交付**：模块必须提供结构化日志、错误码与性能统计，便于后续 Agent 与人工审查。

# 6. 测试与验证设计

## 6.1 测试策略

采用分层测试策略：

| 层次 | 范围 | 运行环境 | 目的 |
| --- | --- | --- | --- |
| 单元测试 | 图校验、序列化、Scheduler 逻辑、KVCache、Tokenizer、错误处理 | 主机 | 快速验证模块不变量与边界条件 |
| Kernel 测试 | 每个融合算子的数值与性能 | 参考机 | 与 PyTorch 逐算子对齐 |
| 集成测试 | 工具链 → 运行时 → Engine → Service | 主机 + 参考机 | 验证模块组合与端到端流程 |
| 数值对齐测试 | 单请求/多请求 logits 与 Token | 参考机 | 与 PyTorch 金标准比较 |
| 性能测试 | Decode/Prefill/首 Token/并发/内存 | 参考机 | 验证 NFR-1、NFR-2 |
| 稳定性测试 | 长时混合负载、取消、OOM、异常恢复 | 参考机 | 验证 NFR-5 |
| 验收测试 | AC-1 ~ AC-7 | 参考机 + 文档审计 | 阶段验收 |

测试原则：

- 每个功能需求至少对应一个用例；正常路径、边界条件、错误路径均需覆盖。
- 数值金标准为 PyTorch 参考实现；FP16 引擎与 FP32/BF16 参考比较时，阈值需在模块文档中明确。
- 测试结果必须记录环境、命令、输入、实际输出、结论、未覆盖项与风险，不得只写“测试通过”。
- 设备侧测试记录 SoC、GPU、驱动、系统版本与电源/温控状态。

## 6.2 测试环境

| 环境 | 配置 | 用途 |
| --- | --- | --- |
| 开发主机 | Windows 11 或 Linux；Python 3.10+；CMake 3.20+；MSVC/Clang；Graph 原生扩展 | 工具链、主机单测、图与序列化测试 |
| 交叉构建 | Android NDK（r26 或以上）；arm64-v8a；Release + 调试符号 | 运行时构建与打包 |
| 参考机 | 高通骁龙 8 系 Android 手机；Adreno GPU；OpenCL 1.2+；机型在阶段一启动时锁定 | Kernel、端到端、性能、稳定性测试 |
| 性能工具 | Snapdragon Profiler/Adreno Profiler、Perfetto、adb/logcat、系统内存工具 | GPU/内存/耗时分析 |
| 测试数据 | Qwen3-0.6B BF16 原始权重、Tokenizer 文件、PyTorch 参考脚本与金标准输出、固定 Prompt 集 | 数值对齐与回归 |

> 参考机型号、驱动版本与测试日期必须在测试报告中记录；不同设备的结果不得混用。

## 6.3 测试用例

| 用例编号 | 对应需求 | 场景与输入 | 方法与通过标准 |
| --- | --- | --- | --- |
| TC-GRAPH-01 | FR-1.1 | 构造合法图与多种非法图（重名、成环、悬空依赖、输出无生产者、I/O 注册非法） | 合法图通过构图与 SortOp；非法图被拒绝并定位到 op/tensor；生产者先于消费者；错误码正确 |
| TC-GRAPH-02 | FR-1.2 | Qwen3-0.6B 图通过 Python 构图并与 C++ `dump_ir()` 比较 | `dump_ir()` 逐字节一致；ModelGraph 的 op/tensor 计数与 IR 一致；Python 侧不存在 torch 属性 |
| TC-CONV-01 | FR-1.3、NFR-7 | Qwen3-0.6B 模型目录转换 | RecordingGraph 与原生 Graph 的 dump_ir 一致；op/常量/权重映射与参考模型一致；转换报告完整；不支持的算子被明确报告 |
| TC-OPT-01 | FR-1.4、FR-1.5、FR-1.7 | 运行 Optimizer，输入不同 MaxNumToken/ContextLength | 所有 Tensor 有 dtype 与 shape；报告可解析；自动不变量检查通过；边界配置下形状正确；量化描述字段可缺省且不影响 FP16 路径 |
| TC-OPT-02 | FR-1.6 | 静态生命周期显存编排；开启/关闭复用对比 | 活跃区间无重叠；arena ≥ 峰值；复用前后图结构与输出绑定一致 |
| TC-SER-01 | FR-1.8 | 序列化 → 加载 → 重建 Graph → 再序列化 | 图结构与属性一致；重建 IR 哈希与 manifest 一致；权重校验和一致；字节级确定性（除报告时间字段） |
| TC-LOAD-01 | FR-1.9 | 正常包、截断文件、错误 SHA-256、format_version 不兼容、IR 哈希不一致、后端 Init 失败、缓冲区分配失败、I/O 缓冲分配失败、后端缺少 H2D/D2H 能力 | 正常加载并重建 Graph；I/O 绑定正确；异常返回对应错误码；已创建资源按逆序回滚；不崩溃；无泄漏 |
| TC-KERNEL-01 | FR-1.12 | 每个后端运行统一契约测试（含 Upload/Download 往返）；每个 Kernel 独立输入，与 PyTorch 对应算子比较 | 契约测试全部通过；H2D/D2H 数据一致；误差在模块文档阈值内；micro-benchmark 记录；无资源泄漏 |
| TC-EXEC-01 | FR-1.12 | 10 条中英文 Prompt，贪心生成 64 Token；Input/Output op 的 storage 覆盖、主机缓冲绑定、H2D/D2H 往返 | 与 PyTorch 参考：Token 一致率 ≥ 95%；logits 余弦相似度 ≥ 0.99；H2D/D2H 数据一致 |
| TC-SCHED-01 | FR-1.10 | 2/4/8 并发请求，不同 Prompt 长度 | 全部完成；请求间无串扰；与单请求基线 Token 一致率 ≥ 95% |
| TC-SCHED-02 | FR-1.10 | Merge 后 Stop、超时、长请求与短请求混合 | 取消/超时及时生效；KV 与请求资源释放；引擎继续服务其他请求 |
| TC-KV-01 | FR-1.11 | 随机分配/追加/释放 10000 次；构造 OOM；前缀共享与 COW | 不变量成立；无越界/重复释放；OOM 触发抢占并可恢复；共享块不被污染 |
| TC-ENG-01 | FR-1.13、FR-1.14 | Load/Destroy 循环 100 次；非法状态调用；重复 Destroy；加载中途失败；I/O 缓冲泄漏检查 | 无内存/设备/线程/I/O 缓冲泄漏；后端句柄计数归零；错误码正确；Destroy 幂等；失败加载可完整回滚 |
| TC-ROBUST-01 | NFR-5 | 截断权重、错误设备、后端 Context 创建失败、Kernel 编译失败、H2D/D2H 传输失败、OOM | 返回错误码；不崩溃；日志可定位；资源可回收；可重试 |
| TC-PERF-01 | NFR-1 | 固定 Prompt 集；单请求与 4 并发；不同 Prompt 长度 | 达到锁定目标（初始目标见 NFR-1）；记录设备与环境 |
| TC-MEM-01 | NFR-2 | 上下文 2048、单会话、连续生成 | 峰值内存 ≤ 3 GB（初始目标）；给出内存分解（含 I/O 缓冲） |
| TC-STAB-01 | NFR-5 | 1 小时混合负载（并发、取消、长短请求） | 无崩溃；RSS 增长 ≤ 5%；无句柄/设备资源泄漏 |
| TC-TOK-01 | FR-2.1 | 中英文、Emoji、空串、特殊 Token、长文本 | 编码-解码往返 100% 一致；流式输出 UTF-8 边界正确 |
| TC-PREFIX-01 | FR-2.2 | 相同 512 Token 前缀 + 不同后缀；并发命中 | 命中输出与无缓存一致（Token 一致率 ≥ 99%）；Prefill 耗时下降 ≥ 30%（待校准） |
| TC-SVC-01 | FR-2.3 | 多会话并发、多轮对话、取消、回调异常 | 会话隔离；流式顺序正确；取消生效；异常不崩溃 |
| TC-PIPE-01 | FR-2.4 | 并发请求下采集时间线 | Tokenizer 与 GPU 执行可观测重叠；吞吐不低于串行基线 |
| TC-COMPAT-01 | NFR-3 | 新增 `Runtime/backend/mock/` 目录（仅实现模板算子、Upload/Download 与自注册）；检查上层 diff | 仅新增一个目录与配置即可加载并完成推理；Graph、Loader、Scheduler、Executor、Engine、Service 源码零改动；传输接口可用；缺失算子/精度/传输方向在加载期报错 |
| TC-DOC-01 | NFR-6 | 审计每个模块交付物 | 设计文档、代码、测试报告、skill 四件齐全；Google Style 与中文注释检查通过 |

## 6.4 通过标准

- **功能**：6.3 中所有用例通过；因客观条件无法执行的用例须记录原因、风险与替代验证，并经评审。
- **数值**：单请求贪心 64 Token 与 PyTorch 参考一致率 ≥ 95%；logits 余弦相似度 ≥ 0.99；逐 Kernel 误差阈值由模块设计文档定义并在实测后锁定。
- **性能**：达到 4.8 节锁定后的目标；相对上一基线性能回退不得超过 10%。
- **内存与稳定**：峰值内存 ≤ 3 GB（初始目标，可校准）；1 小时压测无崩溃、RSS 增长 ≤ 5%、无资源泄漏。
- **交付**：每模块四件套齐全；设计文档与实现一致；接口变更已同步更新文档。
- **缺陷**：无未关闭的阻断级或严重级缺陷；已知问题有记录、影响分析与规避方案。

## 6.5 回归与验收

- **金标准固化**：模型哈希、Tokenizer 版本、Prompt 集、参考 logits/Token、性能基线随代码版本归档；金标准变更需评审。
- **分层回归**：主机单测随每次提交运行；设备侧冒烟测试随每日构建运行；性能与稳定性测试每周或每个里程碑运行。
- **后端契约回归**：后端契约测试随每日构建运行；新增后端必须通过同一套契约测试，且上层源码零改动，方可进入集成测试。
- **性能基线**：记录 Decode/Prefill/首 Token/内存/并发吞吐；超过 10% 回退视为回归，必须定位或说明原因。
- **阶段验收**：阶段一按 AC-1 ~ AC-4、AC-7 逐项出具测试报告；阶段二按 AC-5 出具测试报告；AC-6 在每个模块交付时审计。
- **验收材料**：设计文档、源码、测试报告、skill 四件套；测试报告包含环境、命令、用例、实际输出、结论、未覆盖项与风险。
- **缺陷闭环**：验收前修复阻断与严重缺陷；一般缺陷需有明确计划与影响说明。

# 附录

## A 术语表

| 术语 | 说明 |
| --- | --- |
| 工具链（Build） | 离线模型转换、优化与序列化流程，主要由 Python 实现 |
| 运行时（Runtime） | 端侧模型加载、调度、执行与释放流程，主要由 C++ 实现 |
| Graph | 项目唯一可执行图表示：C++17 runtime 静态图 IR，提供 Graph/Op/Tensor/Storage/GraphEnv/OpEnv；不依赖 torch/LibTorch |
| Graph | 图容器，持有 op、tensor、常量、输入/输出注册与拓扑序列 |
| Op | 图中的一个计算或 I/O op，含 op_type、attr、src_tensors、dst_tensors 与 OpEnv |
| Tensor | op 之间的数据依赖边，内嵌 Storage；常量 tensor 是依赖叶子，不进入执行序列 |
| Storage / Shape | Tensor 的存储描述（precision、layout、device_type、offset、byte_size、data_ptr、reusable）与维度描述（dims、pitch） |
| OpType | 算子类型枚举，决定运行时创建哪种 OpEnv/Kernel |
| GraphEnv / OpEnv | 图级运行环境与单个 op 的执行环境；Engine 创建 OpEnv，`OpEnv::Forward()` 执行计算 |
| SortOp | 从图输出做后序 DFS 生成拓扑序，保证生产者先于消费者 |
| Input op / Output op | Graph 中负责图输入/输出边界的 op；`attr["storage"]` 由 Engine 的 SetInput/SetOutput 覆盖 |
| ModelGraph | Converter 旁路记录的非执行元数据 sidecar（tensor/op/权重来源/量化描述），供 Optimizer 与 Serializer 使用 |
| RecordingGraph | Converter 对 Graph 构图 API 的透明代理，构图同时记录 ModelGraph |
| model.mgf | 自定义模型包格式：112 字节固定头 + 规范化 JSON manifest + 可选独立 weights.bin |
| metadata-only | 未提供真实权重时生成的模型包：图与显存计划完整，权重表标记不可用 |
| 融合算子 | 多个原生算子合并后的 op，对应运行时一个 OpEnv/Kernel |
| 散算子 | 融合前的原子算子记录，用于调试与回退 |
| H2D / D2H | Host to Device / Device to Host 数据传输 |
| ForwardContext | Executor 使用的一次前向上下文，含 host_inputs/host_outputs、KV 视图、attention metadata、采样参数与后端句柄 |
| IR SHA-256 | 重建图 `dump_ir()` 文本的 SHA-256；manifest 记录并在加载期校验 |
| Prefill | 处理 Prompt 并生成首 Token、填充 KV Cache 的阶段 |
| Decode | 逐 Token 自回归生成阶段 |
| Continuous Batching | 迭代级动态组批，每个 Step 重新决定批内请求 |
| Paged Attention | 以固定大小 Block 管理 KV Cache 的分页注意力机制 |
| Block Table | 请求的逻辑 KV 块到物理块的映射表 |
| Prefix Cache | 复用相同 Token 前缀的 KV Cache，减少重复 Prefill |
| MaxNumToken | 单 Step 打包 Token 数的上限 |
| ContextLength | 单请求上下文长度上限 |
| 金标准（Golden） | PyTorch 参考实现产生的 logits/Token，用于数值对齐 |
| 量化（Quantization） | 将权重/激活从高精度映射到低精度（如 INT8/INT4）；后续所有量化能力统一由 Precision Infer 阶段处理 |
| Backend（后端） | 设备执行抽象，负责设备、内存、H2D/D2H 传输、Kernel 与权重预处理；首期为 OCL 后端 |
| Backend Registry | 后端名称到工厂的注册表；后端目录自注册，推理配置按名称选择 |
| Kernel | 某后端上单个融合算子的设备实现，实现统一 Prepare/Execute/Release 接口 |
| 目录级插件 | 新增后端只新增 `Runtime/backend/<name>/` 目录，不改上层代码的接入方式 |
| 契约测试 | 所有后端必须通过的统一测试套件，覆盖缓冲与 H2D/D2H 传输、数值、错误路径与生命周期 |
| OCL | OpenCL，首期设备后端，实现位于 `Runtime/backend/ocl/` |
| JNI/FFI | Java Native Interface / 外部函数接口，用于阶段四 APP 对接 |

## B 参考资料

1. Qwen3 模型卡与技术报告（Alibaba Qwen 团队）。
2. vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention（SOSP 2023）。
3. OpenCL Specification（Khronos Group）。
4. Graph 设计文档（`Graph/.doc/Graph_design.md`，项目内）。
5. safetensors 格式规范（Hugging Face）。
6. Google C++ Style Guide 与 Google Python Style Guide（中文版）。
7. pybind11 官方文档。
8. PyTorch 官方文档（仅作数值金标准参考；Build 与 Runtime 均不依赖 LibTorch）。
9. Android NDK 官方文档。
