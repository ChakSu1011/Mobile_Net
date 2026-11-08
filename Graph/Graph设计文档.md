# 融合算子图（Graph）设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 编写日期 | 2026-11-08 |
| 文档状态 | 设计评审稿 |
| 上位文档 | [`../MobileNet_推理引擎_设计文档.md`](../MobileNet_推理引擎_设计文档.md) |
| 模块目录 | `Graph/` |
| 适用范围 | 全阶段、全模型族的融合算子图表达；Qwen3-0.6B 为阶段一首个验证模型 |
| 需求来源 | 上位文档 FR-1.1、FR-1.2、FR-1.3、FR-1.4、FR-1.5、FR-1.8、FR-1.9、FR-1.12、FR-1.14 |

---

# 1. 背景与目标

## 1.1 背景

本项目的推理引擎分为离线工具链与在线运行时两大部分。离线侧由 Converter 把厂商模型转换为统一融合图（Graph IR），Optimizer 完成精度、形状与显存优化，Serializer 把优化后的图与权重写入模型文件；在线侧由 Loader 反序列化并构建运行时图（RuntimeGraph），Executor 按调度计划调用 `RuntimeGraph.Forward` 完成推理。

Graph 模块提供两个层次的图表示，二者共享同一套图语义：

- **Graph IR（构图表示）**：用于 Converter、Optimizer、Serializer。它不自己定义底层张量元素，全部用 PyTorch/LibTorch 元素组装：Python 侧属性是 `torch.Tensor`、`torch.dtype`、`torch.device`、`torch.Size`、`torch.layout`；C++ 侧属性是 `at::Tensor`、`at::ScalarType`、`at::Device`、`at::IntArrayRef`、`at::Layout`。C++ 是唯一实现，Python 通过 pybind11 与 LibTorch type caster 看到同一对象的 `torch.Tensor` 视图。
- **RuntimeGraph（运行时表示）**：用于 Loader、Executor、Destroy。它只包含轻量 `TensorDesc`、裸指针与 POD，不链接 LibTorch，供移动端加载、绑定 Kernel/arena、Forward 与释放。

Graph 是上述所有模块共同依赖的**唯一图契约**：

- 对 Converter：Graph IR 是构图目标，融合规则与节点属性必须落到 Graph IR 对象。
- 对 Optimizer：Graph IR 是精度、形状、显存三类 Pass 的操作对象。
- 对 Serializer：Graph IR 是序列化的语义来源，`ToTensorDesc` 是 Graph IR 到运行时描述的转换边界。
- 对 Loader：RuntimeGraph 是运行时重建的目标结构，负责把 Kernel 与设备缓冲绑定到节点。
- 对 Executor：RuntimeGraph 提供模型级 `Forward`，内部完成 Input H2D、计算节点执行与 Output D2H。
- 对 Python 工具链：Graph IR 通过 pybind11 暴露；Python 侧看到的是原生 `torch.Tensor` 属性，而不是自定义张量封装。

Graph 是**模型无关**的融合算子图表达：对象模型、校验规则、I/O 契约与 Forward 语义对所有模型族一致；模型差异通过 `ModelConfig`、`FusionRuleTable` 中按 `model_family` 组织的规则集以及 `Node.attributes` 表达，不改变 Graph 结构。Qwen3-0.6B 是阶段一的首个完整实例与验证模型，不是 Graph 的适用范围边界。

如果 Graph 的字段、校验规则、融合边界或 I/O 契约不统一，会出现“离线图能转换、在线加载失败”“Python 构图与 C++ 构图不一致”“融合节点与 Kernel 边界错位”等问题。因此本模块先冻结 Graph IR/RuntimeGraph 双表示、校验规则、融合规则表与 Forward 契约，再进入编码。

## 1.2 目标

| 编号 | 目标 |
| --- | --- |
| G1 | 定义 Graph IR 与 RuntimeGraph 两套对象模型：Graph IR 使用 torch/LibTorch 张量属性，RuntimeGraph 使用轻量 TensorDesc |
| G2 | 定义默认 Input/Output Node 及 H2D/D2H 数据搬运契约 |
| G3 | 定义可执行的图校验规则，非法图在构图/加载期被拒绝并定位到节点或边 |
| G4 | 固化模型无关的 Decoder Block 融合骨架与可扩展融合规则表；Qwen3-0.6B 规则集作为首个完整实例，作为 Converter 的构图依据 |
| G5 | 提供 `RuntimeGraph.Forward`，按拓扑序执行 Input H2D、计算 Node/Kernel、Output D2H |
| G6 | 提供运行时执行表绑定与释放接口，保证 Kernel 生命周期与 Backend 解耦 |
| G7 | Graph IR 不自定义底层张量元素，全部用 PyTorch/LibTorch 元素组装；Python 侧属性为 torch 对象，C++ 侧属性为 at::Tensor 等 LibTorch 对象 |
| G8 | 预留版本化、符号形状与量化描述扩展点，不破坏阶段一 FP16 路径 |
| G9 | Serializer 把 Graph IR 转为 TensorDesc 与权重字节流；RuntimeGraph 不链接 LibTorch，移动端运行时保持轻量 |
| G10 | 通过 pybind11 + LibTorch caster 暴露 Graph IR，保证 Python 的 `torch.Tensor` 与 C++ 的 `at::Tensor` 看到同一张量 |

## 1.3 非目标

- 不实现设备 Kernel、OpenCL/Vulkan/CUDA 后端或显存分配；这些属于 Backend 与 Kernel 模块。
- 不实现 Scheduler、Continuous Batching、KVCache 分配策略；Graph 只接收 `KVCacheView` 与 `AttentionMetadata` 视图。
- 不实现 Protobuf schema 与权重文件格式；Serializer 负责线格式，Graph 只定义语义字段与版本契约。
- 不实现 Precision Infer、Shape Infer、Memory Manage 的算法；Graph 只承载其产物字段。
- 不实现 Tokenizer、Session、PrefixCache 服务。
- 不实现具体模型族的源码解析、Adapter 与权重映射；Converter 负责识别模型结构并查询本模块的规则集。
- 不支持运行时动态增删节点、热更新图或热插拔后端；图在 `Finalize` 后不可变。
- 不实现 PyTorch 自动微分、训练或反向传播；Graph IR 只承载推理图，`at::Tensor` 的 `requires_grad` 必须为 false。
- 不把 LibTorch 引入移动端 RuntimeGraph；LibTorch 仅用于工具链构图与 Python 绑定。
- 阶段一不启用 INT8/INT4 量化执行；只保留 Graph IR 量化元数据与 `TensorDesc.quantization` 描述字段。

## 1.4 术语

| 术语 | 含义 |
| --- | --- |
| Graph IR | 构图侧的融合算子图；C++ 属性为 `at::Tensor` 等 LibTorch 对象，Python 属性为 `torch.Tensor` 等 PyTorch 对象 |
| RuntimeGraph | 运行时轻量图；由 Serializer/Loader 从 Graph IR 转换重建，只含 `TensorDesc`、裸指针与 POD，不链接 LibTorch |
| 计算 Node | 对应运行时一个 Kernel 的融合算子节点；不含 Input/Output Node |
| Input Node | 入度为 0 的默认数据搬运节点，负责主机到设备（H2D） |
| Output Node | 出度为 0 的默认数据搬运节点，负责设备到主机（D2H） |
| fused_ops | 融合节点内部包含的原始散算子有序列表，用于追溯与测试 |
| fusion_rule_id | 该融合节点命中的规则表条目编号 |
| model_family | 模型族标识（如 `qwen3`、`llama`）；`ModelConfig` 与 `FusionRuleTable` 以此索引模型相关配置与规则集 |
| kernel_key | 逻辑 Kernel 键；Backend 据此选择具体实现，不等于厂商 Kernel 名 |
| SymbolicShape | 与 `torch.Tensor` 具体示例形状并存的符号维元数据；供 Shape Infer 绑定运行边界 |
| 示例张量 | Graph IR 中承载具体 dtype/shape/stride/device 的 `at::Tensor`；符号维使用示例值创建 |
| TensorDesc | RuntimeGraph 的轻量张量描述；由 `at::Tensor` + `SymbolicShape` 转换得到 |
| arena | MemoryPlan 中的一段设备内存；RuntimeEdge 通过 `arena_id + memory_offset` 定位 |
| ForwardContext | Executor 传给 `RuntimeGraph.Forward` 的运行时上下文；全部为裸指针/POD |
| Canonical JSON | 用于双端比较与快照测试的确定性 JSON，不是线格式 |

---

# 2. 需求分析

## 2.1 功能需求

| 编号 | 需求描述 | 验收要点 | 追溯 |
| --- | --- | --- | --- |
| G-FR-01 | 提供 Graph IR 与 RuntimeGraph 两套对象模型：Graph IR 聚合 Graph/Node/Edge/WeightDesc/ModelConfig，张量属性为 `at::Tensor`（C++）/`torch.Tensor`（Python）；RuntimeGraph 聚合 RuntimeNode/RuntimeEdge/RuntimeWeightDesc/TensorDesc | 两套对象均可构造并通过只读访问器读取；Graph IR 为 move-only 且禁止拷贝；RuntimeGraph 可 move，且不含任何 torch 类型 | FR-1.1 |
| G-FR-02 | Graph IR 的 Node 包含 name、op_type、fused_ops、attributes、input_edges、output_edges、precision（`at::ScalarType`）、kernel_key、fusion_rule_id；I/O Node 额外包含 binding_name、copy_kind、host_tensor（`std::optional<at::Tensor>`）。RuntimeNode 对应字段用 `DataType` 与 `std::optional<TensorDesc>` 表达 | 两套 Node 字段可构造、校验、序列化；I/O Node 与计算 Node 的字段约束不同且被校验 | FR-1.1 |
| G-FR-03 | 每个图至少包含一个 Input Node 与一个 Output Node；Input Node 入度为 0、负责 H2D；Output Node 出度为 0、负责 D2H；支持多个 I/O Node，`binding_name` 全局唯一。Graph IR 的 host_tensor 必须是 CPU `at::Tensor`，RuntimeGraph 的 host_tensor 是 TensorDesc | 缺少 I/O Node、I/O 边方向错误、binding 重名、IR host_tensor 非 CPU 均被拒绝 | FR-1.1、FR-1.9 |
| G-FR-04 | 提供图校验 V1~V16：名称唯一、无环、拓扑有序、边生产者/消费者一致、dtype/shape 可推导、权重边有数据源、I/O 边合法、融合边界合法、版本兼容；V1~V16 对 Graph IR 与 RuntimeGraph 共用同一套规则，张量检查按表示分派；V17 仅用于 RuntimeGraph 的执行表与 arena 校验 | 每类非法图返回明确错误码并定位到节点/边；同一非法结构在 IR 与 RuntimeGraph 上返回同类错误 | FR-1.1 |
| G-FR-05 | 提供 `RuntimeGraph.Forward(const ForwardContext&) -> Status`，按拓扑序执行 Input Node（Upload）、计算 Node（Kernel）、Output Node（Download） | Mock Kernel 下执行顺序与拓扑序一致；错误携带节点名 | FR-1.1、FR-1.12 |
| G-FR-06 | 提供 `ForwardContext` 与 `IoBinding`，全部为裸指针/POD，不含 torch 类型；主机输入/输出按 `binding_name` 绑定；校验字节数、dtype、shape；主机缓冲在 Forward 期间及 Synchronize 前保持有效 | 缺失绑定、尺寸/精度不匹配返回 `INVALID_ARGUMENT`；H2D/D2H 往返一致 | FR-1.12 |
| G-FR-07 | LLM 生成图固定包含一个 Sampling Node；输入 logits 与逐请求采样参数，输出 Token ID；Sampling 不参与融合 | 含 LmHead/Sampling 的图缺少 Sampling、多个 Sampling、Sampling 位置错误均被拒绝 | FR-1.1 |
| G-FR-08 | 固化模型无关的 Decoder Block 融合骨架：归一化 → 投影+后处理 → attention → 残差 → 归一化 → gate/up+激活 → down+残差；每个计算 Node 只能落在一个阶段内，模型差异由规则集参数化 | 跨阶段融合被校验拒绝；不同模型族的规则集可枚举且互不干扰 | FR-1.3 |
| G-FR-09 | 提供按 `model_family` 索引、可扩展的融合规则表 schema；给出首个完整规则集 Qwen3-0.6B（阶段、node_op_type、fused_ops、输入输出、属性、kernel_key、适用条件） | Converter 产出的融合节点与规则表逐项一致；新增模型族只新增规则集与 Adapter，不改 Graph 结构 | FR-1.3 |
| G-FR-10 | Converter 通过 `FusionRuleTable` 查询规则并在 Graph IR 上记录 `fusion_rule_id` 与 `fused_ops`；规则不满足时显式报错，不静默降级 | 转换报告包含规则命中与未命中原因；非法规则返回错误 | FR-1.3 |
| G-FR-11 | 提供 `GraphBuilder`（构造 Graph IR）与 `RuntimeGraphBuilder`（构造 RuntimeGraph）：添加节点/边/权重、设置 I/O 节点、`Finalize` 完成校验与拓扑排序；两者的拓扑序确定且规则一致 | 相同输入构图产生相同 node/edge 顺序与拓扑序；IR 与 RuntimeGraph 的拓扑序一致 | FR-1.1 |
| G-FR-12 | RuntimeGraph 提供运行时绑定：`BindBackend`、`BindArena`/`BindArenas`、`BindKernel`、`ValidateExecution`；`Release` 逐计算 Node 调用 `Backend::DestroyKernel`，I/O Node 跳过，重复调用幂等 | Backend/arena/Kernel 绑定缺失返回 `STATE_ERROR`，Kernel 不支持返回 `UNSUPPORTED_OP`；Release 后执行表与 arena 绑定为空且可重复调用 | FR-1.9、FR-1.14 |
| G-FR-13 | Graph IR 与 RuntimeGraph 均带 `schema_version` 主/次版本；读取端拒绝主版本不兼容；两者均提供确定性 Canonical JSON 用于比较与快照 | 版本不兼容返回 `VERSION_MISMATCH`；同一图多次输出字节一致 | FR-1.8 |
| G-FR-14 | Graph IR 的张量使用具体示例形状承载于 `at::Tensor`，符号维由独立 `SymbolicShape` 元数据表达；RuntimeGraph 用 `TensorDesc.shape` 中的 `Dim` 表达同一符号信息；符号维由 Shape Infer 绑定到运行边界 | 示例形状与符号维一致；符号维可比较、可在 Canonical JSON 中稳定输出；IR→RuntimeGraph 转换后符号信息不丢失 | FR-1.5 |
| G-FR-15 | Graph IR 的 Edge/WeightDesc 与 RuntimeGraph 的 TensorDesc 均预留可选量化描述（storage dtype、scale、zero_point、粒度）；阶段一默认缺省且不影响 FP16 路径 | 缺省量化字段可正常构图；量化字段不影响校验与 Forward；转换后量化元数据保留 | FR-1.4 |
| G-FR-16 | 通过 pybind11 + LibTorch type caster 暴露 Graph IR 对象、GraphBuilder、Validator、FusionRuleTable；C++ 为唯一事实源，Python 侧属性直接是 `torch.Tensor`/`torch.dtype`；Python 与 C++ 构图结果逐项一致 | TC-GRAPH-02 通过；Python 侧不维护第二套图定义，也不定义自定义张量类 | FR-1.2 |
| G-FR-17 | 错误统一返回 `ErrorCode`/`Status`，消息携带模块、节点名、边名、binding_name、fusion_rule_id；C++ 异常不得穿越 Python/FFI 边界 | 错误码与上下文可被测试断言；无异常泄漏 | NFR-5 |
| G-FR-18 | 提供 Graph IR 与 RuntimeGraph 的统计与调试输出：节点数、边数、权重数、每类 op 数量、拓扑序、Canonical JSON | 统计与快照可复现；不进入运行时热路径 | NFR-4 |
| G-FR-19 | 提供 `ToTensorDesc` 及 Graph IR→RuntimeGraph 转换：`at::Tensor` + `SymbolicShape` → `TensorDesc`；序列化路径先 detach、转 CPU、转连续行主序，2-D 列主序权重转置；权重字节流与 TensorDesc 一并产出 | 转换后的 dtype/shape/stride/layout/device/offset/size_bytes 与归一化后张量一致；符号维与量化信息保留；转换过程不修改源 Graph IR | FR-1.8 |

## 2.2 非功能需求

| 编号 | 类别 | 要求 | 验证方式 |
| --- | --- | --- | --- |
| G-NFR-01 | 性能 | 校验复杂度 O(V+E)；Forward 稳态每计算节点主机侧调度开销目标 ≤ 2 μs（参考机待测）；稳态 Forward 不进行堆分配 | PT-GRAPH-001、PT-GRAPH-002 |
| G-NFR-02 | 内存 | 首个验证模型 Qwen3-0.6B 的 RuntimeGraph 图元数据（不含权重与运行时缓冲）目标 < 5 MB；拓扑序与执行表使用连续数组；Graph IR 元数据仅存在于工具链，不进入移动端常驻内存 | PT-GRAPH-003 |
| G-NFR-03 | 确定性 | 相同输入构图、校验、拓扑排序、IR→RuntimeGraph 转换与 Canonical JSON 输出完全一致 | UT-GRAPH-017 |
| G-NFR-04 | 可移植性 | RuntimeGraph 公共头文件不出现 ATen/torch/c10 及 OpenCL/Vulkan/CUDA 类型，只依赖 C++17 标准库与 Runtime 公共契约；Graph IR 头文件可包含 ATen，但仅工具链目标链接 LibTorch；移动端运行时二进制不得依赖 libtorch | 头文件审计、二进制依赖检查 |
| G-NFR-05 | 可测试性 | Graph IR 单测使用 LibTorch CPU 张量；RuntimeGraph 可在主机侧用 Mock Backend/Kernel 单测 Forward；无需真实 GPU | UT-GRAPH-012、IT-GRAPH-003 |
| G-NFR-06 | 工程规范 | 遵循 Google C++/Python Style；模块级、类级、函数级中文注释覆盖功能边界、输入输出与主要逻辑 | TC-DOC-01 |
| G-NFR-07 | 健壮性 | 非法图、空指针、越界 binding、重复绑定、重复释放均返回错误码，不崩溃；ASan/Valgrind 无泄漏 | UT-GRAPH-004~016、稳定性测试 |
| G-NFR-08 | 并发 | `Finalize` 后 Graph IR 与 RuntimeGraph 均只读；图不提供内部锁；Forward 的并发安全由 Engine/Executor 通过串行化与独立 Stream 保证 | 代码审计、并发测试 |

## 2.3 约束条件

1. 语言标准为 C++17；Python 绑定使用 pybind11；构建使用 CMake。Graph IR 额外依赖 LibTorch，其版本必须与 Python 侧 `torch` 的 ABI 版本一致；RuntimeGraph 不依赖 LibTorch。
2. 错误码与状态复用 `Runtime/common/` 的 `ErrorCode` 与 `Status`，Graph 不定义第二套错误码。
3. `OpType` 等图语义枚举由 `Runtime/kernel/` 公共契约定稿；Graph IR 的张量属性必须使用 `at::Tensor`、`at::ScalarType`、`at::Device`、`at::IntArrayRef`、`at::Layout`，不得自定义张量、dtype、device、layout 等底层元素；`DataType`、`Layout`、`DeviceType`、`TensorDesc` 只属于 RuntimeGraph 与序列化边界。
4. Graph 不依赖 Protobuf；序列化映射由 Serializer 模块完成，Graph IR 只提供只读访问器与字段语义，RuntimeGraph 由 Loader 从序列化数据重建。
5. Graph IR 公共头文件可以包含 ATen/torch 头文件，但只能被工具链目标引用；RuntimeGraph 公共头文件不得包含 `<ATen/...>`、`<torch/...>`、`c10/` 及厂商设备头文件；`Backend`、`Stream`、`Kernel` 仅以公共接口类型出现。
6. 节点名、边名、`binding_name` 限定为 ASCII `[A-Za-z0-9_.-]`，长度 1~128；名称在各自命名空间内唯一。
7. Graph IR 与 RuntimeGraph 在 `Finalize` 后结构不可变；不支持运行时增删节点或修改边。
8. 阶段一运行时精度为 FP16；BF16 权重的转换由 Precision Infer 负责，Graph IR 只记录目标 `at::ScalarType`，RuntimeGraph 只记录目标 `DataType`。
9. 主机缓冲由 Executor 持有；RuntimeGraph 只读取 `IoBinding`，不拥有主机指针。
10. 设备缓冲与 Kernel 句柄由 Backend 创建；RuntimeGraph 拥有 Kernel 句柄的释放职责，但不拥有 Backend。
11. Graph 不绑定具体模型族或模型结构；新增模型族通过新增 `FusionRule` 规则集与 Converter Adapter 接入，不得修改 Graph 对象模型、I/O 契约与 Forward 语义。
12. Graph IR 的张量只承载推理语义：`requires_grad` 必须为 false，禁止携带 autograd 图；Serializer 转换时对张量执行 detach、CPU 拷贝、连续化。
13. `at::Tensor` 只保存具体示例尺寸，符号维必须由 `SymbolicShape` 单独保存；不变量为 `tensor.dim() == symbolic_shape.dims.size()` 且每个常量维满足 `tensor.size(i) == dims[i].value`，权重张量必须全为常量维。
14. Python 侧不得维护第二套图定义或自定义张量封装；Python 的 `torch.Tensor` 必须直接来自 C++ `at::Tensor` 的 pybind11 转换。

## 2.4 验收标准

- 上位文档 TC-GRAPH-01、TC-GRAPH-02 全部通过。
- 本文 6.3、6.4 全部用例通过；校验规则 V1~V17 每条至少一个负向用例。
- 至少一个模型族图（阶段一为 Qwen3-0.6B）在 Python 与 C++ 两侧构造后，Canonical JSON 差异数为 0；Python 侧断言节点属性是 `torch.Tensor`/`torch.dtype`，不是自定义类型。
- Graph IR 经 `ToTensorDesc`/Serializer 转换为 RuntimeGraph 后，dtype、shape、stride、layout、quantization 与符号维逐项一致；转换前后拓扑序一致。
- RuntimeGraph 头文件审计无 ATen/torch/c10 引用；移动端运行时二进制的依赖清单无 libtorch。
- Mock Backend 下完成 Input H2D → 计算 → Output D2H 的完整 Forward，执行顺序与拓扑序一致。
- ASan/Valgrind 下无内存泄漏、无越界；重复 Bind/Release 不崩溃。
- 性能目标未达标时，必须给出实测数据、偏差原因与后续优化项，不得以“未覆盖”代替。

---

# 3. 概要设计

## 3.1 模块定位

```text
Build/Converter ──▶ GraphBuilder ──▶ Graph IR ──▶ Optimizer
                                        │
                                        ├──▶ Serializer（ToTensorDesc + 权重字节流）──▶ 模型文件
                                        │
Runtime/Loader ◀── 反序列化 / RuntimeGraphBuilder ─────┘
      │ RuntimeGraph + BindBackend / BindKernel
      ▼
   Executor ──▶ RuntimeGraph.Forward ──▶ Backend::Upload / Kernel::Execute / Backend::Download
                                        │
                                        └──▶ Backend Plugin（OCL/Vulkan/CUDA/...）
```

Graph 位于“图与契约层”，是离线工具链与在线运行时的共同依赖。Graph IR 供工具链使用，链接 LibTorch，属性直接使用 `at::Tensor`/`torch.Tensor`；RuntimeGraph 供移动端运行时使用，只含 `TensorDesc` 与裸指针，不链接 LibTorch。两者共享 `OpType`、`Dim`/`SymbolicShape`、`AttributeValue`、`FusionRule`、版本语义与拓扑规则。

Graph 不感知具体后端，也不感知调度策略；它只依赖 Kernel/Backend 的公共接口与 Runtime/common 的错误码。Graph 同样不绑定具体模型族：模型差异由 `ModelConfig` 与 `FusionRuleTable` 的 `model_family` 规则集表达，Qwen3-0.6B 是首个规则集实例。

## 3.2 组件划分

| 组件 | 职责 | 不负责 |
| --- | --- | --- |
| GraphCommon | 双表示共享类型：`NodeId`、`EdgeId`、`OpType`、`AttributeValue`、`Dim`、`SymbolicShape`、`QuantizationDesc` | 张量数据、设备执行 |
| GraphIRCore | Graph IR 数据结构、`at::Tensor` 访问器、拓扑序、权重表、I/O 节点索引 | 设备执行、显存分配 |
| RuntimeGraphCore | RuntimeGraph 数据结构、`TensorDesc` 访问器、拓扑序、权重表、I/O 节点索引 | LibTorch、设备执行、显存分配 |
| GraphBuilder | 构造 Graph IR：增量构图、名称校验、边引用解析、Finalize（V1~V16） | 模型源码解析、融合模式匹配 |
| RuntimeGraphBuilder | 从序列化描述构造 RuntimeGraph，复用同一套结构校验 | 解析模型文件字节流（Loader 负责） |
| GraphValidator | V1~V16 双表示共用校验；V17 RuntimeGraph 执行表/arena 校验；错误收集与定位 | 后端能力校验（Loader 负责） |
| FusionRuleTable | 融合规则数据、规则查询、规则版本 | 从模型源码识别算子模式（Converter 负责） |
| TensorDescConverter | `at::Tensor` + `SymbolicShape` → `TensorDesc`；权重 CPU 化、连续化、列主序转置 | 写模型文件（Serializer 负责） |
| RuntimeGraphForward | RuntimeGraph 的 Backend/Kernel 绑定、执行表、Forward、Release | 请求调度、采样停止条件 |
| GraphPython | pybind11 + LibTorch caster 绑定、Python 侧构图工具 | 运行时热路径 |
| GraphDebug | 统计、Canonical JSON、错误格式化 | 线格式序列化 |

### 3.2.1 源码布局

```text
Graph/
├── cpp/
│   ├── common/
│   │   └── include/graph/
│   │       └── graph_common.h        # NodeId、EdgeId、OpType、AttributeValue、Dim、SymbolicShape、QuantizationDesc
│   ├── ir/
│   │   ├── include/graph/
│   │   │   ├── graph_types.h         # Graph IR 类型别名、ModelConfig、RopeConfig
│   │   │   ├── graph.h               # Graph、Node、Edge、WeightDesc
│   │   │   ├── graph_builder.h       # GraphBuilder
│   │   │   ├── graph_validator.h     # ValidationIssue/ValidationReport（V1~V16）
│   │   │   ├── fusion_rule.h         # FusionRule、FusionRuleTable
│   │   │   └── tensor_desc.h         # ToTensorDesc、IR→RuntimeGraph 转换契约
│   │   └── src/
│   │       ├── graph.cc              # 访问器、Canonical JSON、Graph IR 生命周期
│   │       ├── graph_builder.cc      # 构图、边引用解析、拓扑排序
│   │       ├── graph_validator.cc    # V1~V16 校验
│   │       ├── fusion_rule.cc        # 融合规则表与查询（含 Qwen3 规则集）
│   │       └── tensor_desc.cc        # at::Tensor + SymbolicShape → TensorDesc
│   ├── runtime/
│   │   ├── include/graph/
│   │   │   ├── runtime_graph_types.h # DataType、Layout、DeviceType、TensorDesc
│   │   │   ├── runtime_graph.h       # RuntimeGraph、RuntimeNode/Edge/WeightDesc
│   │   │   ├── runtime_graph_builder.h
│   │   │   ├── runtime_validator.h   # V1~V17（RuntimeGraph 分派）
│   │   │   └── forward_context.h     # IoBinding、KVCacheView、AttentionMetadata
│   │   └── src/
│   │       ├── runtime_graph.cc      # 访问器、Canonical JSON、生命周期
│   │       ├── runtime_graph_builder.cc
│   │       ├── runtime_graph_validator.cc
│   │       └── runtime_graph_forward.cc  # Bind/ValidateExecution/Forward/Release
│   └── python/
│       ├── graph_python.cc           # pybind11 + <torch/extension.h>
│       └── graph_tools.py            # Python 构图辅助
└── Graph设计文档.md
```

测试目录：

- `Tests/unit/graph/`：对象模型、Builder、Validator、FusionRule、Forward Mock 单测。
- `Tests/integration/graph/`：Converter/Serializer/Loader/Executor 集成测试。

## 3.3 关键流程

### 3.3.1 离线构图流程

```text
模型源码
  → Converter Adapter 识别算子与权重
  → 查询 FusionRuleTable，确定融合阶段与 node_op_type
  → GraphBuilder.AddNode / AddEdge / AddWeight（at::Tensor）
  → 生成默认 Input/Output Node
  → GraphBuilder.Finalize
      → 解析边引用、派生 producer/consumers
      → 拓扑排序
      → GraphValidator 校验 V1~V16
  → Graph IR 交给 Optimizer（Precision/Shape/Memory）
  → Serializer 调用 ToTensorDesc 写 Protobuf 图与权重字节流
```

### 3.3.2 运行时加载与绑定流程

```text
Loader 读取模型文件
  → 反序列化为 TensorDesc/属性描述
  → RuntimeGraphBuilder.Finalize 重建 RuntimeGraph
  → RuntimeGraph.Validate（V1~V16）
  → 按配置选择 Backend 并 Init
  → 为计算 Node 创建 Kernel
  → RuntimeGraph.BindBackend(backend)
  → RuntimeGraph.BindArena / BindKernel 逐计算 Node
  → RuntimeGraph.ValidateExecution()（V17）
  → Engine 持有 RuntimeGraph，注入 Executor
```

### 3.3.3 Forward 流程

```text
Executor 构建 ForwardContext（host_inputs/host_outputs、KV 视图、attention metadata、采样参数）
  → RuntimeGraph.Forward(ctx)
      → 按拓扑序：
          Input Node  → Backend::Upload（H2D）
          计算 Node   → Kernel::Execute
          Output Node → Backend::Download（D2H）
  → Backend::Synchronize（Executor）
  → Executor 读取 host_outputs
```

### 3.3.4 释放流程

```text
停止调度
  → Backend::Synchronize
  → RuntimeGraph.Release（逐计算 Node → Backend::DestroyKernel）
  → Backend::DestroyBuffer（I/O/Activation/KV/权重/暂存）
  → Backend::Deinit
  → 销毁 RuntimeGraph 与 Backend
```

## 3.4 技术选型与取舍

| 选择 | 方案 | 理由 | 代价 |
| --- | --- | --- | --- |
| 图表示 | Graph IR（工具链）+ RuntimeGraph（移动端）双表示 | IR 复用 torch/LibTorch 语义，工具链表达力强；RuntimeGraph 轻量，移动端不链接 LibTorch | 需要维护转换契约与两套一致性测试 |
| 张量载体 | Graph IR 用 `at::Tensor`；Python 通过 caster 得到 `torch.Tensor`；RuntimeGraph 用 `TensorDesc` | 不自定义底层张量元素；Python/C++ 同一套对象；运行时不依赖 LibTorch | LibTorch 版本需与 Python torch ABI 对齐；序列化需 CPU 拷贝 |
| 图存储 | `std::vector<Node>` / `std::vector<Edge>` + `NodeId/EdgeId` 索引 | 连续内存、遍历快、拓扑序可用索引数组 | 增删节点需要重建索引；Finalize 后不可变，符合约束 |
| 名称查找 | 构图/加载期用 `std::unordered_map<std::string, NodeId>` | 构造与诊断方便 | Forward 热路径不使用字符串查找 |
| 属性容器 | `std::map<std::string, AttributeValue>` | 键有序，Canonical JSON 与序列化确定 | 查找 O(log n)，仅在构图/Python 侧使用 |
| 错误处理 | `ErrorCode` + `Status`，不跨边界抛异常 | 与引擎统一；便于 FFI | 需要显式传播错误 |
| Python 绑定 | pybind11 + LibTorch type caster，C++ 为唯一事实源 | Python 直接获得原生 `torch.Tensor`；避免双套图 | 需要管理 GIL 与张量生命周期 |
| 序列化 | Graph IR 不依赖 Protobuf；Serializer 通过 `ToTensorDesc` 负责映射 | IR 保持工具链语义；线格式与运行时描述解耦、可独立演进 | 需要冻结转换契约与版本规则 |
| 融合规则 | 规则表放在 Graph，模式识别放在 Converter | 规则可版本化、可测试；Graph 只表达结果 | 规则表与 Adapter 需要同步演进 |
| 外部视图 | `KVCacheView`、`AttentionMetadata` 以 POD 视图透传 | Graph 不依赖 KVCache/Executor 实现 | 字段语义需与对应模块文档对齐 |

---

# 4. 详细设计

## 4.1 公共类型与依赖

| 依赖 | 头文件 | 用途 |
| --- | --- | --- |
| `at::Tensor`、`at::ScalarType`、`at::Device`、`at::Layout` | LibTorch `<ATen/ATen.h>` | Graph IR 的张量、dtype、device 与 layout 属性 |
| `ErrorCode`、`Status` | `Runtime/common/status.h` | 统一错误码与返回状态 |
| `OpType` | `Runtime/kernel/kernel_types.h` | 节点算子类型 |
| `Kernel`、`KernelContext`、`KernelDesc` | `Runtime/kernel/kernel.h` | RuntimeGraph 计算节点执行接口 |
| `Backend`、`Stream` | `Runtime/backend/backend.h` | RuntimeGraph 的 H2D/D2H 与同步 |
| pybind11 + `torch/extension.h` | 外部依赖 | Python 绑定；把 `at::Tensor` 映射为 `torch.Tensor` |

包含规则：

- **Graph IR 头文件**（`Graph/cpp/ir/include/graph/*.h`）可以包含 LibTorch/ATen 头文件；只用于工具链构建，不进入移动端运行时。
- **RuntimeGraph 头文件**（`Graph/cpp/runtime/include/graph/*.h`）只包含 Runtime 公共头文件与 C++ 标准库；不得出现 `<ATen/...>`、`<torch/...>`、`c10/`、`cl_*`、`Vk*`、`cuda*` 类型。
- Python 绑定源文件 `Graph/cpp/python/graph_python.cc` 可以包含 `<torch/extension.h>`；该头文件不得被 RuntimeGraph 头文件间接包含。
- Graph IR 与 RuntimeGraph 均不包含 `Proto/*.pb.h`；Serializer 反向依赖 Graph IR 的只读访问器与 `ToTensorDesc` 契约。

## 4.2 双表示与核心数据结构

### 4.2.1 双表示总览

Graph 模块包含两个层次的图表示。二者共享 `ModelConfigT` schema、`OpType`、`FusionRule`、版本语义与拓扑规则，但张量属性载体不同：

| 维度 | Graph IR（构图表示） | RuntimeGraph（运行时表示） |
| --- | --- | --- |
| 使用阶段 | Converter、Optimizer、Serializer | Loader、Executor、Destroy |
| C++ 张量属性 | `at::Tensor`（LibTorch） | `TensorDesc`（轻量结构，无 LibTorch） |
| Python 张量属性 | `torch.Tensor`（pybind11 + LibTorch caster） | 不暴露给 Python |
| dtype | `at::ScalarType` / `torch.dtype` | `DataType` |
| shape | `at::Tensor.sizes()` + `SymbolicShape` | `TensorDesc.shape`（`Dim`） |
| stride | `at::Tensor.stride()` | `TensorDesc.stride` |
| device | `at::Device` | `DeviceType` + arena |
| 权重 | `at::Tensor` | `RuntimeWeightDesc`（offset/bytes/checksum） |
| 主要依赖 | LibTorch；仅工具链构建 | C++17 标准库 + Runtime 公共契约；不链接 LibTorch |
| 生命周期 | 构图、融合、优化、序列化 | 加载、绑定、Forward、释放 |

设计约束：

1. **Graph IR 是构图侧唯一实现**：C++ 类持有 `at::Tensor`；Python 通过 pybind11 暴露同一 C++ 对象，属性在 Python 侧呈现为 `torch.Tensor`。Python 不维护第二套图定义。
2. **RuntimeGraph 是移动端运行时唯一表示**：只包含 `TensorDesc`、裸指针与 POD，不包含 LibTorch 类型；Loader 从序列化数据重建，不经过 Graph IR。
3. **Serializer 是边界**：负责 Graph IR → 序列化 `TensorDesc` + 权重字节流；Graph 模块提供 `ToTensorDesc` 规则与字段契约，Protobuf schema 由 Serializer 定义。
4. **符号形状独立**：`SymbolicShape` 不存储在 `torch.Tensor`/`at::Tensor` 中；张量只保存具体示例形状。Shape Infer 对 `SymbolicShape` 做符号绑定，序列化时写入 `TensorDesc.shape`。

### 4.2.2 基础类型

```cpp
namespace mobilenet::graph {

using NodeId = uint32_t;  // 节点在 Graph::nodes() 中的下标。
using EdgeId = uint32_t;  // 边在 Graph::edges() 中的下标。

inline constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::max();
inline constexpr EdgeId kInvalidEdgeId = std::numeric_limits<EdgeId>::max();

// 属性值：仅允许标量、字符串与一维数组，保证可确定性序列化。
using AttributeValue = std::variant<bool, int64_t, double, std::string,
                                    std::vector<int64_t>, std::vector<double>>;

}  // namespace mobilenet::graph
```

### 4.2.3 Dim、SymbolicShape 与量化描述

```cpp
// 张量维度：常量维或符号维。符号维的 value 是构图期示例值。
struct Dim {
  bool is_symbolic = false;  // true 表示符号维。
  int64_t value = 1;         // 常量维真实值；符号维示例值，必须 > 0。
  std::string symbol;        // 符号维名称，如 "batch"、"seq"；常量维为空。
};

// 符号形状：与 at::Tensor 的具体示例形状并存，供 Shape Infer 绑定。
struct SymbolicShape {
  std::vector<Dim> dims;  // rank 0~8；符号名在同一 shape 内唯一。
};

// 量化描述：阶段一不启用；后续由 Precision Infer 填充。
enum class QuantGranularity { kPerTensor, kPerChannel, kPerGroup };

struct QuantizationDesc {
  DataType storage_dtype = DataType::kInt8;  // 量化后的存储类型。
  QuantGranularity granularity = QuantGranularity::kPerTensor;
  std::vector<float> scales;                 // 量化 scale；粒度决定长度。
  std::vector<int32_t> zero_points;          // 零点；对称量化可为空。
  int64_t group_size = 0;                    // kPerGroup 时的分组大小。
};
```

约束：

- `Dim.value` 必须 `> 0`；符号维的 `symbol` 非空且在同一 `SymbolicShape` 内唯一。
- 常量维的 `symbol` 必须为空，`value` 是真实维度。
- 权重张量必须全为常量维；激活边与 I/O 主机张量可使用符号维。
- `SymbolicShape` 与 `at::Tensor` 必须满足：`tensor.dim() == dims.size()`，且对每个 `i` 有 `tensor.size(i) == dims[i].value`。符号维的 `value` 是构图期示例值（默认 1），用于创建示例张量。
- 运行时 `TensorDesc.shape` 复用同一 `Dim` 结构；符号维在 Shape Infer 后保留符号名与绑定值，Forward 的 `IoBinding.shape` 给出本次调用的具体形状。

### 4.2.4 Graph IR 的张量属性

Graph IR 的所有张量属性都使用 PyTorch/LibTorch 对象表达，不使用运行时 `TensorDesc` 或自定义 dtype/device/layout 枚举：

| Graph IR 属性 | C++ 类型 | Python 类型（pybind11 转换后） |
| --- | --- | --- |
| `Edge.tensor` | `at::Tensor` | `torch.Tensor` |
| `WeightDesc.tensor` | `at::Tensor` | `torch.Tensor` |
| `Node.host_tensor` | `std::optional<at::Tensor>` | `torch.Tensor` 或 `None` |
| `Node.precision` | `at::ScalarType` | `torch.dtype` |
| `ModelConfig.dtype` | `at::ScalarType` | `torch.dtype` |
| `Graph::NodeInputs()` / `NodeOutputs()` | `std::vector<at::Tensor>` | `list[torch.Tensor]` |

说明：

- C++ 侧直接持有 `at::Tensor`；Python 侧通过 pybind11 与 LibTorch 的 type caster 看到同一对象的 `torch.Tensor` 视图，不存在 Python 侧的第二套张量封装。
- `Node` 不重复存储输入/输出张量；它保存有序的 `input_edges`/`output_edges`。`Graph::NodeInputs(NodeId)` 与 `Graph::NodeOutputs(NodeId)` 按端口顺序解析 Edge 并返回 `at::Tensor`；Python 绑定将 `node.inputs`/`node.outputs` 暴露为只读 `list[torch.Tensor]`。
- `at::Tensor` 只保存具体形状；符号维由同一边上的 `SymbolicShape` 表达。两者的一致性由 V10 校验。
- Graph IR 的 `at::Tensor` 不参与自动微分：`tensor.requires_grad()` 必须为 `false`；Graph 只做推理图构图，不执行反向传播。
- `at::Tensor` 可以是 CPU 或 CUDA 张量；Serializer 在写入模型文件前统一转为 CPU、连续、行主序。移动端 RuntimeGraph 不接触 `at::Tensor`。
- Graph IR 持有 `at::Tensor` 句柄，不复制底层数据；Optimizer 可以在 `Finalize` 前替换张量，`Finalize` 后图结构不可变。序列化期间调用方不得修改张量内容。

### 4.2.5 Graph IR 的 Edge、Weight 与 Node

```cpp
// 融合前的原始散算子记录，用于追溯、转换报告与测试。
struct FusedOp {
  std::string source_name;  // 原模型模块路径，如 "model.layers.0.self_attn.q_proj"。
  std::string op_kind;      // 逻辑类型，如 "linear"、"rmsnorm"、"rope"、"silu"。
  std::map<std::string, AttributeValue> attributes;  // 原始算子参数。
};

// 边：表示一个张量数据流。producer/consumers 由 GraphBuilder::Finalize 派生。
struct Edge {
  std::string name;                    // 边唯一名，建议 "<producer>.<role>"。
  at::Tensor tensor;                   // 数据张量；dtype/shape/stride/device 由此读取。
  SymbolicShape symbolic_shape;        // 符号维；与 tensor.sizes() 的示例值一致。
  std::optional<QuantizationDesc> quantization;  // 可选量化元数据。
  std::string producer;                // 生产者节点名；权重边为空。
  std::vector<std::string> consumers;  // 消费者节点名列表，按注册顺序。
  bool is_weight = false;              // 是否为权重边。
  std::string weight_ref;              // is_weight=true 时引用 WeightDesc.ref。
};

// 权重表条目：Edge.is_weight=true 时通过 weight_ref 引用。
struct WeightDesc {
  std::string ref;           // 权重表唯一键，如 "model.layers.0.self_attn.q_proj.weight"。
  std::string source_name;   // 原模型参数名，用于追溯。
  at::Tensor tensor;         // 权重张量；允许 CPU/CUDA，序列化前转 CPU。
  std::optional<QuantizationDesc> quantization;  // 可选量化元数据。
  uint64_t checksum = 0;     // 权重校验和，由 Converter 或 Serializer 填写。
};

// 节点：计算 Node 对应一个 Kernel；Input/Output Node 不创建 Kernel。
struct Node {
  std::string name;                 // 节点唯一名。
  OpType op_type = OpType::kUnknown;  // 融合节点类型。
  std::vector<FusedOp> fused_ops;   // 融合前的散算子有序列表。
  std::map<std::string, AttributeValue> attributes;  // 融合节点属性。
  std::vector<std::string> input_edges;   // 有序输入边名，决定 Kernel 入参顺序。
  std::vector<std::string> output_edges;  // 有序输出边名，决定 Kernel 出参顺序。
  at::ScalarType precision = at::ScalarType::Undefined;  // 节点主精度。
  std::string kernel_key;           // 逻辑 Kernel 键，如 "attention"。
  std::string fusion_rule_id;       // 命中的融合规则编号。

  // I/O 节点专用字段；计算 Node 必须为空。
  std::string binding_name;                   // Executor 绑定主机缓冲的键。
  CopyKind copy_kind = CopyKind::kNone;       // kH2D / kD2H。
  std::optional<at::Tensor> host_tensor;      // I/O 节点的示例主机张量。
};
```

约束：

- `Edge.tensor` 必须 `defined()`；非权重边是激活示例张量，权重边与 `WeightDesc.tensor` 指向同一存储。
- 权重边 `Edge.tensor` 与 `weights[weight_ref].tensor` 必须满足 `is_same()` 或共享同一 `storage().data_ptr()`；权重边没有生产者。
- 计算 Node：`op_type != kInput/kOutput`，`fused_ops` 非空，`kernel_key` 非空，`binding_name` 为空，`copy_kind == kNone`。
- Input Node：`op_type == kInput`，`input_edges` 为空，`copy_kind == kH2D`，`host_tensor` 为 CPU 张量且 dtype 与输出边一致。
- Output Node：`op_type == kOutput`，`output_edges` 为空，`copy_kind == kD2H`，`host_tensor` 为 CPU 张量且 dtype 与输入边一致。
- `fusion_rule_id` 仅对参与块融合的计算 Node 必填；Embedding/LmHead/Sampling/Cast 可为空。
- `Node.precision` 必须与节点输出边 dtype 一致；显式精度转换通过 `kCast` 节点完成。

### 4.2.6 ModelConfig

```cpp
// RoPE 配置：由 Converter 从模型 config.json 读取并校验。
struct RopeConfig {
  float theta = 10000.0f;                 // RoPE base。
  RopeLayout layout = RopeLayout::kNeox;  // 旋转布局；Qwen3 使用 Neox 风格。
  std::string scaling_type = "none";      // "none"/"linear"/"dynamic" 等。
  float scaling_factor = 1.0f;
  int32_t original_max_position_embeddings = 0;
};

// 模型元信息：Graph IR 与 RuntimeGraph 共享字段，dtype 类型由模板参数决定。
template <typename DType>
struct ModelConfigT {
  std::string model_family;        // 如 "qwen3"。
  int32_t num_layers = 0;          // Decoder Block 数。
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  int32_t num_attention_heads = 0;
  int32_t num_kv_heads = 0;        // GQA 的 KV 头数。
  int32_t head_dim = 0;
  int32_t vocab_size = 0;
  int32_t max_context_length = 0;  // 模型最大上下文；运行上限可更小。
  float norm_eps = 0.0f;           // 由 config.json 填写，不硬编码。
  std::string activation = "silu"; // MLP 激活函数。
  bool tie_word_embeddings = false;
  DType dtype;                     // Graph IR 为 at::ScalarType，运行时为 DataType。
  RopeConfig rope;
  std::map<std::string, AttributeValue> extra;  // 模型族特有配置。
};

using ModelConfig = ModelConfigT<at::ScalarType>;           // Graph IR。
using RuntimeModelConfig = ModelConfigT<DataType>;          // RuntimeGraph。
```

说明：Qwen3-0.6B 作为首个验证模型，其具体数值以模型目录 `config.json` 为准，由 Converter 写入并在转换报告中记录；其他模型族同理。Graph 不硬编码任何模型超参数。

### 4.2.7 Graph IR 的 Graph

```cpp
class Graph {
 public:
  Graph(Graph&&) = default;
  Graph& operator=(Graph&&) = default;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  ~Graph();

  // 只读访问器。
  uint32_t schema_version_major() const;
  uint32_t schema_version_minor() const;
  const std::string& fusion_rules_version() const;
  const ModelConfig& model_config() const;
  const std::vector<Node>& nodes() const;
  const std::vector<Edge>& edges() const;
  const std::vector<WeightDesc>& weights() const;
  const std::vector<NodeId>& topo_order() const;
  const std::vector<NodeId>& input_nodes() const;
  const std::vector<NodeId>& output_nodes() const;
  NodeId FindNode(std::string_view name) const;
  EdgeId FindEdge(std::string_view name) const;

  // 张量视图：按 Node 的端口顺序返回 at::Tensor。
  std::vector<at::Tensor> NodeInputs(NodeId id) const;
  std::vector<at::Tensor> NodeOutputs(NodeId id) const;

  // 结构校验与确定性 JSON；不执行、不绑定 Kernel。
  Status Validate(ValidationReport* report) const;
  std::string ToCanonicalJson() const;

 private:
  // 省略成员细节；节点/边/权重使用连续容器，拓扑序为 NodeId 数组。
};
```

Graph IR 不包含 `BindBackend`、`BindKernel`、`Forward`、`Release`；这些接口属于 RuntimeGraph。

### 4.2.8 RuntimeGraph 的 TensorDesc 与数据结构

```cpp
// 运行时元素类型：不依赖 LibTorch，由 at::ScalarType 映射得到。
enum class DataType {
  kUnknown, kFloat32, kFloat16, kBFloat16,
  kInt8, kInt4, kUInt8, kInt32, kInt64, kBool,
};

// 运行时内存布局；阶段一仅支持行主序。
enum class Layout { kRowMajor, kColumnMajor };

// 运行时设备类型；具体设备索引由 arena 绑定提供。
enum class DeviceType { kHost, kDevice };

// 运行时张量描述：由 at::Tensor + SymbolicShape 转换得到。
struct TensorDesc {
  DataType dtype = DataType::kUnknown;  // 元素类型。
  std::vector<Dim> shape;               // 形状；符号维保留名称与绑定值。
  std::vector<int64_t> stride;          // 元素步长；空表示紧凑行主序。
  Layout layout = Layout::kRowMajor;    // 内存布局。
  DeviceType device = DeviceType::kDevice;  // 主机/设备侧。
  uint64_t offset = 0;                  // 张量在所属 Edge 缓冲内的字节偏移。
  uint64_t size_bytes = 0;              // 全静态时可计算；动态时为示例值。
  std::optional<QuantizationDesc> quantization;  // 可选量化描述。
};

// 运行时边：只含描述与绑定信息，不含 at::Tensor。
struct RuntimeEdge {
  std::string name;
  TensorDesc tensor;
  std::string producer;
  std::vector<std::string> consumers;
  bool is_weight = false;
  std::string weight_ref;
  int32_t arena_id = -1;               // MemoryPlan arena 索引；加载期填写。
  uint64_t memory_offset = 0;          // 相对 arena 基址的字节偏移；加载期填写。
};

// 运行时权重：只含描述与文件偏移，权重字节由 Loader 读入设备缓冲。
struct RuntimeWeightDesc {
  std::string ref;
  std::string source_name;
  TensorDesc tensor;
  uint64_t data_offset = 0;            // 权重二进制文件内的字节偏移。
  uint64_t data_bytes = 0;             // 权重字节数。
  uint64_t checksum = 0;               // 权重校验和。
};

// 运行时节点：precision 使用 DataType，host_tensor 使用 TensorDesc。
struct RuntimeNode {
  std::string name;
  OpType op_type = OpType::kUnknown;
  std::vector<FusedOp> fused_ops;
  std::map<std::string, AttributeValue> attributes;
  std::vector<std::string> input_edges;
  std::vector<std::string> output_edges;
  DataType precision = DataType::kUnknown;
  std::string kernel_key;
  std::string fusion_rule_id;
  std::string binding_name;
  CopyKind copy_kind = CopyKind::kNone;
  std::optional<TensorDesc> host_tensor;
};

using RuntimeModelConfig = ModelConfigT<DataType>;
```

约束：

- `RuntimeGraph` 的公共头文件不得包含 `<ATen/...>`、`<torch/...>` 或 `c10/` 头文件。
- `TensorDesc.shape` 的符号维与 Graph IR 的 `SymbolicShape` 一一对应；`size_bytes` 在动态形状下记录示例值或最大绑定值，由 Shape Infer 决定。
- `RuntimeNode.precision` 与输出边 `TensorDesc.dtype` 一致。
- `RuntimeEdge.tensor.offset` 是 Edge 缓冲内的偏移；`arena_id + memory_offset` 是设备地址解析依据。

### 4.2.9 RuntimeGraph 的 I/O 与 ForwardContext

```cpp
// 主机 I/O 绑定：Executor 为每个 Input/Output Node 提供。
struct IoBinding {
  std::string binding_name;   // 对应 RuntimeNode.binding_name。
  void* host_ptr = nullptr;   // 主机缓冲首地址；Forward 期间有效。
  uint64_t bytes = 0;         // 缓冲字节数，必须与张量大小一致。
  DataType dtype = DataType::kUnknown;
  std::vector<Dim> shape;     // 具体形状；符号维必须已绑定。
  std::vector<int64_t> stride;
};

// 逐请求采样参数；由 Executor 按 Packing 顺序传入。
struct SamplingParams {
  float temperature = 1.0f;
  int32_t top_k = 0;          // 0 表示不启用。
  float top_p = 1.0f;
  float repetition_penalty = 1.0f;
  uint64_t seed = 0;
  std::vector<int32_t> stop_ids;
};

struct SamplingBatch {
  const SamplingParams* params = nullptr;  // 长度 num_requests。
  const int32_t* request_ids = nullptr;    // 长度 num_requests。
  int32_t num_requests = 0;
};

// KV Cache 视图：RuntimeGraph 只透传，不解释其内部布局。
struct KVCacheView {
  void* data = nullptr;                  // 设备侧 KV 池基址。
  uint64_t bytes = 0;
  const int32_t* block_table = nullptr;  // 块表地址。
  int32_t block_table_len = 0;
  int32_t block_size = 0;
  int32_t num_layers = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  DataType dtype = DataType::kFloat16;
};

// Attention 元数据：描述本 Step 的 Prefill/Decode 形态。
struct AttentionMetadata {
  bool is_prefill = true;                  // 全 Step 同阶段时的便捷标记。
  bool has_mixed_phases = false;           // true 表示同一 Step 混合 Prefill/Decode。
  const int32_t* request_phases = nullptr; // 每请求阶段：0=PREFILL，1=DECODE。
  int32_t batch_size = 0;
  int32_t query_len = 0;                   // 本 Step 查询 token 总数。
  int32_t kv_len = 0;                      // 本 Step 可见 KV 总长度。
  const int32_t* cu_seqlens_q = nullptr;   // 长度 batch_size+1。
  const int32_t* cu_seqlens_k = nullptr;
  const int32_t* request_ids = nullptr;
};

// RuntimeGraph.Forward 的输入上下文；全部为裸指针/POD，无 LibTorch 类型。
struct ForwardContext {
  std::map<std::string, IoBinding> host_inputs;   // key: binding_name。
  std::map<std::string, IoBinding> host_outputs;  // key: binding_name。
  KVCacheView kv_cache_view;
  AttentionMetadata attention_metadata;
  SamplingBatch sampling_params;
  void* workspace = nullptr;
  uint64_t workspace_bytes = 0;
  Backend* backend = nullptr;  // 非拥有；必须与 RuntimeGraph 绑定的 Backend 相同。
  Stream* stream = nullptr;    // 非拥有；本次 Forward 使用的流。
};

// 设备内存 arena 绑定：Loader 分配完 MemoryPlan 后调用 RuntimeGraph::BindArena。
struct ArenaBinding {
  int32_t arena_id = -1;       // 对应 MemoryPlan arena 索引。
  void* device_ptr = nullptr;  // 设备侧基址；RuntimeGraph 不拥有。
  uint64_t bytes = 0;          // arena 字节数，用于越界校验。
};
```

说明：`KVCacheView`、`AttentionMetadata`、`SamplingBatch` 是 RuntimeGraph 与 KVCache/Executor 之间的视图契约，字段语义由对应模块设计文档冻结；RuntimeGraph 只做透传与基本校验。视图中的指针默认按对应模块约定提供主机或设备地址，RuntimeGraph 不进行隐式拷贝。

### 4.2.10 RuntimeGraph

```cpp
class RuntimeGraph {
 public:
  RuntimeGraph(RuntimeGraph&&) = default;
  RuntimeGraph& operator=(RuntimeGraph&&) = default;
  RuntimeGraph(const RuntimeGraph&) = delete;
  RuntimeGraph& operator=(const RuntimeGraph&) = delete;
  ~RuntimeGraph();

  // 只读访问器。
  uint32_t schema_version_major() const;
  uint32_t schema_version_minor() const;
  const std::string& fusion_rules_version() const;
  const RuntimeModelConfig& model_config() const;
  const std::vector<RuntimeNode>& nodes() const;
  const std::vector<RuntimeEdge>& edges() const;
  const std::vector<RuntimeWeightDesc>& weights() const;
  const std::vector<NodeId>& topo_order() const;
  const std::vector<NodeId>& input_nodes() const;
  const std::vector<NodeId>& output_nodes() const;
  NodeId FindNode(std::string_view name) const;
  EdgeId FindEdge(std::string_view name) const;

  // 结构校验：V1~V16；张量检查使用 TensorDesc。
  Status Validate(ValidationReport* report) const;

  // 运行时绑定：由 Loader 调用。
  Status BindBackend(Backend* backend);
  Status BindArena(int32_t arena_id, void* device_ptr, uint64_t bytes);
  Status BindArenas(const std::vector<ArenaBinding>& arenas);
  Status BindKernel(std::string_view node_name, Kernel* kernel);
  Status ValidateExecution() const;
  Backend* backend() const;  // 返回绑定的非拥有 Backend 指针；未绑定为 nullptr。

  // 模型前向：由 Executor 调用；异步返回，同步由 Executor 负责。
  Status Forward(const ForwardContext& ctx) const;

  // 释放执行表：逐计算 Node 调用 Backend::DestroyKernel；幂等。
  Status Release();

  // 调试与双端比较；非线格式。
  std::string ToCanonicalJson() const;

 private:
  // 省略成员细节，见 4.8 节。
};
```

### 4.2.11 at::Tensor 到 TensorDesc 的转换规则

Serializer 在写模型文件前调用 `ToTensorDesc`，把 Graph IR 张量转换为 RuntimeGraph 可用的描述：

```cpp
// 将 Graph IR 张量转换为运行时描述；只转换元数据，不复制、不修改底层数据。
// 前置条件：tensor.defined()，shape 与 tensor.sizes() 一致。
// 序列化路径要求调用方先完成 detach/CPU/连续化/列主序转置。
TensorDesc ToTensorDesc(const at::Tensor& tensor,
                        const SymbolicShape& shape,
                        std::optional<QuantizationDesc> quantization);
```

转换规则：

| Graph IR（at::Tensor） | TensorDesc | 说明 |
| --- | --- | --- |
| `tensor.scalar_type()` | `dtype` | 按 4.3.2 的 dtype 映射表转换 |
| `shape.dims` | `shape` | 保留符号名与示例值 |
| `tensor.stride()` | `stride` | 元素步长 |
| `tensor.is_contiguous()` | `layout` | 连续行主序为 `kRowMajor`；列主序权重在调用前已转置 |
| `tensor.device()` | `device` | CPU → `kHost`，CUDA → `kDevice`；写入模型文件的 TensorDesc 必须为 `kHost` |
| `tensor.storage_offset() * tensor.element_size()` | `offset` | 存储偏移字节数 |
| `tensor.nbytes()` | `size_bytes` | 符号维使用示例形状；Shape Infer 可更新为最大绑定值 |
| `quantization` | `quantization` | 阶段一缺省；量化张量必须显式提供描述 |

补充规则：

- Graph IR 的 `at::Tensor` 必须满足 `tensor.layout() == at::kStrided`；阶段一不支持稀疏张量。
- 序列化路径由 Serializer 先执行 `tensor.detach().to(at::kCPU).contiguous()`，再调用 `ToTensorDesc`；权重张量逐项从此前状态转换，原始 Graph IR 张量不被修改。
- 2-D 列主序权重在 CPU 连续化阶段转置为行主序；转置只改变权重字节流与 stride，不改变数学语义。
- 非 CPU 张量的复制只发生在离线序列化，不影响运行时；RuntimeGraph 不接触 `at::Tensor`。
- 符号维的 `value` 是构图期示例值，不能直接作为运行时最大形状；Shape Infer 负责把符号绑定到 MaxNumToken/ContextLength 等边界，并写回 `TensorDesc.shape` 的绑定值。
- 转换后的 `TensorDesc` 参与 V10 校验；RuntimeGraph 加载并绑定 arena/Kernel 后，由 V17 校验执行表与 arena。转换失败返回 `INVALID_ARGUMENT` 或 `UNSUPPORTED_DTYPE`/`UNSUPPORTED_LAYOUT`。

### 4.2.12 校验报告

```cpp
// 校验问题：携带节点、边、binding 与融合规则上下文，便于定位。
struct ValidationIssue {
  ErrorCode code = ErrorCode::kOk;
  std::string node_name;
  std::string edge_name;
  std::string binding_name;
  std::string fusion_rule_id;
  std::string detail;
};

struct ValidationReport {
  std::vector<ValidationIssue> issues;
  bool ok() const { return issues.empty(); }
};
```

## 4.3 枚举与 OpType

### 4.3.1 OpType

| 值 | 含义 | 是否计算 Node | Kernel 必需 | 所属融合阶段 |
| --- | --- | --- | --- | --- |
| `kInput` | 主机输入 H2D | 否 | 否 | 图输入 |
| `kOutput` | 设备输出 D2H | 否 | 否 | 图输出 |
| `kEmbedding` | Token ID → hidden | 是 | 是 | 块外 |
| `kRmsNorm` | RMSNorm | 是 | 是 | 阶段 1/5 |
| `kQkvProjection` | Q/K/V 投影 + QK RMSNorm + RoPE | 是 | 是 | 阶段 2 |
| `kAttention` | Attention 核心 + 输出投影 + KV 写入 | 是 | 是 | 阶段 3 |
| `kResidualAdd` | 残差相加 | 是 | 是 | 阶段 4 |
| `kMlpGateUpAct` | gate/up 投影 + 激活 + 逐元素乘 | 是 | 是 | 阶段 6 |
| `kMlpDownResidual` | down 投影 + 残差相加 | 是 | 是 | 阶段 7 |
| `kLmHead` | 末 token 汇聚 + LM Head 投影 | 是 | 是 | 块外 |
| `kSampling` | 采样 | 是 | 是 | 块外 |
| `kCast` | 显式 dtype 转换 | 是 | 是 | 块外/融合屏障 |

约束：

- `kInput`/`kOutput` 不进入 `Node → Kernel*` 执行表。
- `kCast` 是显式节点，用于 host/device dtype 不一致或跨阶段精度变化；Cast 是融合屏障。
- 本表是阶段一 Qwen3-0.6B 所需的 OpType 集合；OpType 是开放枚举，新增模型族如需新的算子语义（如 LayerNorm、MoE 路由），按《Kernel 公共契约》扩展 OpType，不得用属性隐式改变已有 OpType 的语义。
- `OpType` 的最终枚举值由《Kernel 公共契约》冻结；Graph 只依赖上述语义。
- 本文代码片段中的枚举常量名（如 `OpType::kRmsNorm`、`ErrorCode::kInvalidArgument`）为提案命名，最终以 `Runtime/kernel/` 与 `Runtime/common/` 定稿为准；语义与取值顺序不变。

### 4.3.2 Graph IR 的 torch/LibTorch 类型

Graph IR 不定义 `DataType`、`Layout`、`DeviceType`、`TensorDesc` 等底层张量元素；这些类型只属于 RuntimeGraph 与序列化。Graph IR 直接使用 PyTorch/LibTorch 类型：

| Graph IR 属性 | C++（LibTorch） | Python（PyTorch） |
| --- | --- | --- |
| 张量 | `at::Tensor` | `torch.Tensor` |
| dtype | `at::ScalarType` | `torch.dtype` |
| shape | `at::IntArrayRef` / `at::Tensor.sizes()` | `torch.Size` |
| stride | `at::IntArrayRef` / `at::Tensor.stride()` | `tuple[int, ...]` |
| device | `at::Device` | `torch.device` |
| layout | `at::Layout` | `torch.layout` |

dtype 映射（Graph IR → RuntimeGraph/序列化）：

| `at::ScalarType` | `torch.dtype` | `DataType` |
| --- | --- | --- |
| `at::kFloat` | `torch.float32` | `kFloat32` |
| `at::kHalf` | `torch.float16` | `kFloat16` |
| `at::kBFloat16` | `torch.bfloat16` | `kBFloat16` |
| `at::kChar` | `torch.int8` | `kInt8` |
| `at::kByte` | `torch.uint8` | `kUInt8` |
| `at::kInt` | `torch.int32` | `kInt32` |
| `at::kLong` | `torch.int64` | `kInt64` |
| `at::kBool` | `torch.bool` | `kBool` |
| `at::kQInt8` | `torch.qint8` | `kInt8` + `QuantizationDesc` |

补充说明：

- `Layout`（运行时）由 `at::Tensor` 的 stride 与连续性派生：连续行主序为 `kRowMajor`；二维列主序权重在序列化前转置，阶段一运行时只要求 `kRowMajor`。
- `DeviceType`（运行时）由 `at::Device` 映射：CPU → `kHost`，CUDA → `kDevice`；设备索引由 arena 绑定与 Backend 上下文管理。
- `CopyKind`、`RopeLayout`、`BlockStage` 是图语义枚举，不属于底层张量元素，Graph IR 与 RuntimeGraph 共用。
- `DataType`、`Layout`、`DeviceType` 只允许出现在 RuntimeGraph/序列化头文件中；Graph IR 公共头文件不得定义或使用这些自定义张量类型。

## 4.4 GraphBuilder、RuntimeGraphBuilder 与构造规则

### 4.4.1 GraphBuilder（Graph IR）

```cpp
class GraphBuilder {
 public:
  GraphBuilder();
  ~GraphBuilder();

  // 添加节点；name 重复、字段非法时返回 INVALID_ARGUMENT。
  Status AddNode(Node node, NodeId* node_id);
  // 添加边；name 重复、at::Tensor 未定义或 SymbolicShape 不匹配时返回 INVALID_ARGUMENT。
  Status AddEdge(Edge edge, EdgeId* edge_id);
  // 添加权重；ref 重复、张量未定义时返回 INVALID_ARGUMENT。
  Status AddWeight(WeightDesc weight);
  // 设置图输入/输出节点；节点必须存在且 op_type 匹配。
  Status SetInputNodes(std::vector<NodeId> ids);
  Status SetOutputNodes(std::vector<NodeId> ids);
  Status SetModelConfig(ModelConfig config);
  Status SetSchemaVersion(uint32_t major, uint32_t minor);
  Status SetFusionRulesVersion(std::string version);

  // 完成边引用解析、拓扑排序与 V1~V16 校验；成功后 builder 失效。
  Status Finalize(std::unique_ptr<Graph>* graph);

  // 最近一次校验报告；Finalize 失败时可读取。
  const ValidationReport& validation_report() const;
};
```

说明：

- `GraphBuilder` 默认写入当前 `schema_version`（1.0）与 `FusionRuleTable::Instance().version()`；Converter 可为旧模型显式保留版本，再由 `Validate` 执行兼容性检查。
- `AddEdge` 接收 `at::Tensor`；Python 侧由 pybind11 接收 `torch.Tensor` 并转换为 `at::Tensor`。`SymbolicShape` 与 `tensor.sizes()` 的一致性在 `AddEdge` 即时校验，也在 Finalize 的 V10 中复检。
- `GraphBuilder` 只构建 Graph IR，不绑定 Backend/Kernel，不提供 Forward。

### 4.4.2 RuntimeGraphBuilder（RuntimeGraph）

```cpp
class RuntimeGraphBuilder {
 public:
  RuntimeGraphBuilder();
  ~RuntimeGraphBuilder();

  // 从反序列化数据重建运行时图；不接受 at::Tensor。
  Status AddNode(RuntimeNode node, NodeId* node_id);
  Status AddEdge(RuntimeEdge edge, EdgeId* edge_id);
  Status AddWeight(RuntimeWeightDesc weight);
  Status SetInputNodes(std::vector<NodeId> ids);
  Status SetOutputNodes(std::vector<NodeId> ids);
  Status SetModelConfig(RuntimeModelConfig config);
  Status SetSchemaVersion(uint32_t major, uint32_t minor);
  Status SetFusionRulesVersion(std::string version);

  // 完成边引用解析、拓扑排序与 V1~V16 校验；不执行 V17。
  Status Finalize(std::unique_ptr<RuntimeGraph>* graph);

  const ValidationReport& validation_report() const;
};
```

说明：`RuntimeGraphBuilder` 由 Loader 调用，输入来自 Serializer 写入的 `TensorDesc` 与权重偏移；它不依赖 LibTorch，不接触 `at::Tensor`。V17 在 Backend/Kernel/arena 绑定后由 `RuntimeGraph::ValidateExecution()` 执行。

### 4.4.3 构造规则

1. 节点与边先添加到 builder，名称即时查重；边引用在 Finalize 解析。
2. `Node.input_edges`/`output_edges` 是**有序且权威**的端口列表；`Edge.producer`/`consumers` 由 Finalize 派生。
3. 每条边必须被至少一个 Node 引用；权重边必须被至少一个计算 Node 引用。
4. Input Node 的 `output_edges` 产生图输入边；Output Node 的 `input_edges` 消费图输出边。
5. 拓扑排序使用 Kahn 算法，就绪节点按 `NodeId` 升序出队，保证确定性。
6. Finalize 成功后 builder 进入 `kFinalized` 状态，再调用 AddNode/AddEdge 返回 `STATE_ERROR`。
7. Finalize 在临时结构上派生边关系并校验，失败不修改 builder 已添加数据；builder 可修正后重试。成功后 builder 失效，图所有权移交给 `std::unique_ptr<Graph>`。
8. Graph IR 的每条边在加入时校验 `at::Tensor` 已定义、`SymbolicShape` 与 `tensor.sizes()` 一致；权重边还必须校验与 `WeightDesc.tensor` 共享存储。
9. RuntimeGraphBuilder 不执行上述 `at::Tensor` 校验，改为校验 `TensorDesc` 的 dtype/shape/stride/offset/size_bytes 自洽；V17 在绑定后执行。

### 4.4.4 命名规范

| 对象 | 规范 | 示例 |
| --- | --- | --- |
| Input Node | `input.<binding_name>` | `input.input_ids` |
| Output Node | `output.<binding_name>` | `output.token_ids` |
| Embedding | `embedding` | `embedding` |
| Block 内节点 | `layer.<i>.<role>` | `layer.0.attn_norm` |
| 末归一化 | `final_norm` | `final_norm` |
| LM Head | `lm_head` | `lm_head` |
| Sampling | `sampling` | `sampling` |
| Cast | `cast.<id>` | `cast.0` |
| 边 | `<producer>.<role>` | `layer.0.q` |
| 权重 ref | 原模型参数名 | `model.layers.0.self_attn.q_proj.weight` |

## 4.5 图校验

### 4.5.1 校验规则

| 编号 | 规则 | 失败错误码 |
| --- | --- | --- |
| V1 | 节点名非空、符合命名约束、全局唯一 | `INVALID_ARGUMENT` |
| V2 | 边名非空、符合命名约束、全局唯一 | `INVALID_ARGUMENT` |
| V3 | I/O Node 的 `binding_name` 非空且全局唯一 | `INVALID_ARGUMENT` |
| V4 | 至少各有一个 Input Node 与 Output Node | `INVALID_ARGUMENT` |
| V5 | Input Node：入度为 0、`copy_kind==kH2D`、有 `host_tensor`、无 kernel；`host_tensor` 为 CPU 张量（IR：`at::Tensor::device().is_cpu()`；运行时：`TensorDesc.device==kHost`），输出边 dtype 与 host dtype 一致 | `INVALID_ARGUMENT` |
| V6 | Output Node：出度为 0、`copy_kind==kD2H`、有 `host_tensor`、无 kernel；`host_tensor` 为 CPU 张量（IR：`at::Tensor::device().is_cpu()`；运行时：`TensorDesc.device==kHost`），输入边 dtype 与 host dtype 一致 | `INVALID_ARGUMENT` |
| V7 | 非权重边恰好一个生产者；每条边至少一个消费者；无悬空边；Node 端口不重复引用同一条边 | `INVALID_ARGUMENT` |
| V8 | 权重边 `is_weight==true`、`weight_ref` 存在、无生产者，且 IR 中 `Edge.tensor` 与 `WeightDesc.tensor` 共享存储；非权重边 `is_weight==false`、`weight_ref` 为空、有生产者 | `INVALID_ARGUMENT` |
| V9 | 图无环；拓扑排序成功 | `INVALID_ARGUMENT` |
| V10 | 生产者输出张量与消费者输入张量 dtype 相同、rank 相同、维度兼容（常量维相等；符号维名称一致）；IR 检查 `at::Tensor::scalar_type()`/`sizes()`/`stride()` 与 `SymbolicShape` 一致，运行时检查 `TensorDesc` 的 `dtype`/`shape`/`stride`/`size_bytes` 自洽 | `INVALID_ARGUMENT` |
| V11 | 图输入边必须由 Input Node 产生；图输出边必须由 Output Node 消费；I/O 边不得为权重；I/O 边 host/device dtype 一致 | `INVALID_ARGUMENT` |
| V12 | 计算 Node 的 `fused_ops` 非空、`kernel_key` 非空；I/O Node 无 `fused_ops`/`kernel_key`；若图中存在 `kLmHead` 或 `kSampling`，则 Sampling Node 恰好一个且位于 LmHead 之后 | `INVALID_ARGUMENT` |
| V13 | 若融合 Node 的 `fusion_rule_id` 非空，必须存在于规则表；`op_type`、`fused_ops` 顺序与规则一致；不得跨阶段；块外算子（Embedding/LmHead/Sampling/Cast）可为空 | `INVALID_ARGUMENT` |
| V14 | 拓扑序中 Input Node 在全部计算 Node 之前，Output Node 在全部计算 Node 之后 | `INTERNAL` |
| V15 | `schema_version` 主版本受支持；次版本不高于当前实现 | `VERSION_MISMATCH` |
| V16 | ModelConfig 不变量：层数/头数/维度为正；`num_attention_heads % num_kv_heads == 0`；`head_dim * num_attention_heads` 与配置一致；符号维非空 | `INVALID_ARGUMENT` |
| V17 | RuntimeGraph 执行表校验：计算 Node 全部绑定非空 Kernel；I/O Node 无 Kernel；Backend 非空；所有 Edge（含权重边）的 arena 已绑定且 `offset + size_bytes` 不越界 | `STATE_ERROR` / `UNSUPPORTED_OP` |

说明：V1~V16 同时适用于 Graph IR 与 RuntimeGraph；V5/V6/V8/V10 的检查对象按表示不同：Graph IR 使用 `at::Tensor` 与 `SymbolicShape`，RuntimeGraph 使用 `TensorDesc`。V17 仅适用于 RuntimeGraph。

### 4.5.2 校验算法

```cpp
// Graph IR 与 RuntimeGraph 共用同一套结构校验流程；张量检查通过各自的
// 访问器分派：IR 读取 at::Tensor，运行时读取 TensorDesc。
Status RuntimeGraph::Validate(ValidationReport* report) const {
  report->issues.clear();
  // 1. 名称与引用完整性：V1~V3、V7、V8。
  // 2. I/O 节点与 I/O 边：V4~V6、V11。
  // 3. 拓扑排序与环检测：V9、V14。
  // 4. dtype/shape 与融合边界：V10、V12、V13。
  // 5. 版本与 ModelConfig：V15、V16。
  return report->ok() ? Status::Ok()
                      : MakeStatus(report->issues.front().code,
                                   FormatIssues(*report));
}
```

- `Graph::Validate` 与 `RuntimeGraph::Validate` 共享同一校验流程；IR 读取 `at::Tensor`，运行时读取 `TensorDesc`；V17 由 `ValidateExecution` 单独执行。
- 复杂度 O(V+E)；每条规则只遍历一次邻接关系。
- 错误收集上限 32 条，避免恶意图导致内存膨胀；超过上限追加一条 `INTERNAL` 汇总。
- 校验不修改图；可在 Loader 重建后重复调用。
- `ValidateExecution` 单独执行 V17，不重复结构校验。

### 4.5.3 错误定位

每条 `ValidationIssue` 至少包含：

- `code`：错误码。
- `node_name`/`edge_name`/`binding_name`/`fusion_rule_id`：能定位到的最小对象。
- `detail`：人类可读原因，例如 `expected float16, got bfloat16`。
- 错误消息格式：`[Graph] <CODE> node=<...> edge=<...> binding=<...> rule=<...>: <detail>`。

## 4.6 融合规则

### 4.6.1 统一融合骨架

Decoder-only Transformer Block 的算子融合统一遵循以下骨架：

```text
归一化 → 投影+后处理 → attention → 残差 → 归一化 → gate/up+激活 → down+残差
```

该骨架对 Decoder-only Transformer 类模型通用；阶段划分与融合边界是模型无关的，具体 `op_type`、属性与 kernel_key 由各模型族规则集实例化。下表以 Qwen3-0.6B 为例。

| 阶段 | 名称 | 融合 Node（Qwen3 实例） | 阶段职责 |
| --- | --- | --- | --- |
| 1 | 归一化 | `kRmsNorm` | Attention 前的 RMSNorm |
| 2 | 投影+后处理 | `kQkvProjection` | Q/K/V 投影、QK RMSNorm、RoPE |
| 3 | attention | `kAttention` | Attention 核心、输出投影、KV 写入 |
| 4 | 残差 | `kResidualAdd` | Attention 输出与残差输入相加 |
| 5 | 归一化 | `kRmsNorm` | MLP 前的 RMSNorm |
| 6 | gate/up+激活 | `kMlpGateUpAct` | gate/up 投影、SiLU、逐元素乘 |
| 7 | down+残差 | `kMlpDownResidual` | down 投影与残差相加 |

骨架是**硬边界**：一个融合 Node 只能覆盖一个阶段，不得把相邻阶段合并。阶段 4 的残差单独成节点；阶段 7 按骨架要求把 down 与残差合并。

Graph 的图结构不限制模型结构：上述骨架是 Decoder-only Transformer 的默认融合规则；Encoder-only、Encoder-Decoder、MoE、多模态等结构通过新增 `model_family` 规则集、扩展 `BlockStage`/`OpType` 表达，仍使用同一 Node/Edge 对象模型与 RuntimeGraph Forward 契约。Graph IR 用 `at::Tensor` 承载张量属性，RuntimeGraph 用 `TensorDesc` 承载同一语义。

### 4.6.2 融合规则表（首个模型族：Qwen3-0.6B）

规则表按 `model_family` 组织，是 Graph 的一等契约。本节给出 Qwen3-0.6B 的完整规则集作为首个实例；其他模型族按同一 schema 增加规则集，必要时扩展 `OpType` 与 `kernel_key`，但不改变 Graph 对象模型、I/O 契约与校验流程。

| 阶段 | Node 名 | op_type | fused_ops（有序） | 输入 | 输出 | 关键属性 | kernel_key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `layer.<i>.attn_norm` | `kRmsNorm` | `rmsnorm` | `hidden_in`、`attn_norm_weight` | `attn_normed` | `eps`、`precision` | `rms_norm` |
| 2 | `layer.<i>.qkv` | `kQkvProjection` | `q_linear`、`k_linear`、`v_linear`、`qk_rmsnorm`、`rope` | `attn_normed`、`q_weight`、`k_weight`、`v_weight`、`q_norm_weight`、`k_norm_weight`、`position_ids` | `q`、`k`、`v` | `num_heads`、`num_kv_heads`、`head_dim`、`rope_theta`、`rope_layout`、`has_qk_norm`、`qk_norm_eps`、`bias` | `qkv_projection` |
| 3 | `layer.<i>.attn` | `kAttention` | `kv_cache_write`、`qk_matmul`、`softmax`、`pv_matmul`、`output_projection` | `q`、`k`、`v`、`o_weight`；`kv_cache`/`block_table` 经 ForwardContext | `attn_out` | `causal`、`softmax_scale`、`sliding_window`、`has_output_projection` | `attention` |
| 4 | `layer.<i>.attn_residual` | `kResidualAdd` | `add` | `attn_out`、`hidden_in` | `hidden_mid` | `precision` | `residual_add` |
| 5 | `layer.<i>.mlp_norm` | `kRmsNorm` | `rmsnorm` | `hidden_mid`、`mlp_norm_weight` | `mlp_normed` | `eps`、`precision` | `rms_norm` |
| 6 | `layer.<i>.mlp_gate_up` | `kMlpGateUpAct` | `gate_linear`、`up_linear`、`silu`、`mul` | `mlp_normed`、`gate_weight`、`up_weight` | `mlp_act` | `activation="silu"`、`intermediate_size` | `mlp_gate_up_act` |
| 7 | `layer.<i>.mlp_down_residual` | `kMlpDownResidual` | `down_linear`、`add` | `mlp_act`、`down_weight`、`hidden_mid` | `hidden_out` | `precision` | `mlp_down_residual` |

块外算子规则：

| Node 名 | op_type | fused_ops | 输入 | 输出 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `embedding` | `kEmbedding` | `embedding_lookup` | `input_ids`、`embedding_weight` | `hidden` | 权重可被 LM Head 共享（tie） |
| `final_norm` | `kRmsNorm` | `rmsnorm` | `hidden`、`final_norm_weight` | `final_normed` | 末层归一化 |
| `lm_head` | `kLmHead` | `last_token_gather`、`linear` | `final_normed`、`last_token_indices`、`lm_head_weight` | `logits` | `gather_last_token` 默认 true；`last_token_indices` 由 Input Node 提供 |
| `sampling` | `kSampling` | `sampling` | `logits`；采样参数经 ForwardContext | `token_ids` | 固定为图节点，不融合 |
| `cast.<id>` | `kCast` | `cast` | 输入边 | 输出边 | 显式精度转换，融合屏障 |

### 4.6.3 阶段约束

以下约束以 Qwen3-0.6B 规则集为例；其他模型族在各自规则集中给出对应约束，但“一个 Node 只覆盖一个阶段”的边界规则通用。

**阶段 1/5：RMSNorm**

- 输入：hidden 张量与 norm 权重；输出：同 shape、同 dtype 的归一化张量。
- 属性：`eps` 来自 `ModelConfig.norm_eps`，节点可覆盖但必须大于 0。
- 不与前置/后置投影融合；原因：归一化需要独立归约，融合会迫使 Kernel 处理两种 layout，降低可复用性。

**阶段 2：QKV 投影 + 后处理**

- 输入：`attn_normed`、Q/K/V 权重、QK norm 权重、`position_ids`。
- 输出：`q`、`k`、`v` 三个设备张量。
- `fused_ops` 顺序固定为 `q_linear → k_linear → v_linear → qk_rmsnorm → rope`；Qwen3 的 q_norm/k_norm 在 head_dim 上执行，随后应用 RoPE。
- 属性：`has_qk_norm` 由 Adapter 根据模型结构设置；Qwen3-0.6B 为 true。`rope_layout` 由 Adapter 校验后设置为 `kNeox`；若模型使用其他布局，规则表需新增条目。
- 不包含 Attention 核心；不写 KV Cache（KV 写入属于阶段 3）。

**阶段 3：Attention**

- 输入：`q`、`k`、`v`、`o_weight`；`KVCacheView`、`AttentionMetadata` 经 ForwardContext 传入。
- 输出：`attn_out`。
- `fused_ops` 顺序固定为 `kv_cache_write → qk_matmul → softmax → pv_matmul → output_projection`。
- Prefill 与 Decode 使用同一 Node、同一 `kernel_key`；Backend 可在 `CreateKernel` 时创建包含 Prefill/Decode 两套 Program 的 Kernel，由 Kernel 依据 `AttentionMetadata.is_prefill`/`request_phases` 选择内部实现；Graph 不拆分 Node，也不在 Forward 中重建 Kernel。
- 若 Scheduler 保证一个 Step 内阶段一致，`request_phases` 可为空且 `has_mixed_phases=false`；若同一 Step 混合 Prefill/Decode，必须提供 `request_phases` 并置 `has_mixed_phases=true`，Attention Kernel 需支持混合批次。
- KV 写入在 Attention Kernel 内完成，避免额外遍历；写入位置由 `block_table` 与 `kv_len` 决定。
- 不包含残差相加；残差属于阶段 4。

**阶段 4：Attention 残差**

- 输入：`attn_out`、`hidden_in`；输出：`hidden_mid = attn_out + hidden_in`。
- 不融合其他算子；要求输入/输出 shape、dtype 完全一致。
- 选择单独节点的理由：Attention 输出 layout 由 Kernel 决定，残差需要与 block 输入布局对齐；独立节点便于后端分别优化，也便于数值排查。

**阶段 6：gate/up + 激活**

- 输入：`mlp_normed`、`gate_weight`、`up_weight`；输出：`mlp_act`。
- `fused_ops` 顺序固定为 `gate_linear → up_linear → silu → mul`。
- 属性：Qwen3 规则集阶段一只支持 `"silu"`；其他激活需在对应模型族规则集中扩展。
- gate/up 输出必须同 shape、同 dtype；逐元素乘在 Kernel 内完成。

**阶段 7：down + 残差**

- 输入：`mlp_act`、`down_weight`、`hidden_mid`；输出：`hidden_out = down(mlp_act) + hidden_mid`。
- `fused_ops` 顺序固定为 `down_linear → add`。
- 残差输入必须是阶段 4 的输出 `hidden_mid`，不得复用其他张量。
- 选择融合的理由：down GEMM 的 epilogue 可自然累加残差，减少一次全局内存写读。

### 4.6.4 融合边界与禁止项

| 编号 | 规则 |
| --- | --- |
| R1 | 一个融合 Node 只能覆盖一个阶段的算子；跨阶段合并一律拒绝 |
| R2 | 归一化不得与投影融合；Attention 残差不得并入 Attention |
| R3 | 不同 precision 的算子不得融合；显式 Cast 是融合屏障 |
| R4 | I/O Node、Sampling Node、Cast Node 不参与块融合 |
| R5 | 融合 Node 的内部中间张量不暴露为 Edge；`fused_ops` 记录其逻辑顺序 |
| R6 | 权重必须以 `is_weight=true` 的 Edge 挂接；一个融合 Node 可引用多个权重；Linear 的 bias（若存在）在同一阶段内作为额外权重边与 `fused_ops` 项处理，不跨阶段 |
| R7 | 规则表按 `model_family` 选择并版本化；转换报告必须记录 `fusion_rule_id` |
| R8 | 规则不满足时显式报错并指出规则 ID；不得静默拆分为多个 Node 后继续 |
| R9 | 符号形状不影响融合决策；融合在 Shape Infer 之前完成 |
| R10 | 量化只改变 Graph IR 量化元数据/TensorDesc 与 Kernel 执行路径，不改变融合骨架与节点边界 |

### 4.6.5 规则应用与报告

```cpp
// 融合规则条目：由 Graph 模块维护，Converter 查询使用。
struct FusionRule {
  std::string rule_id;                      // 规则唯一编号，如 "qwen3.block.qkv.v1"。
  std::string model_family;                 // 适用模型族，如 "qwen3"。
  BlockStage stage;                         // 所属骨架阶段。
  OpType node_op_type;                      // 融合后的节点类型。
  std::vector<std::string> required_op_kinds;  // 融合前的算子序列。
  std::map<std::string, AttributeValue> required_attributes;  // 必须匹配的属性。
  std::string kernel_key;                   // 逻辑 Kernel 键。
};

class FusionRuleTable {
 public:
  static const FusionRuleTable& Instance();

  // 按模型族、阶段与算子序列查找规则；未命中返回 nullptr。
  const FusionRule* Find(std::string_view model_family, BlockStage stage,
                         const std::vector<std::string>& op_kinds) const;

  const std::vector<FusionRule>& rules() const;
  std::string version() const;  // 规则表版本，写入图元数据与转换报告。
};
```

Converter 构图时：

1. 按模型族选择 `FusionRuleTable`。
2. Adapter 识别原始算子模式，得到 `op_kind` 序列。
3. 调用 `FusionRuleTable::Find(model_family, stage_id, op_kinds)`。
4. 命中后创建融合 Node，写入 `op_type`、`fused_ops`、`fusion_rule_id`、`kernel_key`。
5. 未命中时记录规则 ID、阶段、实际 `op_kinds`，返回 `INVALID_ARGUMENT` 或 `UNSUPPORTED_OP`；不得静默降级。
6. 转换报告输出：每层融合命中情况、未命中原因、规则表版本。

规则表版本字段 `fusion_rules_version` 写入图元数据与转换报告；规则变更必须提升版本并更新本文档。

### 4.6.6 Block 图示例（Qwen3-0.6B）

```text
hidden_in
  ├──▶ layer.i.attn_norm ──▶ layer.i.qkv ──▶ layer.i.attn ──▶ layer.i.attn_residual ──▶ hidden_mid
  │                                                                  ▲
  └──────────────────────────────────────────────────────────────────┘

hidden_mid
  ├──▶ layer.i.mlp_norm ──▶ layer.i.mlp_gate_up ──▶ layer.i.mlp_down_residual ──▶ hidden_out
  │                                                          ▲
  └──────────────────────────────────────────────────────────┘
```

整图：

```text
input.input_ids ──▶ embedding ──▶ layer.0 ... layer.N-1 ──▶ final_norm ──▶ lm_head ──▶ sampling ──▶ output.token_ids
input.position_ids ────────────────────────────────────────────────────────▶ 各层 qkv
input.last_token_indices ──────────────────────────────────────────────────▶ lm_head
```

## 4.7 RuntimeGraph.Forward

### 4.7.1 前置条件

`Forward` 返回成功前必须满足：

1. RuntimeGraph 已 `Finalize` 且结构校验通过。
2. `BindBackend` 已调用，`ctx.backend == runtime_graph.backend()`。
3. 所有计算 Node 已绑定非空 Kernel，`ValidateExecution()` 通过。
4. `ctx.stream` 非空。
5. 每个 Input/Output Node 在 `ctx.host_inputs`/`ctx.host_outputs` 中有对应 `binding_name` 的 `IoBinding`。
6. `IoBinding.host_ptr` 非空、`bytes`、`dtype`、`shape` 与节点 `host_tensor` 及具体形状匹配。
7. 含 Attention Node 时，`KVCacheView` 与 `AttentionMetadata` 已填充且字段自洽；`has_mixed_phases=true` 时 `request_phases` 非空且长度等于 `batch_size`。
8. 含 Sampling Node 时，`SamplingBatch.params` 与 `num_requests` 匹配。

### 4.7.2 执行算法

```cpp
Status RuntimeGraph::Forward(const ForwardContext& ctx) const {
  if (!finalized_ || released_) {
    return MakeStatus(ErrorCode::kStateError,
                      "[Graph] graph not finalized or already released");
  }
  if (backend_ == nullptr || ctx.backend != backend_) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "[Graph] backend mismatch or not bound");
  }
  if (ctx.stream == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument, "[Graph] stream is null");
  }

  // 按拓扑序执行；Input 在前、计算在中、Output 在后。
  for (NodeId id : topo_order_) {
    const RuntimeNode& node = nodes_[id];
    if (node.op_type == OpType::kInput) {
      Status status = ExecuteInput(node, ctx);
      if (!status.ok()) return status;
      continue;
    }
    if (node.op_type == OpType::kOutput) {
      Status status = ExecuteOutput(node, ctx);
      if (!status.ok()) return status;
      continue;
    }

    Kernel* kernel = node_kernels_[id];
    if (kernel == nullptr) {
      return MakeStatus(ErrorCode::kUnsupportedOp,
                        "[Graph] kernel not bound: node=" + node.name);
    }
    KernelContext kernel_ctx = BuildKernelContext(node, ctx);
    Status status = kernel->Execute(kernel_ctx);
    if (!status.ok()) {
      return PrefixStatus(status, "[Graph] node=" + node.name);
    }
  }
  return Status::Ok();
}
```

### 4.7.3 Input/Output 执行

```cpp
Status RuntimeGraph::ExecuteInput(const RuntimeNode& node,
                                  const ForwardContext& ctx) const {
  // 1. 按 binding_name 查找 IoBinding。
  // 2. 校验 host_ptr、bytes、dtype、shape 与 node.host_tensor（TensorDesc）一致。
  // 3. 解析输出边的设备地址：arena_base + edge.memory_offset + tensor.offset。
  // 4. 调用 Backend::Upload(device_ptr, host_ptr, bytes, ctx.stream)。
  // 5. 只入队，不在此处同步；返回后主机缓冲仍需保持有效。
}

Status RuntimeGraph::ExecuteOutput(const RuntimeNode& node,
                                   const ForwardContext& ctx) const {
  // 1. 按 binding_name 查找 IoBinding。
  // 2. 校验 host_ptr、bytes、dtype、shape。
  // 3. 解析输入边的设备地址。
  // 4. 调用 Backend::Download(host_ptr, device_ptr, bytes, ctx.stream)。
  // 5. 只入队；Executor 在 Backend::Synchronize 后读取 host_ptr。
}
```

### 4.7.4 KernelContext 映射

| KernelContext 字段 | 来源 |
| --- | --- |
| `node` | 当前 Node 的只读指针 |
| `input_tensors` | `input_edges` 对应的 TensorDesc 数组，按端口顺序 |
| `output_tensors` | `output_edges` 对应的 TensorDesc 数组，按端口顺序 |
| `input_ptrs` | arena 基址 + Edge.memory_offset + TensorDesc.offset |
| `output_ptrs` | 同上 |
| `attributes` | Node.attributes |
| `kv_cache` | `ForwardContext.kv_cache_view` |
| `attention` | `ForwardContext.attention_metadata` |
| `sampling` | `ForwardContext.sampling_params` |
| `workspace` | `ForwardContext.workspace` / `workspace_bytes` |
| `stream` | `ForwardContext.stream` |

`KernelContext` 的最终字段由《Kernel 公共契约》冻结；本表是 Graph 侧必须提供的映射。

### 4.7.5 异步与同步语义

- `RuntimeGraph.Forward` 只负责把 Input Upload、Kernel Execute、Output Download 按序入队，**不调用** `Backend::Synchronize`。
- 主机输入缓冲在 Forward 调用期间及设备完成读取前必须有效；主机输出缓冲在 `Backend::Synchronize` 前不得读取。
- 同步由 Executor 在 Forward 返回后统一执行；RuntimeGraph 不感知请求队列与 Step 边界。
- 任一步失败立即返回；已入队的操作不回滚，由 Engine 的设备错误处理策略决定是否受控停止。

### 4.7.6 错误路径

| 场景 | 错误码 |
| --- | --- |
| RuntimeGraph 未 Finalize、Backend 未绑定、Kernel 未绑定 | `STATE_ERROR` / `UNSUPPORTED_OP` |
| `ctx.backend` 与绑定 Backend 不一致 | `INVALID_ARGUMENT` |
| 缺少 Input/Output binding | `INVALID_ARGUMENT` |
| binding 的 bytes/dtype/shape 不匹配 | `INVALID_ARGUMENT` |
| `host_ptr == nullptr` | `INVALID_ARGUMENT` |
| `Backend::Upload/Download` 失败 | 透传 `BUFFER_ERROR` / `DEVICE_ERROR` |
| `Kernel::Execute` 失败 | 透传 `KERNEL_ERROR` / `DEVICE_ERROR` |
| 执行表内部不变量破坏 | `INTERNAL` |

错误消息格式：`[Graph] <CODE> node=<name> edge=<name> binding=<name>: <detail>`。

## 4.8 运行时执行表与资源释放

### 4.8.1 绑定

```cpp
Status RuntimeGraph::BindBackend(Backend* backend) {
  // 1. backend 不得为空。
  // 2. 只允许绑定一次；重复绑定返回 STATE_ERROR。
  // 3. 保存非拥有指针，用于 Release 时销毁 Kernel。
}

Status RuntimeGraph::BindArena(int32_t arena_id, void* device_ptr, uint64_t bytes) {
  // 1. arena_id 必须在 MemoryPlan 声明的范围内。
  // 2. device_ptr 不得为空；bytes 必须覆盖该 arena 内所有 Edge 的偏移范围。
  // 3. 保存非拥有指针；RuntimeGraph 不负责分配与释放 arena。
}

Status RuntimeGraph::BindArenas(const std::vector<ArenaBinding>& arenas) {
  // 批量绑定；任一 arena 非法则整体失败，已绑定项保持不变。
}

Status RuntimeGraph::BindKernel(std::string_view node_name, Kernel* kernel) {
  // 1. 查找节点；不存在返回 INVALID_ARGUMENT。
  // 2. I/O Node 不允许绑定 Kernel，返回 INVALID_ARGUMENT。
  // 3. kernel 不得为空；同一节点重复绑定返回 STATE_ERROR。
  // 4. 按 NodeId 写入 node_kernels_，Forward 热路径不再做名称查找。
}

Status RuntimeGraph::ValidateExecution() const {
  // 1. Backend 已绑定。
  // 2. 所有计算 Node 的 node_kernels_[id] 非空。
  // 3. I/O Node 对应槽位为空。
  // 4. 执行表大小等于 nodes_.size()。
  // 5. 所有 Edge（含权重边）的 arena 已绑定，且 offset + size 不越界。
}
```

### 4.8.2 释放

```cpp
Status RuntimeGraph::Release() {
  // 1. 已释放则直接返回 OK，保证幂等。
  // 2. 若从未绑定任何 Kernel，直接清空并返回 OK。
  // 3. 存在 Kernel 但 backend_ 为空时返回 STATE_ERROR。
  // 4. 按逆拓扑序逐计算 Node 调用 Backend::DestroyKernel。
  // 5. 单个 Kernel 销毁失败时继续释放其余 Kernel，返回首个错误。
  // 6. 清空执行表与 arena 绑定并置 released_=true；保留 RuntimeGraph 结构供诊断。
}
```

- **所有权**：RuntimeGraph 拥有 Kernel 句柄的释放职责；Backend 拥有 Kernel 的创建与具体销毁实现；RuntimeGraph 不拥有 Backend、Buffer、arena。
- **析构**：`~RuntimeGraph()` 不调用 Backend；调试构建断言执行表为空，防止泄漏。
- **顺序**：必须在 `Backend::Deinit` 之前调用 `RuntimeGraph::Release`；释放顺序见上位文档 4.3.6。
- **幂等**：重复 Release 返回 `OK`；Release 后 Forward 返回 `STATE_ERROR`。

## 4.9 版本化与序列化契约

### 4.9.1 版本规则

- 当前 `schema_version_major = 1`，`schema_version_minor = 0`。
- 读取端：主版本必须等于当前主版本；次版本小于等于当前次版本可读；次版本高于当前实现返回 `VERSION_MISMATCH`。
- 升级主版本：字段删除、语义变更、枚举值重排等不兼容变更。
- 升级次版本：仅新增可选字段或新增枚举值，旧读取端可忽略。

### 4.9.2 序列化字段

Serializer 负责把 Graph IR 映射为 Protobuf；映射前先通过 `ToTensorDesc` 把 `at::Tensor` 转为 `TensorDesc`，Graph IR 保证以下字段可只读访问：

| 类别 | 字段 |
| --- | --- |
| 版本 | `schema_version_major`、`schema_version_minor`、`fusion_rules_version` |
| 模型 | `ModelConfig` 全部字段（`dtype` 由 `at::ScalarType` 映射为 `DataType`） |
| 权重 | `WeightDesc.ref/source_name/tensor`（转换为 TensorDesc）+ Serializer 写入的 `data_offset/data_bytes/checksum` |
| 节点 | `name`、`op_type`、`fused_ops`、`attributes`、`input_edges`、`output_edges`、`precision`（`at::ScalarType` → `DataType`）、`kernel_key`、`fusion_rule_id`、`binding_name`、`copy_kind`、`host_tensor`（`at::Tensor` → TensorDesc） |
| 边 | `name`、`tensor`（`at::Tensor` → TensorDesc：dtype/shape/stride/layout/device/quantization；`offset` 运行时填充，全静态 `size_bytes` 可序列化）、`is_weight`、`weight_ref`、`producer`、`consumers` |
| I/O | `input_nodes`、`output_nodes` 的节点名 |
| 运行时 | `node_kernels_`、`arena_id`、`memory_offset` **不序列化** |

- `topo_order` 由 Loader/RuntimeGraphBuilder 重建时重新计算并校验，不作为持久化事实源。
- `arena_id`/`memory_offset` 由 Loader 根据 MemoryPlan 填写，不进入模型文件。
- Canonical JSON 仅用于测试/调试，字段顺序固定、浮点格式固定、不包含指针与运行时字段。

### 4.9.3 确定性

- `std::map` 属性按 key 排序输出。
- 节点、边、权重按插入顺序输出；拓扑序按 NodeId 升序。
- 浮点数使用最短可往返表示；避免平台相关格式。
- 同一图多次 `ToCanonicalJson()` 输出字节一致。

## 4.10 Python 绑定

### 4.10.1 暴露接口

| Python 对象 | 对应 C++ | 说明 |
| --- | --- | --- |
| `Dim` | `Dim` | 常量维/符号维 |
| `SymbolicShape` | `SymbolicShape` | 符号形状元数据 |
| `QuantizationDesc` | `QuantizationDesc` | 量化描述（预留） |
| `WeightDesc` | `WeightDesc` | 权重表条目；`tensor` 暴露为 `torch.Tensor` |
| `FusedOp` | `FusedOp` | 融合前散算子 |
| `Node` | `Node` | Graph IR 节点；张量属性暴露为 `torch.Tensor`/`torch.dtype` |
| `Edge` | `Edge` | Graph IR 边；`tensor` 暴露为 `torch.Tensor` |
| `ModelConfig` | `ModelConfig` | 模型配置；`dtype` 暴露为 `torch.dtype` |
| `GraphBuilder` | `GraphBuilder` | 构造 Graph IR |
| `Graph` | `Graph` | 只读 Graph IR；不提供 Forward |
| `ValidationReport` | `ValidationReport` | 校验报告 |
| `FusionRuleTable` | `FusionRuleTable` | 规则查询 |
| `OpType`/`CopyKind` | 同名枚举 | Graph IR 枚举常量 |

### 4.10.2 绑定规则

1. C++ 为唯一事实源；Python 侧不定义第二套 Node/Edge 结构，也不定义自定义 Tensor/dtype/device 类。
2. `GraphBuilder.add_node(node)` 返回 `NodeId`；失败抛出 `GraphError`，携带 `code`、`node_name`、`edge_name`、`detail`。
3. `Graph.node(name)`/`Graph.edge(name)` 返回**只读视图**；`at::Tensor` 通过 LibTorch caster 暴露为共享存储的 `torch.Tensor`，不复制底层数据，避免 Python 侧悬空引用。
4. Graph IR 不暴露 `Forward`；运行时 `RuntimeGraph.Forward` 只在 C++ 侧使用，不通过 pybind11 导出。
5. 所有 C++ 异常在绑定层捕获并转换为 `GraphError`，不得穿越 Python 边界。
6. Python 侧构图工具放在 `Graph/cpp/python/graph_tools.py`，提供节点/边命名与批量添加辅助，不改变 C++ 校验规则。

### 4.10.3 双端一致性

- Python 与 C++ 使用同一 `GraphBuilder`、`GraphValidator` 与同一套 `at::Tensor`；Python 侧张量就是 `torch.Tensor`。
- 测试流程：Python 用 `torch.Tensor` 构建首个模型族（Qwen3）图 → `to_canonical_json()`；C++ 用同一描述与 `at::Tensor` 构建 → `ToCanonicalJson()`；逐字节比较。
- 差异定位：Canonical JSON 按节点/边/属性排序，差异报告输出首个不一致路径。

## 4.11 错误处理

| 场景 | 错误码 | 处理 |
| --- | --- | --- |
| 名称重复、字段缺失、非法枚举 | `INVALID_ARGUMENT` | 拒绝构图/加载，报告节点/边 |
| 成环、悬空边、端口重复引用 | `INVALID_ARGUMENT` | 拒绝构图，报告边名 |
| dtype/shape 不匹配 | `INVALID_ARGUMENT` | 拒绝构图，报告生产/消费边 |
| I/O 边方向或 dtype 非法 | `INVALID_ARGUMENT` | 拒绝构图/加载 |
| 融合规则未命中/跨阶段 | `INVALID_ARGUMENT` | 拒绝构图，报告 rule_id |
| schema 主版本不兼容 | `VERSION_MISMATCH` | 拒绝加载 |
| 未 Finalize/未 Bind 就 Forward | `STATE_ERROR` | 返回错误，不执行 |
| 计算节点缺 Kernel | `UNSUPPORTED_OP` | 拒绝执行，报告节点名 |
| Upload/Download 失败 | `BUFFER_ERROR`/`DEVICE_ERROR` | 透传后端错误 |
| Kernel 执行失败 | `KERNEL_ERROR`/`DEVICE_ERROR` | 透传并附加节点名 |
| 内部不变量破坏 | `INTERNAL` | 记录日志并返回 |

原则：

- 错误不携带裸指针；消息只包含名称与数值。
- 构图期错误可累积；Forward 期错误在首个失败点返回。
- RuntimeGraph 不吞掉 Backend/Kernel 错误码；只附加节点上下文。
- 所有公共接口不抛出异常；内存分配失败在构图期返回 `INTERNAL`，Forward 稳态不分配。

## 4.12 性能与内存

| 项目 | 目标/策略 | 验证 |
| --- | --- | --- |
| 校验复杂度 | O(V+E)，单次遍历邻接表 | PT-GRAPH-001 |
| 拓扑排序 | Kahn + 最小堆（按 NodeId），O((V+E) log V)，首个验证模型（Qwen3）图规模可接受 | UT-GRAPH-002 |
| Forward 调度 | 按 `topo_order_` 连续数组遍历；`node_kernels_` 按 NodeId 索引；switch 分派 | PT-GRAPH-002 |
| 稳态分配 | Forward 不分配堆内存；错误路径除外 | 代码审计、分配计数 |
| 图元数据 | RuntimeGraph 元数据：首个验证模型 Qwen3-0.6B 目标 < 5 MB（不含权重/缓冲）；其他模型族按节点/边规模线性估算；Graph IR 仅存在于工具链 | PT-GRAPH-003 |
| Python 绑定 | 仅离线/测试使用；构图目标 < 100 ms/模型（待测） | PT-GRAPH-004 |
| 缓存友好 | Node/Edge 连续存储；拓扑序为 `uint32_t` 数组 | 代码审计 |

## 4.13 线程安全与并发

- `Finalize` 后 Graph IR 与 RuntimeGraph 结构均只读；`RuntimeGraph::Forward` 为 `const`，不修改图结构。
- RuntimeGraph 不提供内部锁；同一 RuntimeGraph 的并发 Forward 需要调用方提供独立 Stream、workspace 与缓冲，并由 Engine/Executor 串行化。
- `BindBackend`/`BindKernel`/`Release` 不是线程安全的，只允许在加载/销毁阶段调用，且不得与 Forward 并发。
- Python 绑定层：构图与校验持有 GIL；Graph IR 不导出 Forward；调用方不得在 Python 侧并发修改 Graph IR 对象。
- 多 Engine 实例各自持有独立 RuntimeGraph 与 Backend，互不共享可变状态。

---

# 5. 实施步骤

## 5.1 阶段划分

| 阶段 | 目标 | 主要产出 | 退出条件 |
| --- | --- | --- | --- |
| G0 | 冻结双表示与公共类型 | 本文评审通过；`graph_common.h`、Graph IR/RuntimeGraph 头文件草案；与 Kernel/Backend 契约、LibTorch/PyTorch 版本对齐 | 接口评审通过，无未决类型；IR/runtime 依赖边界明确 |
| G1 | 双表示对象模型与 Builder | Graph IR/RuntimeGraph 数据结构、访问器、GraphBuilder、RuntimeGraphBuilder、拓扑排序 | 单元测试通过，合法图可构造 |
| G2 | 校验器 | V1~V16 双表示共用校验、V17 RuntimeGraph 校验、错误报告、Canonical JSON | 每条规则至少一个负向用例 |
| G3 | 融合规则 | FusionRuleTable、Qwen3 规则集（首个实例）、转换报告接口 | 规则表与首个模型族图快照一致 |
| G4 | RuntimeGraph Forward 与执行表 | Bind/ValidateExecution/Forward/Release；Mock Backend/Kernel | Mock 下 H2D→计算→D2H 通过 |
| G5 | Python 绑定 | pybind11 + LibTorch caster、构图工具、双端一致性测试 | TC-GRAPH-02 通过；Python 属性为 torch 类型 |
| G6 | 转换与集成加固 | `ToTensorDesc`、与 Converter/Serializer/Loader/Executor 联调；无 LibTorch 依赖审计；性能与稳定性 | TC-GRAPH-01、集成用例与依赖审计通过 |

## 5.2 任务拆解

| 编号 | 任务 | 主要内容 | 依赖 | 验收 |
| --- | --- | --- | --- | --- |
| G-T1 | 双表示边界与公共类型冻结 | 与 Kernel/Backend 对齐 OpType/Status/KernelContext；确定 LibTorch/PyTorch 版本；发布 `graph_common.h`、Graph IR 与 RuntimeGraph 头文件草案 | 上位文档 T1-1、T1-8 | 接口评审通过；依赖边界评审通过 |
| G-T2 | 双表示数据结构与访问器 | Graph IR：SymbolicShape/Edge/Node/WeightDesc/ModelConfig/Graph（at::Tensor）；RuntimeGraph：Dim/TensorDesc/RuntimeEdge/RuntimeNode/RuntimeGraph/ForwardContext | G-T1 | 单元测试通过 |
| G-T3 | 双 Builder | GraphBuilder（at::Tensor 构图）与 RuntimeGraphBuilder（TensorDesc 重建）；命名与引用解析 | G-T2 | 合法图构造、重复名拒绝、拓扑序一致 |
| G-T4 | GraphValidator | V1~V16 双表示分派校验、V17 RuntimeGraph 校验；ValidationReport；错误定位 | G-T2、G-T3 | 规则覆盖 100% |
| G-T5 | FusionRuleTable | 规则 schema、按 model_family 查询、Qwen3-0.6B 规则集（首个实例）、版本 | G-T2 | 规则表与本文一致；多模型族查询隔离 |
| G-T6 | RuntimeGraph Forward 与执行表 | BindBackend/BindArena/BindKernel/ValidateExecution/Forward/Release | G-T2、G-T4、G-T5 | Mock 集成通过 |
| G-T7 | Python 绑定 | pybind11 类、LibTorch type caster、异常转换、GIL、构图工具 | G-T2、G-T3 | TC-GRAPH-02 通过；Python 侧 `torch.Tensor` 断言通过 |
| G-T8 | 转换与序列化契约 | `ToTensorDesc`、IR→RuntimeGraph、权重 CPU/连续化/列主序转置、字段清单、版本规则、Canonical JSON；与 Serializer 联调 | G-T2、G-T3 | 转换逐项一致、往返一致 |
| G-T9 | 测试与基准 | 单元/集成/性能/模糊测试；无 LibTorch 依赖审计；测试报告 | G-T3~G-T8 | 全部通过或记录偏差 |
| G-T10 | 文档与 skill | 更新本文；沉淀 Graph 双表示/校验/融合规则 skill | G-T9 | 四件套齐全 |

## 5.3 依赖关系

- 内部：G-T1 → G-T2 → G-T3/G-T4/G-T5 → G-T6 → G-T7/G-T8 → G-T9 → G-T10。
- 外部：
  - G-T1 依赖 Kernel/Backend 公共接口草案（上位文档 T1-8）与 LibTorch/PyTorch 版本选型。
  - G-T5/G-T8 依赖 Converter（T1-2）与 Serializer（T1-4）的字段需求。
  - G-T6 依赖 Loader（T1-7）与 Executor（T1-9）的绑定/调用时序。
  - G-T7/G-T8 依赖 Python 侧 `torch` 与 C++ LibTorch 的 ABI 一致性验证。
- 可并行：G-T3、G-T4、G-T5 在 G-T2 后并行；G-T7 与 G-T6 可并行；G-T9 随各任务持续执行。

## 5.4 里程碑

| 里程碑 | 内容 | 判定 |
| --- | --- | --- |
| M-G0 | 双表示接口与依赖边界冻结 | 头文件评审通过；IR/runtime 依赖清单确认 |
| M-G1 | 双表示对象模型 + 双 Builder + 校验器 | UT-GRAPH-001~010、UT-GRAPH-028/029 通过 |
| M-G2 | 首个模型族（Qwen3）融合规则 + 图快照 | 规则表与 Qwen3 图 Canonical JSON 锁定 |
| M-G3 | IR→RuntimeGraph 转换 + Mock Forward | UT-GRAPH-025、UT-GRAPH-012~015 通过；PT-GRAPH-005 记录 |
| M-G4 | Python 双端一致 | TC-GRAPH-02 与 UT-GRAPH-026 通过 |
| M-G5 | 与 Converter/Serializer/Loader/Executor 集成 + 无 LibTorch 审计 | TC-GRAPH-01、IT-GRAPH-001~009 通过 |

## 5.5 风险与应对

| 风险 | 影响 | 概率 | 应对 |
| --- | --- | --- | --- |
| Kernel/Backend 接口变更 | Graph 头文件与 KernelContext 反复修改 | 中 | G0 先冻结最小契约；变更走评审并同步本文 |
| LibTorch/PyTorch ABI 不匹配 | Python `torch.Tensor` 与 C++ `at::Tensor` 无法互操作或崩溃 | 中 | 锁定同一版本与编译选项；CI 做 import/构造冒烟；禁止混用系统 torch |
| 运行时误链 LibTorch | 移动端包体膨胀、ABI 冲突 | 中 | RuntimeGraph 头文件审计；CMake 目标隔离；二进制依赖清单检查（UT-GRAPH-027、IT-GRAPH-009） |
| 双表示语义漂移 | IR 与 RuntimeGraph 的 dtype/shape/量化不一致 | 中 | 单一 `ToTensorDesc` 实现；转换契约测试；Canonical JSON 对照 |
| 融合规则过刚 | 新模型结构无法接入 | 中 | 规则表版本化；属性化差异；明确错误而非静默降级 |
| Python/C++ 双端漂移 | 图不一致，工具链与运行时错位 | 中 | Canonical JSON 逐字节比较；CI 固定用例 |
| Forward 调度开销 | 小模型 Decode 性能下降 | 中 | 连续数组 + 索引执行表；早期 micro-benchmark |
| Kernel 句柄所有权错误 | 重复释放/泄漏 | 中 | Release 幂等；逆拓扑序；ASan 与句柄计数 |
| 序列化字段漂移 | 旧模型文件无法加载 | 中 | 版本规则 + 字段契约 + TC-SER-01 回归 |
| Qwen3 结构细节差异 | 融合规则与实际模型不符 | 中 | 以模型源码与 `config.json` 为准；转换报告记录；Converter 校验 |
| 测试模型文件缺失 | 集成测试无法执行 | 低 | 先用合成图与 Mock；模型文件到位后补金标准 |

---

# 6. 测试与验证设计

## 6.1 测试策略

| 层级 | 目标 | 手段 |
| --- | --- | --- |
| 单元测试 | Graph IR/RuntimeGraph 对象模型、双 Builder、校验、融合、转换、Forward、Release | GoogleTest + Mock Backend/Kernel |
| 集成测试 | Converter → Graph IR → ToTensorDesc/Serializer → Loader → RuntimeGraph → Executor 贯通 | 合成图 + Qwen3-0.6B 图快照 |
| 双端测试 | Python `torch.Tensor` 构图与 C++ `at::Tensor` 构图一致 | Canonical JSON 逐字节比较 + 类型断言 |
| 依赖审计 | RuntimeGraph 不依赖 LibTorch | 头文件扫描 + 独立构建 + 二进制依赖检查 |
| 性能测试 | 校验、调度、转换、内存开销 | 主机基准；参考机后续补 |
| 健壮性测试 | 非法图、空指针、重复绑定、随机图 | 负向用例 + 模糊测试 + ASan |

## 6.2 测试环境

| 项目 | 配置 |
| --- | --- |
| 主机 OS | Windows 11 / Ubuntu 22.04（CI 可用 Linux） |
| 编译器 | MSVC 2022 / GCC 11+ / Clang 14+ |
| 语言标准 | C++17 |
| 构建 | CMake 3.20+ |
| 测试框架 | GoogleTest |
| LibTorch | 与 Python `torch` 同版本、同 ABI 的 LibTorch（版本在 G-T1 锁定）；仅工具链目标链接 |
| Python | 3.10+；pybind11 2.11+；`torch` 与 LibTorch 同版本 |
| 构建目标 | `graph_common`、`graph_ir`、`graph_runtime`、`graph_python` 分离；`graph_runtime` 不链接 LibTorch |
| Mock 后端 | `Runtime/backend/mock/`，实现 Upload/Download 与模板 Kernel |
| 模型数据 | 首个验证模型 Qwen3-0.6B 的 `config.json` 与图快照；其他模型族按同一 schema 生成快照；单元测试不依赖真实权重 |
| 目标设备 | Android arm64 + Adreno（阶段一后续） |

## 6.3 单元测试用例

| 用例 | 对应需求 | 场景 | 通过标准 |
| --- | --- | --- | --- |
| UT-GRAPH-001 | G-FR-01/11 | 构造最小合法图：1 个 Input、1 个计算 Node、1 个 Output | Finalize 成功；节点/边/拓扑序正确 |
| UT-GRAPH-002 | G-FR-11 | 多分支图，多个就绪节点 | 拓扑序确定，按 NodeId 升序；重复运行一致 |
| UT-GRAPH-003 | G-FR-04 | 构造环 | 返回 `INVALID_ARGUMENT`，定位到成环边 |
| UT-GRAPH-004 | G-FR-04 | 悬空边、端口重复引用 | 返回 `INVALID_ARGUMENT`，定位到边/节点 |
| UT-GRAPH-005 | G-FR-04 | 节点/边/binding 重名 | 返回 `INVALID_ARGUMENT`，指出重复名称 |
| UT-GRAPH-006 | G-FR-03 | 缺少 Input 或 Output Node | 返回 `INVALID_ARGUMENT` |
| UT-GRAPH-007 | G-FR-03 | Input 有入边、Output 有出边、I/O 边为权重 | 返回 `INVALID_ARGUMENT` |
| UT-GRAPH-008 | G-FR-06 | I/O host/device dtype 不一致且无 Cast | 返回 `INVALID_ARGUMENT` |
| UT-GRAPH-009 | G-FR-04 | 权重边缺 weight_ref、非权重边带 weight_ref | 返回 `INVALID_ARGUMENT` |
| UT-GRAPH-010 | G-FR-04 | 生产/消费 dtype、rank、常量维不匹配 | 返回 `INVALID_ARGUMENT`，指出边名 |
| UT-GRAPH-011 | G-FR-08/09 | 融合节点跨阶段、fused_ops 顺序错误 | 返回 `INVALID_ARGUMENT`，指出 rule_id |
| UT-GRAPH-012 | G-FR-05 | Mock Kernel 记录调用顺序 | 顺序与拓扑序一致；Input 先于计算，Output 后于计算 |
| UT-GRAPH-013 | G-FR-05/06 | Mock Backend 验证 Upload/Download 参数 | 设备地址、host 地址、bytes、stream 正确 |
| UT-GRAPH-014 | G-FR-06 | 缺少 binding、bytes/dtype/shape 不匹配、空指针 | 返回 `INVALID_ARGUMENT` |
| UT-GRAPH-015 | G-FR-12 | BindBackend/BindArena/BindKernel 重复调用；arena 越界；I/O Node 绑定 Kernel；Release 幂等 | 非法绑定被拒绝；Release 后执行表与 arena 绑定为空；重复 Release 返回 OK |
| UT-GRAPH-016 | G-FR-13 | 主版本不兼容、次版本过高 | 返回 `VERSION_MISMATCH` |
| UT-GRAPH-017 | G-FR-13/18 | 同一图多次 Canonical JSON | 字节级一致；字段顺序稳定 |
| UT-GRAPH-018 | G-FR-16 | Python（`torch.Tensor`）与 C++（`at::Tensor`）构造同一最小图 | Canonical JSON 差异为 0 |
| UT-GRAPH-019 | G-NFR-01 | 1000 节点随机 DAG 校验 | 时间与内存符合目标；无崩溃 |
| UT-GRAPH-020 | G-NFR-07 | 模糊测试：随机名称、随机边、随机 dtype | 只返回错误码，不崩溃、不泄漏 |
| UT-GRAPH-021 | G-FR-07 | 缺少 Sampling、多个 Sampling、Sampling 在 LmHead 之前 | 返回 `INVALID_ARGUMENT`，指出节点名 |
| UT-GRAPH-022 | G-FR-14 | 构造符号维张量与符号维不匹配的边 | 合法符号维通过；名称不一致被拒绝 |
| UT-GRAPH-023 | G-FR-15 | Graph IR Edge/TensorDesc 缺省量化字段与填写 INT8 量化字段 | 两种图均可构造；FP16 路径校验不受影响；转换后量化元数据保留 |
| UT-GRAPH-024 | G-FR-09 | FusionRuleTable 多模型族查询：Qwen3 规则命中、未知 `model_family` 查询、规则 ID 全局唯一 | 未知模型族返回 nullptr；Qwen3 规则不受影响；`rule_id` 无冲突 |
| UT-GRAPH-025 | G-FR-19 | 构造非连续张量、列主序 2-D 权重、含符号维的边，调用 `ToTensorDesc` | dtype/shape/stride/layout/device/offset/size_bytes 正确；列主序权重转置后数据一致；SymbolicShape 保留 |
| UT-GRAPH-026 | G-FR-16 | Python 侧访问 Node/Edge/ModelConfig 属性并检查类型 | `isinstance(x, torch.Tensor)`、`isinstance(dtype, torch.dtype)` 均为真；不存在自定义张量类 |
| UT-GRAPH-027 | G-NFR-04 | 扫描 RuntimeGraph 头文件并构建仅 runtime 目标 | 无 `<ATen/...>`、`<torch/...>`、`c10/` 引用；二进制依赖清单无 libtorch |
| UT-GRAPH-028 | G-FR-01/11 | 仅用 TensorDesc 通过 RuntimeGraphBuilder 构造图，与 IR 同名图比较 | RuntimeGraph 不含 torch 类型；节点/边/拓扑序与 IR 一致 |
| UT-GRAPH-029 | G-FR-04 | 同一非法结构（成环、悬空边）分别构造 Graph IR 与 RuntimeGraph | 两表示返回同类错误码并定位到同一节点/边 |

## 6.4 集成测试用例

| 用例 | 对应需求 | 场景 | 通过标准 |
| --- | --- | --- | --- |
| IT-GRAPH-001 | G-FR-10/19 | Converter 构建 Qwen3-0.6B Graph IR（`at::Tensor`）→ `ToTensorDesc`/Serializer → Loader 重建 RuntimeGraph → 比较 | 节点/边/权重/属性、dtype/shape/量化一致；融合规则命中记录完整 |
| IT-GRAPH-002 | G-FR-05/09 | Loader 重建 RuntimeGraph → Mock Backend 绑定 Kernel → Executor Forward | 执行顺序正确；H2D/D2H 数据一致 |
| IT-GRAPH-003 | G-FR-12 | Mock Backend 端到端：Input → 计算 → Output → Release | 无泄漏；句柄计数归零 |
| IT-GRAPH-004 | G-FR-03 | 多 Input/Output Node 图 | 所有 binding 正确匹配；数据一致 |
| IT-GRAPH-005 | G-FR-17 | Kernel 返回错误、Upload/Download 失败 | 错误码与节点名透传；引擎不崩溃 |
| IT-GRAPH-006 | G-FR-12 | Release 期间单个 DestroyKernel 失败 | 其余 Kernel 继续释放；返回首个错误；无泄漏 |
| IT-GRAPH-007 | G-FR-13 | 旧次版本图加载、重大版本图拒绝 | 兼容/拒绝行为符合 4.9 |
| IT-GRAPH-008 | G-FR-18 | Qwen3 Graph IR 与 RuntimeGraph 统计与快照 | 节点数/边数/权重数/op 统计一致且可复现 |
| IT-GRAPH-009 | G-NFR-04 | 以仅 RuntimeGraph 的目标构建并运行 H2D→计算→D2H | 构建与运行均不需要 LibTorch；依赖清单无 libtorch |

## 6.5 性能与稳定性测试

| 用例 | 指标 | 通过标准 |
| --- | --- | --- |
| PT-GRAPH-001 | Qwen3-0.6B 图校验耗时 | 记录实测值；目标 ≤ 10 ms（主机，待锁定） |
| PT-GRAPH-002 | Forward 每计算节点调度开销 | Mock Kernel 下目标 ≤ 2 μs/节点（主机，待锁定） |
| PT-GRAPH-003 | RuntimeGraph 元数据内存 | 目标 < 5 MB（不含权重/缓冲，待实测） |
| PT-GRAPH-004 | Python 构图耗时 | 记录实测值；目标 < 100 ms/模型（待锁定） |
| PT-GRAPH-005 | Graph IR→RuntimeGraph 转换耗时 | Qwen3-0.6B 图记录实测值；转换不进入运行时热路径 |
| ST-GRAPH-001 | 重复 Bind/Release 1000 次 | 无泄漏、无崩溃、句柄计数归零 |
| ST-GRAPH-002 | 随机非法图 10000 次 | 无崩溃；错误码合法；内存不增长 |

性能数字为初始目标，必须由实际测试报告校准后锁定；不同设备结果不得混用。

## 6.6 通过标准

- 6.3、6.4 全部用例通过；失败用例必须修复，不得删除或放宽断言。
- TC-GRAPH-01、TC-GRAPH-02 通过。
- 校验规则 V1~V17 每条至少一个负向用例；融合规则表每个阶段至少一个正向与一个边界用例。
- Forward 覆盖 Input、计算、Output、Sampling、Cast、多 I/O 节点路径。
- Graph IR→RuntimeGraph 转换逐项一致；Python 侧属性类型断言通过；RuntimeGraph 无 LibTorch 依赖审计通过。
- ASan/Valgrind 无泄漏、无越界；模糊测试无崩溃。
- 性能目标达到，或提供实测数据、偏差原因与后续优化计划。
- 代码通过 Google Style 与中文注释审计。

## 6.7 需求追溯矩阵

| 需求 | 设计章节 | 测试用例 |
| --- | --- | --- |
| G-FR-01 | 4.2.1、4.2.7、4.2.10、4.4 | UT-GRAPH-001、UT-GRAPH-002、UT-GRAPH-028 |
| G-FR-02 | 4.2.4、4.2.5、4.2.8 | UT-GRAPH-001、UT-GRAPH-005 |
| G-FR-03 | 4.2.5、4.2.8、4.5 | UT-GRAPH-006、UT-GRAPH-007 |
| G-FR-04 | 4.5 | UT-GRAPH-003~010、UT-GRAPH-029 |
| G-FR-05 | 4.7 | UT-GRAPH-012、IT-GRAPH-002 |
| G-FR-06 | 4.2.9、4.7.3 | UT-GRAPH-013、UT-GRAPH-014 |
| G-FR-07 | 4.3.1、4.5 V12 | UT-GRAPH-012、UT-GRAPH-021、IT-GRAPH-002 |
| G-FR-08 | 4.6.1、4.6.4 | UT-GRAPH-011 |
| G-FR-09 | 4.6.1、4.6.2、4.6.5 | UT-GRAPH-024、IT-GRAPH-001、IT-GRAPH-008 |
| G-FR-10 | 4.6.5 | IT-GRAPH-001 |
| G-FR-11 | 4.4 | UT-GRAPH-001、UT-GRAPH-002、UT-GRAPH-028 |
| G-FR-12 | 4.8 | UT-GRAPH-015、IT-GRAPH-003、IT-GRAPH-006 |
| G-FR-13 | 4.9 | UT-GRAPH-016、UT-GRAPH-017、IT-GRAPH-007 |
| G-FR-14 | 4.2.3、4.2.11 | UT-GRAPH-010、UT-GRAPH-017、UT-GRAPH-022、UT-GRAPH-025 |
| G-FR-15 | 4.2.3、4.2.11 | UT-GRAPH-001、UT-GRAPH-017、UT-GRAPH-023、UT-GRAPH-025 |
| G-FR-16 | 4.10 | UT-GRAPH-018、UT-GRAPH-026 |
| G-FR-17 | 4.11 | UT-GRAPH-014、IT-GRAPH-005 |
| G-FR-18 | 4.9.3、4.10 | UT-GRAPH-017、IT-GRAPH-008 |
| G-FR-19 | 4.2.11、4.9 | UT-GRAPH-025、IT-GRAPH-001 |
| G-NFR-01 | 4.12 | PT-GRAPH-001、PT-GRAPH-002 |
| G-NFR-02 | 4.12 | PT-GRAPH-003 |
| G-NFR-03 | 4.9.3 | UT-GRAPH-017 |
| G-NFR-04 | 4.1、4.2.1 | UT-GRAPH-027、IT-GRAPH-009、头文件与二进制审计 |
| G-NFR-05 | 6.1 | UT-GRAPH-012、IT-GRAPH-003 |
| G-NFR-06 | 全文 | TC-DOC-01 |
| G-NFR-07 | 4.11 | UT-GRAPH-020、ST-GRAPH-001/002 |
| G-NFR-08 | 4.13 | 并发审计、ST-GRAPH-001 |

## 6.8 未覆盖项与风险

| 项目 | 说明 | 计划 |
| --- | --- | --- |
| 真实 GPU 性能 | 本文性能目标为主机初始值；Adreno 实测在 OCL 后端完成后补充 | 阶段一后续 |
| 量化执行 | INT8/INT4 的 Kernel 与数值验证不在本文范围 | Precision Infer + 量化设计文档 |
| 多模型并发 | RuntimeGraph 只描述单模型；多模型由 Engine 管理多个 RuntimeGraph | Engine 设计文档 |
| 动态图/热更新 | 明确不支持；图在 Finalize 后不可变 | 不在阶段一范围 |
| 平台字节序 | 目标平台为小端；未覆盖大端平台 | 后续如需要再评估 |
| Python 绑定性能 | Python 不在运行时热路径；构图耗时仅作离线指标 | 实测后校准 |
| 模型族差异 | Qwen3-0.6B 仅为首个验证模型；其他模型族需新增 Adapter 与规则集，Graph 结构、校验与 Forward 契约不变 | Converter 模块迭代 |
| LibTorch/PyTorch ABI | 本文锁定“同版本同 ABI”原则，具体版本在 G-T1 选定；跨版本互操作不在阶段一范围 | G-T1 与 CI 冒烟 |
| CUDA 张量转换 | 单元测试以 CPU/非连续张量覆盖转换规则；真实 CUDA 张量在 GPU 工具链可用后补测 | 阶段一后续 |
| 真实权重数值 | 单元测试不依赖权重；集成数值对齐在 OCL Kernel 完成后执行 | TC-EXEC-01 |

---

# 附录 A：OpType 与 Kernel Key 对照

| op_type | kernel_key | Prefill/Decode 变体 | 说明 |
| --- | --- | --- | --- |
| `kEmbedding` | `embedding` | 否 | Token ID → hidden |
| `kRmsNorm` | `rms_norm` | 否 | 归一化 |
| `kQkvProjection` | `qkv_projection` | Kernel 内部分派 | QKV + QK Norm + RoPE |
| `kAttention` | `attention` | Kernel 内部分派 | Attention + O Proj + KV 写入 |
| `kResidualAdd` | `residual_add` | 否 | 逐元素加 |
| `kMlpGateUpAct` | `mlp_gate_up_act` | 否 | gate/up + SiLU + Mul |
| `kMlpDownResidual` | `mlp_down_residual` | 否 | down + Add |
| `kLmHead` | `lm_head` | 否 | 末 token 汇聚 + 投影 |
| `kSampling` | `sampling` | 否 | 采样 |
| `kCast` | `cast` | 否 | 精度转换 |

`kernel_key` 是逻辑键；Backend 可据此选择厂商 Kernel、编译选项或内部变体，但不得改变 Graph 语义。本表为阶段一 Qwen3-0.6B 的映射；其他模型族按同一结构增加 OpType 与 kernel_key 条目。

# 附录 B：整图示例（Qwen3-0.6B）

```text
input.input_ids ──▶ embedding ──▶ hidden

for i in 0..N-1:
  hidden ──▶ layer.i.attn_norm ──▶ layer.i.qkv ──▶ layer.i.attn ──▶ layer.i.attn_residual ──▶ hidden_mid
       │                                  ▲                                   ▲
       │                                  │                                   │
       │                         input.position_ids                           │
       │                                                                      │
       └──────────────────────────────────────────────────────────────────────┘

  hidden_mid ──▶ layer.i.mlp_norm ──▶ layer.i.mlp_gate_up ──▶ layer.i.mlp_down_residual ──▶ hidden
                                                                        ▲
                                                                        │
                                                                   hidden_mid（残差）

hidden ──▶ final_norm ──▶ lm_head ──▶ sampling ──▶ output.token_ids
                            ▲
                            │
                  input.last_token_indices
```

说明：图中 `layer.i.attn_residual` 的输出 `hidden_mid` 同时作为 `layer.i.mlp_down_residual` 的残差输入；这是块内唯一的跨阶段数据依赖，不构成融合。

# 附录 C：参考资料

1. `../MobileNet_推理引擎_设计文档.md`：整体架构、模块职责、需求编号与测试用例。
2. `../Docs/knowledge/backend_plugin_architecture.md`：后端插件架构知识库（Graph 与 Backend 边界参考）。
3. Qwen3 模型源码与 `config.json`：融合规则与模型超参数的最终事实源；转换期由 Converter 校验。
4. Google C++ Style Guide、Google Python Style Guide：编码规范。
