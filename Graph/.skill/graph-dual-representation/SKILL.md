---
name: graph-dual-representation
description: 为推理引擎构建“Graph IR（torch/at::Tensor）+ RuntimeGraph（TensorDesc）”双表示图模块，并用 Canonical JSON 逐字节校验 Python/C++ 双端一致；适用于把模型结构固化为融合算子图、校验图合法性、转换到运行时可执行图。
---

# Graph 双表示构建与双端一致性校验

> **已废弃（2026-11-13）**：本 skill 描述的 torch/LibTorch Graph IR +
> RuntimeGraph/TensorDesc 双表示已被 `Graph/.doc/Graph_design.md` 的
> 单一 C++17 runtime 静态图取代。当前项目请使用
> `ToolChain/.skill/build-stage-pipeline/SKILL.md` 与
> `Graph/.skill/runtime-static-graph-cpp-pybind/SKILL.md`。以下内容仅作历史参考。

## 适用场景

- 需要用一个与模型无关的图（Graph IR）表达融合算子与权重，同时导出不依赖
  LibTorch 的运行时图（RuntimeGraph）。
- Python 侧使用 `torch.Tensor`/`torch.dtype` 构图，C++ 侧使用
  `at::Tensor`/`at::ScalarType` 构图，并要求两侧输出完全一致。
- 需要把图校验（拓扑、端口、dtype/shape、融合规则、I/O 规则）与运行时绑定
  校验（arena、Kernel、binding、执行表）分层的项目。
- 需要为某个模型族（例如 Qwen3）建立第一份参考图与金样快照，后续扩展到
  其他模型族。

## 不适用场景

- 只需要一次性脚本导出 ONNX/Protobuf，不需要运行时图与校验分层。
- 目标环境无法提供 PyTorch/LibTorch，且不需要 Python 构图能力。
- 需要图优化/算子编译（本方法只负责表达、校验与转换，不做调度优化）。

## 输入

1. 模型族结构信息：层数、hidden/intermediate、head 数、KV head 数、
   head_dim、词表、上下文长度、归一化 eps、rope 参数、dtype、权重共享策略。
2. 融合规则骨架：`归一化 → 投影+后处理 → attention → 残差 → 归一化 →
   gate/up+激活 → down+残差`；按模型族在规则表中给出 rule_id、阶段、OpType、
   required_op_kinds 与 kernel_key。
3. 权重元数据：参数名、形状、dtype；允许 meta 张量（不加载真实权重）。
4. 设计文档：必须包含背景、需求、概要设计、详细设计、实施步骤、测试设计
   六部分，并给出可追溯的 UT/IT 编号。

## 输出（四件套）

1. 设计文档（markdown）。
2. 代码：公共类型、Graph IR、RuntimeGraph、Builder/Validator、融合规则表、
   TensorDesc 转换、Python 绑定、构建脚本与测试。
3. 测试报告：环境、命令、用例、实际输出、结论、未覆盖项与风险。
4. 本 skill 文档。

## 核心不变量

- Graph IR 只用 torch 原生元素：张量属性为 `at::Tensor`/`torch.Tensor`，
  dtype 为 `at::ScalarType`/`torch.dtype`；不得自定义底层张量/dtype/device/
  layout 类型。符号维与量化信息作为独立元数据（`SymbolicShape`、
  `QuantizationDesc`）。
- RuntimeGraph 只用 `TensorDesc`/`DataType`/`Layout`/`DeviceType`，头文件不得
  出现 `<ATen/...>`、`<torch/...>`、`<c10/...>`，目标不得链接 LibTorch。
- `ToTensorDesc` 只做元数据转换，不 detach、不拷贝、不连续化、不转置；
  序列化路径的归一化由 Serializer 负责。
- Canonical JSON 是双端一致性的唯一判定：字段顺序、数组顺序、数字格式必须
  确定；任何插入顺序变化都会改变字节输出。
- Graph 是模型无关的；Qwen3 只是首个验证模型族。新增模型族通过规则表与
  构建器扩展，不修改公共图结构。

## 标准流程

### 1. 冻结双表示与公共类型

- 先写设计文档，明确 Graph IR 与 RuntimeGraph 的边界、字段、错误码、
  校验规则编号（V1~V17）与测试编号。
- 公共层只放与 torch 无关的枚举、`Dim`/`SymbolicShape`、
  `QuantizationDesc`、`Status`、`AttributeValue`。
- RuntimeGraph 头文件从第一天起就禁止 torch include；用扫描测试锁定。

### 2. Graph IR 构图

- 先加边（含 weight 边），再加节点，或用 Builder 统一解析名称引用。
- I/O 边必须是具体形状（host tensor 参与逐值比较）；内部激活可用符号维
  `batch`/`sequence`。
- `tie_word_embeddings` 时让 LM Head 与 embedding 共享同一个张量对象，
  用 `data_ptr`/storage 验证。
- Finalize 内完成：边 producer/consumer 解析、Kahn 拓扑排序、V1~V16 校验；
  失败返回 `report.ToString()`。

### 3. 运行时图构造与转换

- 用 `RuntimeGraphBuilder` 以 `TensorDesc` 重建图，或由
  `ConvertGraphToRuntimeGraph` 从 IR 转换。
- 转换后逐项比较 dtype/shape/stride/layout/device/offset/size_bytes 与
  符号维、量化信息。
- Serializer 侧再做 detach/CPU/contiguous/列主序转置，不污染 IR。

### 4. Python 绑定

- 使用 pybind11；正常构建走 torch 官方 caster。
- 无 LibTorch 的测试构建使用桥接头文件：`THPVariable_Unpack` 解包，
  动态解析 `THPVariable_Wrap` 包装；两边共享 TensorImpl，不复制数据。
- 异常在绑定边界统一转换为带 `code/detail/node_name/edge_name` 的
  `GraphError`。

### 5. 双端一致性校验

- Python 构建器与 C++ 夹具的权重、节点、边插入顺序必须完全一致。
- 比较 `to_canonical_json()` 的**字节**，不是解析后的对象；失败时用路径
  差异工具定位首个不同字段。
- 固化金样快照（大小 + SHA256），后续改动必须显式更新快照。

### 6. 测试与交付

- C++：RuntimeGraph 用 Mock Backend/Kernel 覆盖 Forward 顺序、H2D/D2H、
  binding/arena 校验、Kernel 错误透传、Release 幂等与逆拓扑序。
- Python：UNIX 风格 unittest 覆盖 V1~V16 正常/负向路径、ToTensorDesc 规则、
  融合规则表、torch 原生类型断言。
- 集成：Qwen3-0.6B 双端图统计、Canonical JSON 字节相等、RuntimeGraph 转换
  相等、权重共享、金样快照。
- 报告必须写实际命令与输出；未覆盖项（如无 MSVC/LibTorch 的 CMake 构建、
  真实权重推理、性能/模糊测试）要显式列出。

## 检查清单

- [ ] Graph IR 头/源文件中没有自定义张量/dtype/device/layout 类型。
- [ ] RuntimeGraph 头文件无 torch/ATen/c10 include，测试可执行文件无 torch DLL。
- [ ] I/O 边为具体形状；内部符号维名称与示例值一致。
- [ ] 权重边有 `weight_ref`，非权重边无 `weight_ref`。
- [ ] `ToTensorDesc` 对未定义张量、非连续张量、列主序、符号维均有明确行为。
- [ ] Canonical JSON 重复调用稳定，Python/C++ 字节相等。
- [ ] 融合规则 rule_id 全局唯一，未知模型族返回空。
- [ ] Forward 失败时错误码与节点名透传；Release 幂等且逆拓扑序。
- [ ] 所有新增 C++ 代码通过 Google style（80 列、命名、include 顺序）。
- [ ] 所有模块/类/函数有中文注释，覆盖功能边界、输入输出、主要逻辑。

## 常见陷阱

1. **meta 张量不能调用 `storage()`**：meta/fake 张量的 TensorImpl 设置了
   `storage_access_should_throw_`，`tensor.storage()` 会抛 c10 异常；异常跨
   pybind11/MSVC-libc++ 边界时可能变成 `std::terminate`，表现为
   “Unhandled exception caught in c10/util/AbortHandler.h”。应使用
   `tensor.unsafeGetTensorImpl()->unsafe_storage()` 并检查数据指针。
2. **把优化级别当 ABI 问题**：上述崩溃在 `-O0` 下可能偶然不触发，容易误判为
   `-O2` ABI 问题；应先用符号化堆栈定位到具体调用点。
3. **I/O 边用符号维**：host tensor 比较需要具体值，I/O 边必须是常量形状。
4. **插入顺序漂移**：权重/节点/边顺序变化会改变 Canonical JSON；Python 与
   C++ 构建器必须同步维护。
5. **`RuntimeNode` 的边字段是 `vector<EdgeId>`**：运行时图按索引引用，测试
   构造时要先加边拿到 id，不能写边名。
6. **`has_storage()` 不等于可安全访问存储**：meta 张量也可能返回 true；必须
   同时判断 `is_meta()` 或走 `unsafe_storage()`。
7. **桥接模式符号签名**：`THPVariable_Wrap` 的导出名与参数（`const
   at::TensorBase&`）必须与 torch 头文件一致，否则运行期解析失败或栈损坏。
8. **CMake 目标隔离**：不要把 `graph_ir` 链进 `graph_runtime`；否则会把
   LibTorch 带进移动端运行时。

## 历史实现状态

双表示图模块（`at::Tensor` Graph IR + `TensorDesc` RuntimeGraph）、
对应构建脚本与测试已从仓库删除，仅保留本文作为迁移记录。当前实现见：

- 设计：`Graph/.doc/Graph_design.md`
- 代码：`Graph/include/`、`Graph/src/`
- Python 绑定：`Graph/python/graph_python.cc`
- 测试：`Graph/tests/graph_test.cc`、`Graph/tests/test_graph.py`
- 当前 skill：`Graph/.skill/runtime-static-graph-cpp-pybind/SKILL.md`
