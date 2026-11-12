# Graph 模块实施与测试设计

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 日期 | 2026-11-12 |
| 基础设计 | [Graph设计文档.md](Graph设计文档.md) |
| 代码目录 | `Graph_Fixed/` |
| 验证目录 | `Build/graph_fixed/` |
| 测试报告 | [TestResults.md](TestResults.md) |
| 配套 skill | `Skills/runtime-static-graph-cpp-pybind/SKILL.md` |

本文档记录 `Graph_Fixed` 从基础设计到 C++ 实现、Python 绑定、构建测试和
交付验证的落地设计。基础设计文档描述 IR 的抽象结构；本文档描述当前代码
实际提供的接口、约束、错误路径和验证方式，并显式列出与基础设计的差异。

## 1. 背景与目标

### 1.1 背景

推理引擎在 Python 工具链端完成：

```text
模型源码 -> Python 端静态推理图 -> 量化 / 精度推导 -> Shape 推导
        -> 显存管理 -> 序列化
```

Runtime 端在端侧 Device 上加载并执行静态图。`Graph_Fixed` 是 Runtime 侧的
静态图 IR，负责回答“图里有哪些 op、tensor、常量和输入输出，它们如何连接，
按什么顺序执行”，不负责算子实现、显存分配、精度/Shape 推导或序列化。

仓库中已有 `Graph/` 工具链图模块，它依赖 torch/ATen 并以 `at::Tensor` 表达
图 IR。`Graph_Fixed` 与其定位不同：

- `Graph/` 服务于 Python 工具链构图和双端一致性校验；
- `Graph_Fixed/` 服务于 Runtime 端静态图，头文件和目标中不得出现
  torch/ATen/c10，也不得依赖 GPU/OCL 算子库；
- 两者可以并存，但共享的只有 `Runtime/common/status.h` 的错误码与
  `Status` 类型。

基础设计文档 `Graph_Fixed/Graph设计文档.md` 已定义 `Graph`、`Op`、
`Tensor`、`Storage`、`Shape`、`GraphEnv`、`OpEnv` 和 `OpType`。本任务要求
按该文档生成 C++ 代码，并把图构建接口通过 pybind11 暴露给 Python。

### 1.2 目标

| 编号 | 目标 |
|------|------|
| G-01 | 实现不依赖 torch/ATen/c10、不依赖 GPU/OCL 的 C++17 Runtime Graph IR。 |
| G-02 | 实现 `Graph` 的构图、常量/输入/输出注册、post DFS 拓扑排序和查询接口。 |
| G-03 | 实现 `GraphEnv` 共享 Storage 元信息和 `OpEnv` 执行环境接口。 |
| G-04 | 通过 pybind11 暴露 Graph/Tensor/枚举和构图接口；不暴露 Op、Storage、Shape 对象。 |
| G-05 | 提供可重复的 CMake 构建、C++ 单测、Python 单测和 CTest 集成。 |
| G-06 | 按项目规则交付设计文档、代码、真实测试结果和可复用 skill。 |

### 1.3 非目标

| 编号 | 非目标 |
|------|--------|
| NG-01 | 不实现真实 GPU/OCL kernel；`MegaAttentionOpEnv::Forward()` 只返回 `UNSUPPORTED_OP`。 |
| NG-02 | 不实现显存申请、workspace 规划、权重 H2D 拷贝和反初始化策略。 |
| NG-03 | 不实现 shape、precision、layout 的推导算法。 |
| NG-04 | 不实现模型加载、序列化/反序列化和版本兼容。 |
| NG-05 | 不实现图优化、算子融合、并发执行或流水线调度。 |
| NG-06 | 不把 Python 侧做成完整的模型构图前端；当前只暴露原生 IR 操作。 |

### 1.4 交付范围

| 交付物 | 位置 | 状态 |
|--------|------|------|
| 设计文档 | `Graph_Fixed/Graph设计文档.md`、本文件 | 已交付 |
| C++ 代码 | `Graph_Fixed/include/runtime_graph/`、`Graph_Fixed/src/` | 已交付 |
| Python 绑定 | `Graph_Fixed/python/graph_fixed_python.cc`、`Graph_Fixed/python/mobilenet_runtime_graph/` | 已交付 |
| 测试 | `Graph_Fixed/tests/graph_fixed_test.cc`、`Graph_Fixed/tests/test_graph_fixed.py` | 已交付并通过 |
| 测试结果 | `Graph_Fixed/TestResults.md` | 已交付 |
| skill | `Skills/runtime-static-graph-cpp-pybind/SKILL.md` | 已交付 |

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收方式 |
|------|------|----------|
| FR-01 | `Graph` 以名称为标识，独占持有 op、普通 tensor、constant tensor、GraphEnv。 | UT-C02、UT-C03、UT-P01 |
| FR-02 | `CreateOp` 创建普通计算 op 和 `out_cnt` 个输出 tensor，维护双向连接，拒绝空名称、重复名称、非法 op 类型、空输出参数、0 输出、外部 tensor。 | UT-C02、UT-C04、UT-P02、UT-P09 |
| FR-03 | `CreateConstant` 创建无 producer op 的 constant tensor，保存 Storage，强制 `reusable=false`，并加入常量列表。 | UT-C02、UT-P03 |
| FR-04 | `CreateInput` 支持 legacy 和显式 precision/layout 两种重载；输入 op 与输入 tensor 同名，登记到输入列表，`attr["storage"]` 保存声明。 | UT-C02、UT-P04、UT-P06 |
| FR-05 | `CreateOutput` 创建只消费已有 tensor、不产生新 tensor 的输出 op，登记到输出列表。 | UT-C02、UT-P02 |
| FR-06 | `SortOp` 从输出 op 出发做 post DFS，生成拓扑序 `op_list_`，constant 作为依赖叶子，不进入执行序列；重复调用结果稳定。 | UT-C02、UT-P02、UT-P08 |
| FR-07 | 提供常量、输入 op、输出 op、执行序列的只读查询。 | UT-C02、UT-C03、UT-P02 |
| FR-08 | `GraphEnv` 以名称为键注册和查询共享 Storage，拒绝空名称和重复名称。 | UT-C03 |
| FR-09 | `OpEnv` 提供 `Init`、`Forward`、`NeedTempData`；`MegaAttentionOpEnv` 作为派生示例。 | UT-C03 |
| FR-10 | Python 绑定暴露 Graph、Tensor、Precision、Layout、DeviceType、OpType 和构图/排序/dump 接口；不暴露 Op、Storage、Shape。 | UT-P01 至 UT-P10 |
| FR-11 | `dump_ir` 输出确定性的 inputs/constants/ops/outputs 文本，不输出真实指针。 | UT-P01 至 UT-P08 |
| FR-12 | `mobilenet_runtime_graph` 公共包可从源码树或 `GRAPH_FIXED_PYTHON_DIR` 加载扩展。 | UT-P10 |
| FR-13 | CMake 构建静态库、Python 扩展、C++ 单测并注册 CTest。 | TR-01、TR-02 |

### 2.2 非功能需求

| 编号 | 需求 | 落地方式 |
|------|------|----------|
| NFR-01 | Runtime Graph 不依赖 torch/ATen/c10 和 GPU/OCL 库。 | 源文件与头文件不 include 相关头文件；CMake 只链接 `Runtime/common/status.cc`。 |
| NFR-02 | C++ 代码遵守 Google Style，新增行不超过 80 显示列。 | 命名、缩进、include 顺序自检；脚本按东亚字符宽度检查。 |
| NFR-03 | 所有模块、类、函数有中文注释，覆盖功能边界、输入输出、主要逻辑。 | 逐文件检查，文件头、类、函数均有注释。 |
| NFR-04 | 错误通过 `Status` 返回，不依赖异常表达可预期错误；Python 边界统一转异常。 | C++ 返回 `INVALID_ARGUMENT`/`STATE_ERROR`；pybind 转 `RuntimeError`/`ValueError`。 |
| NFR-05 | 图对象所有权清晰，禁止悬空引用。 | `unique_ptr` 持有，列表保存非拥有裸指针；Tensor 绑定使用 `reference` 策略并文档化生命周期。 |
| NFR-06 | 输出确定性，便于逐行断言和回归。 | `std::map` 保证属性顺序；列表保留注册顺序；`StorageToString` 不打印指针值。 |
| NFR-07 | 构图和排序在单线程完成；不承诺线程安全。 | 类注释明确说明；不加锁，避免隐藏同步成本。 |
| NFR-08 | 构建可在无 MSVC 的 Windows 环境完成。 | CMake + Ninja + Zig C++ 包装脚本；归档器可配置。 |

### 2.3 约束条件

| 编号 | 约束 |
|------|------|
| C-01 | C++17，禁止 RTTI 之外的非标准扩展；当前使用 `std::variant`、`std::map`、`std::unique_ptr`。 |
| C-02 | 复用 `mobilenet::Status` 和 `mobilenet::ErrorCode`，不重复定义错误模型。 |
| C-03 | 命名空间为 `mobilenet::runtime_graph`，避免与 `Graph/` 工具链模块冲突。 |
| C-04 | 构图期权重地址由调用方提供，Graph 不复制权重数据。 |
| C-05 | Python 侧最低版本 3.10；本地验证使用 Python 3.13.3。 |
| C-06 | `kMaxShapeDim=8`，超过 8 维的 shape 在 Python 入口拒绝。 |

### 2.4 验收标准

| 编号 | 验收标准 |
|------|----------|
| AC-01 | 干净构建 16/16 目标成功，无编译警告。 |
| AC-02 | C++ 单测返回 0，输出 `graph_fixed_test: ALL TESTS PASSED`。 |
| AC-03 | Python 单测 10/10 通过，覆盖正常构图、排序、dump 和错误路径。 |
| AC-04 | CTest 2/2 通过，退出码为 0。 |
| AC-05 | `_graph_fixed` 扩展可在 Python 中导入，公共包可加载并导出相同类型。 |
| AC-06 | 设计文档、代码、测试结果、skill 四件套齐全且相互可追溯。 |

## 3. 概要设计

### 3.1 分层结构

```text
+-------------------------------------------------------------+
| Python 模型加载 / 测试脚本                                   |
|   mobilenet_runtime_graph（公共包，转发到 _graph_fixed）      |
+------------------------------+------------------------------+
                               | pybind11
+------------------------------v------------------------------+
| _graph_fixed 扩展                                            |
|   Graph / Tensor / 枚举绑定；Status -> Python 异常            |
+------------------------------+------------------------------+
                               | C++17 API
+------------------------------v------------------------------+
| runtime_graph 静态库                                         |
|   Graph -> Op -> OpEnv                                      |
|   Graph -> Tensor -> Storage -> Shape                        |
|   Graph -> GraphEnv -> Storage                               |
+------------------------------+------------------------------+
                               | 仅 Status
+------------------------------v------------------------------+
| Runtime/common/status.h                                      |
+-------------------------------------------------------------+
```

依赖方向严格向下：上层可以依赖下层，下层不能反向依赖 pybind、torch 或
GPU/OCL。

### 3.2 目录与职责

| 路径 | 职责 |
|------|------|
| `include/runtime_graph/types.h` | `Precision`、`Layout`、`DeviceType`、`kMaxShapeDim`、名称/字节数查询。 |
| `include/runtime_graph/shape.h` | 定长 `dims`/`pitch`、逻辑元素数和物理元素数计算。 |
| `include/runtime_graph/storage.h` | Shape、Precision、Layout、DeviceType、offset、byte_size、data_ptr、reusable。 |
| `include/runtime_graph/attr.h` | `AttrValue` variant、`AttrMap`、稳定文本格式化。 |
| `include/runtime_graph/tensor.h` | 数据边、producer/consumer、内嵌 Storage、constant 标记。 |
| `include/runtime_graph/op.h` | 节点、属性、输入输出连接、OpEnv 生命周期。 |
| `include/runtime_graph/op_type.h` | 全部算子枚举和 `OpTypeToStr`。 |
| `include/runtime_graph/op_env.h` | 执行环境抽象和 `MegaAttentionOpEnv` 示例。 |
| `include/runtime_graph/graph_env.h` | graph 级共享 Storage 表。 |
| `include/runtime_graph/graph.h` | 图根对象、构图、图边界、排序和查询。 |
| `src/*.cc` | 与头文件一一对应的实现。 |
| `python/graph_fixed_python.cc` | pybind11 绑定和属性类型转换。 |
| `python/mobilenet_runtime_graph/__init__.py` | 公共包加载器，导出原生类型和枚举。 |
| `tests/graph_fixed_test.cc` | C++ 单元测试。 |
| `tests/test_graph_fixed.py` | Python 单元测试。 |
| `tools/` | CMake + Zig 本地构建包装脚本。 |
| `CMakeLists.txt` | 静态库、扩展、测试和 CTest 注册。 |

### 3.3 所有权关系

```text
Graph
 ├── op_map_              : map<string, unique_ptr<Op>>       （拥有）
 ├── tensor_map_          : map<string, unique_ptr<Tensor>>   （拥有）
 ├── constant_tensor_map_ : map<string, unique_ptr<Tensor>>   （拥有）
 ├── graph_env_           : unique_ptr<GraphEnv>              （拥有）
 ├── op_list_             : vector<Op*>                       （借用）
 ├── constant_tensor_list_: vector<Tensor*>                   （借用）
 ├── input_op_list_       : vector<Op*>                       （借用）
 └── output_op_list_      : vector<Op*>                       （借用）

Op
 ├── src_tensors_ : vector<Tensor*>   （借用，Graph 保证生命周期）
 ├── dst_tensors_ : vector<Tensor*>   （借用，Graph 保证生命周期）
 └── op_env_      : unique_ptr<OpEnv> （拥有）

Tensor
 ├── src_ops_ : vector<Op*>   （借用）
 ├── dst_ops_ : vector<Op*>   （借用）
 └── storage_: Storage        （值语义，不拥有 data_ptr）

OpEnv
 ├── op_        : Op*         （借用）
 └── graph_env_ : GraphEnv*   （借用，可为空）
```

关键不变量：

1. `Tensor` 只由 `tensor_map_` 或 `constant_tensor_map_` 之一持有，不会同时
   存在于两者中。
2. constant tensor 没有 producer op，`is_constant_=true`，`storage_.reusable`
   被强制为 `false`。
3. 普通 tensor 至少有一个 producer op；输入 tensor 的 producer 是 input op。
4. 执行序列只包含从输出可达的 op，constant 不产生执行节点。
5. `Graph` 析构函数先对所有 op 调用 `ClearOpEnv()`，再让成员逆序析构，确保
   派生 `OpEnv` 的析构逻辑不会访问已经释放的 `GraphEnv`；`OpEnv` 本身不
   拥有 `Op` 或 `GraphEnv`。

### 3.4 关键流程

#### 3.4.1 构图流程

```text
CreateConstant -> 校验名称/Shape -> 建 Tensor -> 写 Storage
               -> reusable=false -> constant 列表/索引
CreateInput    -> 校验 op/tensor 名称 -> 建 input op + tensor
               -> attr["storage"] -> input 列表/索引
CreateOp       -> 校验名称/输出参数/op 类型/源 tensor
               -> 预生成并校验全部输出 tensor 名
               -> 建 op -> 连接 src/dst -> 注册索引 -> 返回输出
CreateOutput   -> 校验 op 名称/输入 tensor 归属
               -> 建 kOutput op -> 连接 input -> output 列表/索引
```

所有 `Create*` 都先完成校验再提交，避免失败时留下半成品节点。

#### 3.4.2 排序流程

```text
SortOp:
  clear op_list_
  if output_op_list_ empty -> STATE_ERROR
  states = {}
  for output_op in output_op_list_:
      VisitOp(output_op, states)
  return OK

VisitOp(op, states):
  state = states[op]            // 0 未访问，1 访问中，2 已完成
  if state == 1 -> 依赖环
  if state == 2 -> 返回
  state = 1
  for tensor in op.SrcTensors:
      if tensor == null -> STATE_ERROR
      if tensor.IsConstant -> continue
      if tensor.SrcOps empty -> STATE_ERROR
      for producer in tensor.SrcOps:
          VisitOp(producer, states)
  state = 2
  op_list_.push_back(op)
```

结果是 post-order：producer 先于 consumer，input op 位于最前，output op 位于
最后。

#### 3.4.3 Python 调用流程

```text
Python 调用 graph.create_op(...)
  -> pybind 解析 list[Tensor*] 和 attr dict
  -> ToAttrMap 把 bool/int/float/str/list/tuple 转成 AttrValue
  -> Graph::CreateOp
  -> 失败：ThrowIfError 抛 std::runtime_error -> Python RuntimeError
  -> 成功：按 reference 策略返回 Tensor 句柄列表
```

### 3.5 技术选型与取舍

| 主题 | 选择 | 理由 | 代价 |
|------|------|------|------|
| 属性容器 | `std::variant` + `std::map` | 支持有限类型集合，顺序稳定，无第三方依赖 | 每增加类型需改 `AttrValue` 和格式化分支 |
| 对象所有权 | `unique_ptr` + 非拥有裸指针 | 所有权唯一，执行序列重建成本低 | 裸指针生命周期需文档和测试约束 |
| 拓扑排序 | 从输出出发的递归 post DFS | 只访问可达子图，天然跳过常量 | 递归深度受图深度限制，超大图需改迭代 |
| Python 绑定 | pybind11 | 与 C++ 类型映射直接，异常转换简单 | 需显式管理 Tensor 非拥有句柄 |
| Python 公共包 | 薄加载器 | 兼容源码树和构建目录，不复制图逻辑 | 安装到 site-packages 时默认路径失效 |
| 构建 | CMake + Ninja | 跨平台，易注册 CTest | 无 MSVC 时需 Zig 包装和归档器覆盖 |
| 错误模型 | 复用 `Status` | 与 Runtime 其他模块一致 | Python 只保留消息，错误码未单独暴露 |

### 3.6 与基础设计文档的落地差异

| 编号 | 差异 | 原因 | 影响 |
|------|------|------|------|
| D-01 | `OpType` 增加 `kLinearRmsNorm`、`kLinearRmsNormRope`、`kLinearRmsNormRopeQuantPagedKVStore`、`kLinearRopeRotateQuant`、`kLinearMul`、`kLinearApe`、`kRmsNormRopeQuantPagedKVStore`、`kRopeQuantLinear`。 | 基础设计文档的 OpType 表中引用了这些值，但枚举片段遗漏；已同步补入基础设计文档和代码。 | Python 枚举、`OpTypeToStr` 与文档表一致。 |
| D-02 | 新增 C++ 重载 `CreateInput(name, precision, layout, output)`。 | 基础设计文档的接口表描述了显式 precision/layout 声明，但 C++ 片段只有 legacy 版本。 | Python `create_input(name, precision, layout)` 可直接使用。 |
| D-03 | 输出 tensor 固定命名为 `<op>.out`（单输出）或 `<op>.out.<index>`（多输出）。 | 基础设计文档未规定命名；固定规则保证 dump 和测试可重复。 | 外部加载器需遵循该命名，或改为显式传入名称。 |
| D-04 | `GraphEnv` 当前只提供共享 Storage 的注册/查询；`OpEnv` 只提供接口与 MegaAttention 占位。 | 本次范围是 IR 与绑定，不含 Engine、Manager 和真实 kernel。 | Engine 接入前需扩展初始化、workspace 回填和具体 OpEnv。 |
| D-05 | Python 不暴露 `Op`、`Storage`、`Shape` 对象；`create_constant` 用散装参数组装 Storage。 | 保持绑定面最小，避免 Python 侧依赖 C++ 内部布局。 | 无法从 Python 直接读取/修改 Storage；调试依赖 `dump_ir`。 |
| D-06 | 未实现基础设计文档提到的 `Tensor(..., dtype=..., is_input=True)` / `mark_input(name, dtype)` 对象式包装。 | 原生 pybind 已提供等价的 `create_input(name, precision, layout)`；对象式构造需要额外的图归属和生命周期设计。 | 当前 Python 侧使用 `graph.create_input(...)`；后续可在薄包装层补充，不改变 C++ IR。 |
| D-07 | `MegaAttentionOpEnv::Forward()` 返回 `UNSUPPORTED_OP`，`NeedTempData()` 返回 `true`。 | 真实 MegaAttention kernel 不在本模块范围。 | 只能验证接口和生命周期，不能验证 attention 数值结果。 |
| D-08 | 新增 `mobilenet_runtime_graph` Python 公共包。 | 统一导出原生类型，并支持源码树/环境变量两种加载路径。 | 属于便利层，不改变 `_graph_fixed` 的行为。 |
| D-09 | `Graph` 析构函数不再使用 `= default`，而是先对所有 op 调用 `ClearOpEnv()`。 | `graph_env_` 是最后一个成员，默认析构会先释放 GraphEnv，再析构 OpEnv；派生 OpEnv 若在析构中访问 GraphEnv 会悬空。 | 析构顺序安全；单测新增 `CountingOpEnv` 覆盖析构清理。 |

## 4. 详细设计

### 4.1 公共类型

#### 4.1.1 `Precision` / `Layout` / `DeviceType`

- `Precision`：`kUnknown`、`kFloat64`、`kFloat32`、`kFloat16`、`kBFloat16`、
  `kFloat8E4M3`、`kFloat8E5M2`、`kInt64`、`kInt32`、`kInt16`、`kInt8`、
  `kUInt8`、`kBool`。
- `Layout`：`kRowMajor`、`kColumnMajor`。
- `DeviceType`：`kUnknown`、`kHost`、`kDevice`。
- `PrecisionByteSize` 返回单元素字节数；未知类型返回 0。
- `PrecisionName` / `LayoutName` / `DeviceTypeName` 返回稳定的小写字符串，
  供 `StorageToString` 和测试使用。

#### 4.1.2 `Shape`

- `dims`、`pitch` 为 `std::array<int64_t, kMaxShapeDim>`，默认全部为 1。
- `max_dim` 默认 0，表示空 shape，此时元素数为 1。
- `GetShapeSize()` 返回 `dims[0, max_dim)` 的乘积；`GetByteSize()` 返回
  `pitch[0, max_dim)` 的乘积。
- 越界、负维度、`int64_t` 乘法溢出返回 `-1`。
- `GetByteSize()` 的名称沿用基础设计，实际语义是“物理 pitch 元素数”，
  不是字节数；真实字节数由 `Storage::byte_size` 表达。

#### 4.1.3 `Storage`

| 字段 | 类型 | 默认值 | 语义 |
|------|------|--------|------|
| `shape` | `Shape` | 空 shape | 逻辑维度和物理 pitch。 |
| `precision` | `Precision` | `kUnknown` | 元素精度。 |
| `layout` | `Layout` | `kRowMajor` | 逻辑布局。 |
| `device_type` | `DeviceType` | `kUnknown` | `data_ptr` 当前所在设备类别。 |
| `offset` | `size_t` | 0 | workspace 内逻辑偏移，当前约定为字节。 |
| `byte_size` | `size_t` | 0 | 当前 storage 字节数。 |
| `data_ptr` | `void*` | `nullptr` | 非拥有数据地址。 |
| `reusable` | `bool` | `true` | 是否允许参与 workspace 复用。 |

`StorageToString` 输出固定字段顺序，不输出真实地址，只输出
`data=set`/`data=unset`，便于逐字符串断言。

#### 4.1.4 `AttrValue` / `AttrMap`

`AttrValue` 支持：

```cpp
bool
int64_t
double
std::string
std::vector<int64_t>
std::vector<double>
std::vector<std::string>
Storage
```

`AttrMap` 为 `std::map<std::string, AttrValue>`，因此 `dump_ir` 中属性按
key 字典序输出。数组格式为 `[a,b]`，布尔为 `true`/`false`。

### 4.2 `Graph` 接口与行为

#### 4.2.1 `CreateOp`

```cpp
Status CreateOp(const std::string& name,
                const std::vector<Tensor*>& src_tensors,
                size_t out_cnt, OpType op_type,
                const AttrMap& attr,
                std::vector<Tensor*>* outputs);
```

校验顺序：

1. op 名称非空且未被 `op_map_` 占用；
2. `outputs != nullptr`；
3. `out_cnt > 0`；
4. `op_type` 不是 `kUnknown`、`kInput`、`kOutput`；
5. 每个源 tensor 非空，且属于当前 graph 的 `tensor_map_` 或
   `constant_tensor_map_`；
6. 预生成全部输出 tensor 名称，并确认未被
   `tensor_map_`/`constant_tensor_map_` 占用。

只有全部校验通过后才创建 op 和 tensor。连接关系为：

- `op.src_tensors_ += src_tensors`；
- 每个源 tensor 的 `dst_ops_ += op`；
- 每个输出 tensor 的 `src_ops_ += op`；
- `op.dst_tensors_ += output`。

#### 4.2.2 `CreateConstant`

```cpp
Status CreateConstant(const std::string& name,
                      const Storage& storage, Tensor** output);
```

- 名称必须在 tensor 命名空间中唯一；
- `output` 非空；
- `storage.shape.GetShapeSize() >= 0`；
- 复制 Storage，强制 `reusable=false`；
- 不创建 producer op，不写入 `tensor_map_`；
- 追加到 `constant_tensor_list_` 和 `constant_tensor_map_`。

`byte_size`、`data_ptr`、`precision` 与 shape 的一致性由调用方保证；当前
实现不交叉校验，这是有意的信任边界。

#### 4.2.3 `CreateInput`

legacy：

```cpp
Status CreateInput(const std::string& name, Tensor** output);
```

显式重载：

```cpp
Status CreateInput(const std::string& name, Precision precision,
                   Layout layout, Tensor** output);
```

公共实现 `CreateInputInternal`：

1. 校验 op 名称和 tensor 名称（同名、均在各自命名空间唯一）；
2. 校验 `output` 非空；
3. 创建 `kInput` op 和同名 tensor；
4. `op.dst_tensors_ += tensor`，`tensor.src_ops_ += op`；
5. `attr["storage"]` 写入默认 `Storage{}` 或显式 precision/layout；
6. 登记 `input_op_list_`、`op_map_`、`tensor_map_`。

输入 tensor 的 Storage 自身仍保持默认；构图期声明的 precision/layout 只在
input op 的 attr 中，供 Finalizer 后续同步。

#### 4.2.4 `CreateOutput`

```cpp
Status CreateOutput(const std::string& name, Tensor* input);
```

- 校验输出 op 名称非空且未占用；
- 校验 `input` 属于当前 graph；
- 创建 `kOutput` op，写入默认 `attr["storage"] = Storage{}`；
- `op.src_tensors_ += input`，`input.dst_ops_ += op`；
- 不创建输出 tensor，不写 `tensor_map_`；
- 登记 `output_op_list_` 和 `op_map_`。

#### 4.2.5 `SortOp`

- 每次调用先清空 `op_list_`，支持重复排序；
- 没有输出 op 返回 `STATE_ERROR`；
- 对每个输出 op 执行 `VisitOp`；
- `VisitOp` 使用 `std::map<const Op*, int>` 记录 0/1/2 状态；
- 遇到常量跳过 producer；
- 遇到非 const 且无 producer 的 tensor 返回 `STATE_ERROR`；
- 遇到 null tensor/op 返回 `STATE_ERROR`；
- 发现状态 1 返回依赖环错误；
- 状态 2 直接返回，避免重复加入；
- 访问完成后压入 `op_list_`。

由于公开构图接口只能“消费已有 tensor、产生新 tensor”，正常情况下无法
构造出环；环检测是防御性分支，供未来反序列化或内部构建路径使用。

#### 4.2.6 查询与 GraphEnv

- `GetAllConstantTensor()` 返回 `constant_tensor_list_`；
- `GetAllInputOp()` 返回 `input_op_list_`；
- `GetAllOutputOp()` 返回 `output_op_list_`；
- `GetOrderlyOPList()` 返回排序后的 `op_list_`；
- `SetGraphEnv(nullptr)` 返回 `INVALID_ARGUMENT`，成功时接管所有权；
- `GetGraphEnv()` 未设置返回 `nullptr`；
- `ClearGraphEnv()` 释放 GraphEnv；
- `SetGraphEnv` 只应在 Engine 初始化阶段调用一次；如果在既有 OpEnv 仍持有
  `graph_env_` 时替换或清理 GraphEnv，会留下悬空指针，调用方必须先
  `ClearOpEnv`。

### 4.3 `Op` / `Tensor` / `OpEnv`

#### 4.3.1 `Op`

- 构造函数保存名称、类型、属性副本；
- `SetAttr` 拒绝空属性名，同名覆盖；
- `SetOpEnv(nullptr)` 返回 `INVALID_ARGUMENT`，成功时替换并销毁旧 OpEnv；
  调用方不得继续持有旧 OpEnv 指针，Engine 初始化阶段应只绑定一次；
- `ClearOpEnv()` 释放执行环境；
- `AddSrcTensor` / `AddDstTensor` 为私有，只允许 `Graph` 调用；空指针被忽略，
  但 `Graph` 在调用前已经拒绝空源 tensor。

#### 4.3.2 `Tensor`

- `SrcOps()` / `DstOps()` 返回只读连接列表；
- `GetStorage()` 提供可写和只读重载；
- `Reusable()` / `SetReusable()` 读写 `storage_.reusable`；
- `IsConstant()` 返回 `is_constant_`；
- `AddSrcOp` / `AddDstOp` / `SetConstant` 为私有，只允许 `Graph` 调用；
- `ref_count_` 当前保留为生命周期分析字段，尚未在 `SortOp` 中使用。

#### 4.3.3 `OpEnv`

- `OpEnv(Op*)` 保存所属 op；
- `Init(GraphEnv*)` 默认只保存指针并返回 OK，允许 `graph_env=nullptr`；
- `Forward()` 纯虚，由派生类实现；
- `NeedTempData()` 纯虚，表示是否需要临时 workspace；
- `MegaAttentionOpEnv::Forward()` 返回 `UNSUPPORTED_OP`；
- `MegaAttentionOpEnv::NeedTempData()` 返回 `true`。

### 4.4 Python 绑定

#### 4.4.1 暴露面

| Python 符号 | 类型 | 说明 |
|-------------|------|------|
| `Graph` | class | 构造参数 `name`；提供 `create_op`、`create_constant`、`create_input`、`create_output`、`sort_op`、`dump_ir`。 |
| `Tensor` | class | 只读 `name`、`is_constant`、`reusable`；方法 `set_reusable`。 |
| `Precision` | enum | 全部精度值。 |
| `Layout` | enum | 行主序/列主序。 |
| `DeviceType` | enum | unknown/host/device。 |
| `OpType` | enum | 与 C++ 枚举一一对应。 |

不暴露 `Op`、`Storage`、`Shape`、`GraphEnv`、`OpEnv`。

#### 4.4.2 属性转换

| Python 值 | C++ `AttrValue` |
|-----------|-----------------|
| `bool` | `bool` |
| `int` | `int64_t` |
| `float` | `double` |
| `str` | `std::string` |
| `list[int]` / `tuple[int]` | `std::vector<int64_t>` |
| `list[float]` / `tuple[float]` | `std::vector<double>` |
| `list[str]` / `tuple[str]` | `std::vector<std::string>` |
| `[]` | 空 `std::vector<int64_t>` |
| `None`（attr 整体） | 空 `AttrMap` |

拒绝：其他标量类型、数字与字符串混合数组、非字符串 key、非 dict 的 attr。

#### 4.4.3 错误映射

- `Graph::Create*`、`SortOp` 返回的 `Status` 失败：
  `ThrowIfError` 抛出 `std::runtime_error`，Python 侧捕获为 `RuntimeError`；
- 参数/属性类型转换失败：抛出 `std::invalid_argument`，Python 侧为
  `ValueError`；
- Python 当前不区分 `ErrorCode`，只保留错误消息。后续如需程序化处理，可增加
  异常类型或错误码属性。

#### 4.4.4 生命周期

- `Tensor` 以 `py::return_value_policy::reference` 返回，不转移所有权；
- Python 必须保证 `Graph` 比所有 `Tensor` 句柄存活更久；
- `create_constant` 只保存 `data_ptr`，不复制数据；调用方必须保证缓冲区在
  Graph 使用该权重期间有效；
- 模块 docstring 和公共包 docstring 均写明该约束。

#### 4.4.5 `dump_ir` 文本格式

```text
graph <name>
inputs <count>
  <input_name> -> [<tensor_name>,...]
constants <count>
  <constant_name> <storage(...)>
ops <count>
  [<index>] <op_name> <op_type> src=[...] dst=[...] attrs={k=v,...}
outputs <count>
  <output_name> <- [<tensor_name>,...]
```

- 段顺序固定；
- 属性按 key 字典序；
- 不输出真实指针；
- 未调用 `sort_op()` 时 `ops 0`，但 inputs/constants/outputs 仍可输出。

### 4.5 错误处理

| 场景 | C++ 返回 | Python 结果 |
|------|----------|-------------|
| op/常量/输入/输出名称为空或重复 | `INVALID_ARGUMENT` | `RuntimeError` |
| `CreateOp` 的 `outputs` 为空、`out_cnt=0`、op 类型为 input/output/unknown | `INVALID_ARGUMENT` | `RuntimeError` |
| 源 tensor 为空或不属于当前 graph | `INVALID_ARGUMENT` | `RuntimeError` |
| constant 的 `output` 为空或 shape 非法 | `INVALID_ARGUMENT` | `RuntimeError` |
| input/output 的输出参数为空或 tensor 不属于当前 graph | `INVALID_ARGUMENT` | `RuntimeError` |
| `SortOp` 时无输出、依赖环、非 const tensor 无 producer、空指针 | `STATE_ERROR` | `RuntimeError` |
| `SetGraphEnv(nullptr)` / `SetOpEnv(nullptr)` | `INVALID_ARGUMENT` | 不暴露 |
| Python 维度超过 8、pitch 长度不匹配、维度为负 | 绑定层 `invalid_argument` | `ValueError` |
| Python attr 类型不支持或数组类型混合 | 绑定层 `invalid_argument` | `ValueError` |
| `MegaAttentionOpEnv::Forward()` | `UNSUPPORTED_OP` | 不暴露 |

### 4.6 性能、内存与线程约束

- `SortOp` 时间复杂度 O(V+E)，每个 op 最多访问一次；
- `CreateOp` / `CreateOutput` 的 `OwnsTensor` 当前线性扫描两张 map，最坏
  O(N)；大图（数万 tensor）构图可能成为瓶颈，后续可改为指针集合索引；
- `op_map_` / `tensor_map_` 使用 `std::map`，查找 O(log N)，同时保证稳定
  遍历顺序；
- Graph 不复制权重数据，constant 只保存元信息和外部指针；
- 执行序列、输入输出列表、常量列表均为非拥有指针，重建排序不会复制对象；
- 不使用锁，构图和排序要求单线程；Forward 阶段是否并发由上层 Engine 决定。

### 4.7 构建与测试集成

- `graph_fixed`：静态库，包含 `src/*.cc` 和 `Runtime/common/status.cc`；开启
  `POSITION_INDEPENDENT_CODE`，保证 Linux/macOS 上可链接进 Python MODULE；
- `graph_fixed_python`：MODULE 库，输出 `Build/graph_fixed/python/_graph_fixed.pyd`；
- `graph_fixed_test`：C++ 单测可执行文件；
- CTest：
  - `graph_fixed_unit`：直接运行 C++ 单测；
  - `graph_fixed_python`：设置 `GRAPH_FIXED_PYTHON_DIR` 后运行 Python 单测；
- CMake 选项：
  - `GRAPH_FIXED_BUILD_PYTHON=ON`；
  - `GRAPH_FIXED_BUILD_TESTS=ON`；
  - `GRAPH_FIXED_AR` / `GRAPH_FIXED_RANLIB` 用于 Zig 等非 MSVC 工具链；
- 编译选项：MSVC `/W4`，其他 `-Wall -Wextra`；
- C++ 标准：C++17，不允许编译器扩展。

## 5. 实施步骤

### 5.1 阶段划分

| 阶段 | 任务 | 产物 | 状态 |
|------|------|------|------|
| S1 | 对照基础设计冻结公共类型和命名空间 | `types.h`、`shape.h`、`storage.h`、`attr.h` | 完成 |
| S2 | 实现 `Tensor`、`Op`、`OpEnv`、`GraphEnv` | 对应头文件和源文件 | 完成 |
| S3 | 实现 `Graph` 构图和 post DFS 排序 | `graph.h`、`graph.cc` | 完成 |
| S4 | 实现 pybind11 绑定和公共 Python 包 | `graph_fixed_python.cc`、`mobilenet_runtime_graph` | 完成 |
| S5 | 建立 CMake 构建和 CTest | `CMakeLists.txt`、`tools/` | 完成 |
| S6 | 编写并运行 C++/Python 测试 | `tests/`、`TestResults.md` | 完成 |
| S7 | 补齐设计文档、测试报告和 skill | 本文件、`TestResults.md`、skill | 完成 |

### 5.2 依赖关系

```text
S1 -> S2 -> S3 -> S4 -> S5 -> S6 -> S7
```

- S2 依赖 S1 的枚举、Shape、Storage、Attr；
- S3 依赖 S2 的 Op/Tensor 所有权和连接接口；
- S4 依赖 S3 的稳定 C++ API；
- S5 必须先于 S6；
- S7 依赖 S6 的真实测试输出。

### 5.3 里程碑

| 里程碑 | 判定条件 |
|--------|----------|
| M1 公共类型可用 | 头文件可独立编译，Shape/Storage 单测通过 |
| M2 图可构建 | input/constant/op/output 连接正确，dump 可用 |
| M3 图可排序 | 拓扑顺序稳定，constant 不进入执行序列 |
| M4 Python 可用 | `_graph_fixed` 可导入，10 个 Python 用例通过 |
| M5 可交付 | 四件套齐全，测试报告记录真实输出 |

### 5.4 风险与应对

| 编号 | 风险 | 实际应对 |
|------|------|----------|
| R-01 | 本机没有 MSVC/C++ 标准库，CMake 默认工具链不可用。 | 使用 Zig 0.17.1 作为 C++ 编译器，静态链接 libc++，扩展只依赖 Python 和系统 DLL。 |
| R-02 | Zig 下 CMake 探测把 `CMAKE_AR` 置为 `NOTFOUND`。 | 增加 `GRAPH_FIXED_AR` / `GRAPH_FIXED_RANLIB` 缓存覆盖，指向 `python -m ziglang ar` 包装脚本。 |
| R-03 | `.cmd` 包装脚本含中文注释时，cmd.exe 按本地代码页解析导致命令错乱。 | 全部改为 ASCII `rem` 注释和 CRLF 行尾。 |
| R-04 | CTest 使用 `python -I`，忽略 `PYTHONUTF8`，中文输出在 GBK 控制台乱码。 | 测试本身仍通过；报告记录该现象，展示可读输出时使用 `PYTHONUTF8=1` 且不使用 `-I`。 |
| R-05 | Python Tensor 是非拥有句柄，Graph 先析构会悬空。 | 绑定使用 `reference` 策略；模块 docstring、公共包 docstring、测试和本文档均明确生命周期。 |
| R-06 | 基础设计文档的 OpType 表与枚举片段不一致。 | 补齐 8 个枚举值，并同步更新基础设计文档、代码、Python 绑定和测试。 |
| R-07 | 公共包的默认构建目录只适用于源码树。 | 优先 `sys.path`，其次 `GRAPH_FIXED_PYTHON_DIR`，最后源码树默认路径；安装场景需设置环境变量或安装扩展。 |
| R-08 | 公开构图接口无法构造依赖环，环检测分支缺少正向测试。 | 保留防御性实现并在测试报告中列为未覆盖项，后续通过内部构建器或反序列化夹具补充。 |
| R-09 | `.cmd` 与 `.sh` 传入的编译器路径斜杠/盘符大小写不一致，CMake 会重建缓存并丢失归档器覆盖。 | 两个脚本统一使用正斜杠，`.sh` 将盘符首字母转小写；交替执行验证为 `ninja: no work to do`。 |

### 5.5 后续工作

1. 为每个 `OpType` 实现真实 `OpEnv`，接入 GPU/OCL kernel。
2. 增加 Finalizer 接口，把构图期 shape/precision/layout 同步到 Tensor Storage。
3. 增加 Manager 接口，遍历 GraphEnv 和 Tensor 完成 workspace 规划。
4. 增加序列化/反序列化，使 Runtime 能从工具链产物重建图。
5. 在 Python 薄包装层补充对象式 `Tensor(..., dtype=..., is_input=True)` 或
   `mark_input(name, dtype)`，与基础设计文档的 wrapper 描述对齐。
6. 优化 `OwnsTensor` 为指针集合索引，支持超大图构图。
7. 增加 Linux/macOS 原生编译验证和 ASan/UBSan 运行。

## 6. 测试与验证设计

### 6.1 测试策略

| 层级 | 目标 | 方式 |
|------|------|------|
| C++ 单元测试 | 验证类型、构图、排序、所有权、GraphEnv、OpEnv 和错误路径 | `graph_fixed_test` 可执行文件，失败返回非 0 |
| Python 单元测试 | 验证绑定可用性、属性转换、dump 稳定性、错误映射和公共包 | `unittest`，10 个用例 |
| CTest 集成 | 验证构建产物可执行、Python 环境变量传递正确 | `ctest --output-on-failure` |
| 人工检查 | 验证 dump 文本可读、错误消息清晰、生命周期文档完整 | 运行示例并阅读输出 |

### 6.2 测试环境

| 项目 | 实际值 |
|------|--------|
| 操作系统 | Windows 11 Home China 10.0.26200 |
| Shell | Git Bash（POSIX sh） |
| Python | 3.13.3 |
| pybind11 | 3.1.0 |
| CMake | 4.4.4 |
| Ninja | 1.13.2 |
| Zig | 0.17.1 |
| 构建类型 | Release |
| 构建目录 | `Build/graph_fixed` |
| 编译器 | `python -m ziglang c++` 包装脚本 |

### 6.3 C++ 用例设计

| 编号 | 用例 | 覆盖点 |
|------|------|--------|
| UT-C01 | `TestShape` | 默认 shape、dims/pitch 乘积、越界、负维度、溢出。 |
| UT-C02 | `TestGraphConstruction` | constant/input/linear/add/output 构图、连接关系、排序顺序、常量排除、SetAttr、重复排序。 |
| UT-C03 | `TestEnvironment` | GraphEnv 注册/查询/重复名、Graph 接管所有权、MegaAttentionOpEnv 生命周期、Forward 未实现、ClearGraphEnv、Graph 析构自动清理 OpEnv。 |
| UT-C04 | `TestErrors` | 空图排序、空名称、0 输出、空输出参数、非法 op 类型、重复 op、空源 tensor、重复常量、空常量输出、非法 shape、外部 tensor。 |

### 6.4 Python 用例设计

| 编号 | 用例 | 覆盖点 |
|------|------|--------|
| UT-P01 | `test_graph_sections` | dump 段标题、计数、graph 名称。 |
| UT-P02 | `test_topological_order` | input/linear/add/output 顺序、src/dst 连接、constant 不出现。 |
| UT-P03 | `test_constant_dump` | constant Storage 文本、精度、形状、不可复用。 |
| UT-P04 | `test_input_attr_storage` | 显式 input 的 attr storage 精度/布局。 |
| UT-P05 | `test_tensor_properties` | name、is_constant、reusable、set_reusable、不暴露 storage。 |
| UT-P06 | `test_legacy_create_input` | legacy input 默认 Storage、排序和 dump。 |
| UT-P07 | `test_attribute_types` | bool/int/float/str/数组属性输出。 |
| UT-P08 | `test_multiple_outputs` | 多输出命名、dst 列表。 |
| UT-P09 | `test_errors` | 重复名称、外部 tensor、空图排序、维度/pitch/attr/out_cnt 错误。 |
| UT-P10 | `test_public_package` | `mobilenet_runtime_graph` 加载和符号一致性。 |

### 6.5 测试命令

```bash
cmake --build Build/graph_fixed --clean-first
ctest --test-dir Build/graph_fixed --output-on-failure
Build/graph_fixed/graph_fixed_test.exe
GRAPH_FIXED_PYTHON_DIR="D:\Mobile_Net\Build\graph_fixed\python" \
  PYTHONUTF8=1 python Graph_Fixed/tests/test_graph_fixed.py -v
```

本地也可使用：

```bash
cmd //c Graph_Fixed\\tools\\build_with_zig.cmd
bash Graph_Fixed/tools/build_with_zig.sh
```

### 6.6 通过标准

1. C++ 单测退出码 0，输出 `ALL TESTS PASSED`；
2. Python 单测 10 个用例全部 `ok`；
3. CTest 2/2 通过；
4. 干净构建无警告；
5. `_graph_fixed` 和 `mobilenet_runtime_graph` 均可导入；
6. `dump_ir` 文本与断言的固定片段一致；
7. 失败路径返回或抛出预期错误。

### 6.7 需求到测试的追溯

| 需求 | C++ 用例 | Python 用例 |
|------|----------|-------------|
| FR-01 | UT-C02、UT-C03 | UT-P01、UT-P10 |
| FR-02 | UT-C02、UT-C04 | UT-P02、UT-P08、UT-P09 |
| FR-03 | UT-C02 | UT-P03、UT-P09 |
| FR-04 | UT-C02 | UT-P04、UT-P06 |
| FR-05 | UT-C02 | UT-P02 |
| FR-06 | UT-C02 | UT-P02、UT-P06 |
| FR-07 | UT-C02、UT-C03 | UT-P02 |
| FR-08 | UT-C03 | 无（未暴露） |
| FR-09 | UT-C03 | 无（未暴露） |
| FR-10 | 无（绑定层） | UT-P01 至 UT-P10 |
| FR-11 | 无（绑定层） | UT-P01 至 UT-P08 |
| FR-12 | 无（Python 包） | UT-P10 |
| FR-13 | 构建与 CTest | 构建与 CTest |

### 6.8 未覆盖项与风险

| 编号 | 未覆盖项 | 影响 | 后续动作 |
|------|----------|------|----------|
| U-01 | 依赖环、非 const tensor 无 producer 的 `VisitOp` 分支 | 公开构图 API 无法构造这些状态，防御性代码未被正向用例覆盖 | 增加内部构图表或反序列化夹具 |
| U-02 | 真实 GPU/OCL kernel 和 `Forward()` 数值结果 | 只能验证 OpEnv 生命周期和未实现错误 | 接入具体 OpEnv 后增加端到端测试 |
| U-03 | 显存申请、workspace 复用、权重 H2D | 不属于本模块 | Manager/Engine 实现后补测 |
| U-04 | 序列化/反序列化 | 当前无格式定义 | 设计序列化格式后增加往返测试 |
| U-05 | 大图性能和递归深度 | 未做万级 op 压测，递归 DFS 可能栈深 | 增加压力测试，必要时改迭代 DFS |
| U-06 | 并发构图/排序 | 类明确不保证线程安全 | 如上层需要并发，增加同步或分图设计 |
| U-07 | Python 错误码区分 | 只抛 `RuntimeError`/`ValueError`，不暴露 `ErrorCode` | 需要程序化处理时增加异常属性 |
| U-08 | 常量数据内容校验 | 只验证元信息和指针标记，不读取权重 | 权重加载接入后增加数值校验 |
| U-09 | Linux/macOS 原生构建 | 仅在 Windows + Zig 验证 | 增加 CI 矩阵 |
| U-10 | 对象式 `mark_input` 包装 | 基础设计文档提到但未实现 | 在 Python 薄包装层补充并测试 |
| U-11 | `byte_size` 与 shape/precision 的交叉校验 | 当前信任调用方，常量元信息可能不一致 | 在权重加载或 Finalizer 阶段校验 |
| U-12 | `SetGraphEnv` / `SetOpEnv` 重复设置或清理顺序 | 替换/清理会让既有 OpEnv 的 `graph_env_` 悬空；当前依赖调用方前置条件 | Engine 接入时增加状态检查或调试断言 |

### 6.9 测试结果

真实环境、命令、输出、结论和风险记录见
[TestResults.md](TestResults.md)。该文件只记录实际执行结果，不复制本文件的
设计性描述。
