---
name: runtime-static-graph-cpp-pybind
description: 从 Runtime 静态图设计出发，实现不依赖 torch/ATen 的 C++17 Graph IR（Graph/Op/Tensor/Storage/GraphEnv/OpEnv），用 pybind11 暴露最小构图接口，并完成 CMake 构建、C++/Python 单测和四件套交付。
---

# Runtime 静态图 C++ IR 与 pybind11 绑定

## 适用场景

- 推理引擎 Runtime 端需要一个模型无关的静态图 IR，表达 op、tensor、constant、
  输入输出和拓扑顺序。
- Runtime 不能依赖 torch/ATen/c10，也不能直接依赖 GPU/OCL 算子库。
- 权重常量以 CPU 指针进入图，后续由 Engine/Manager realize 成设备地址。
- 需要用 pybind11 把图构建接口暴露给 Python 工具链或测试脚本。
- 需要把“构图、排序、查询”与“算子执行、显存管理、序列化”分层。
- 目标环境没有 MSVC，或需要 CMake + Ninja + Zig 的本地验证链。

## 不适用场景

- 需要直接执行 GPU kernel、管理显存或实现量化/Shape 推导。
- 需要图优化、算子融合、自动微分或训练。
- Python 侧需要完整的模型构图 DSL；本方法只提供最小原生 IR 绑定。
- 目标平台无法使用 C++17 或 pybind11。

## 输入

1. 设计文档：模块定位、IR 结构、类接口、错误码、OpType 列表、验证方式。
2. 对象所有权约定：谁拥有 op/tensor/constant/GraphEnv/OpEnv，哪些列表只借用。
3. 构图规则：常量是否进入执行序列、输出 tensor 命名、输入输出边界属性。
4. 运行环境：C++ 标准、Python 版本、pybind11、CMake、编译器。
5. 测试边界：哪些是 IR/绑定测试，哪些留给 Engine/kernel 测试。

## 输出（四件套）

1. 设计文档：六段式（背景与目标、需求分析、概要设计、详细设计、实施步骤、
   测试与验证设计），需求编号可追溯到测试。
2. 代码：
   - 公共类型：Precision/Layout/DeviceType/Shape/Storage/Attr；
   - IR 对象：Tensor/Op/OpEnv/GraphEnv/Graph；
   - pybind11 绑定与 Python 公共包；
   - CMake、CTest、构建脚本。
3. 测试报告：环境、命令、真实输出、通过/失败、未覆盖项与风险。
4. 本 skill 文档。

## 核心不变量

1. `Graph` 用 `map<string, unique_ptr<...>>` 独占持有 op、普通 tensor、
   constant tensor 和 GraphEnv；执行序列、输入/输出/常量列表只保存非拥有
   裸指针。
2. constant tensor 只存在于 constant 容器，不写入普通 tensor 容器；它没有
   producer op，是拓扑遍历的依赖叶子，不进入执行序列。
3. 普通 tensor 至少有一个 producer；输入 tensor 的 producer 是 input op。
4. 所有 `Create*` 先完成名称、参数、归属和输出名称校验，再提交图结构；
   失败不能留下半成品。
5. `SortOp` 从输出 op 出发做 post DFS；状态 0/1/2 分别表示未访问/访问中/
   完成；状态 1 表示依赖环。
6. 析构顺序必须保证 OpEnv 在 GraphEnv 之前释放。若 GraphEnv 是 Graph 的
   最后一个成员，应在 Graph 析构函数中先对所有 op 调用 `ClearOpEnv()`。
7. Python 返回的 Tensor 必须是 `reference` 非拥有句柄；Graph 必须比所有
   Tensor 句柄存活更久。
8. `dump_ir` 只输出稳定文本，不输出真实指针地址，属性按 key 排序。
9. Runtime Graph 头文件和目标不得 include/链接 torch/ATen/c10 或 GPU/OCL
   运行时。

## 标准流程

### 1. 冻结边界与所有权

- 先写设计文档，明确 Graph/Op/Tensor/Storage/GraphEnv/OpEnv 的职责和不做
  什么。
- 画所有权图：拥有关系用 `unique_ptr`，借用关系用裸指针，写清生命周期。
- 决定 constant 是否进入执行序列、输入输出 op 的 attr key、输出 tensor
  命名规则。
- 固定 OpType 顺序；枚举顺序是 IR 契约，Python 绑定和序列化都不能随意调整。

### 2. 实现公共类型

- `Shape` 使用定长数组 `dims/pitch` 和 `max_dim`，默认维度值为 1，
  `max_dim=0` 表示空 shape。
- 乘积函数遇到负维度、越界或 `int64_t` 溢出返回 -1。
- `Storage` 只保存描述，不拥有 `data_ptr`；`reusable` 默认 true。
- `AttrValue` 用 `std::variant` 限定类型集合；`AttrMap` 用 `std::map`
  保证稳定遍历顺序。
- 所有枚举提供稳定字符串名称，供日志和 dump 使用。

### 3. 实现 IR 对象

- `Tensor`：名称、producer/consumer 列表、内嵌 Storage、constant 标记、
  reusable 访问器。
- `Op`：名称、类型、属性、输入输出 tensor、`unique_ptr<OpEnv>`。
- `GraphEnv`：按名称注册/查询共享 Storage，拒绝空名和重复名。
- `OpEnv`：`Init(GraphEnv*)`、`Forward()`、`NeedTempData()`；派生类实现
  真实 kernel。
- 连接维护函数设为私有，`Graph` 通过 `friend` 调用，避免外部破坏双向一致性。

### 4. 实现构图与排序

- `CreateOp`：校验 op 名称、输出参数、输出数量、op 类型、源 tensor 归属；
  预生成并校验全部输出 tensor 名称后再提交。
- `CreateConstant`：校验名称和 shape，复制 Storage，强制 `reusable=false`，
  标记 constant，不创建 producer。
- `CreateInput`：legacy 写默认 `Storage{}`，显式重载把 precision/layout
  写入 `attr["storage"]`；输入 op 与 tensor 同名。
- `CreateOutput`：只创建 `kOutput` op，不产生 tensor，写入默认 storage attr。
- `SortOp`：清空执行序列；无输出返回状态错误；从输出做 post DFS；常量跳过；
  非 const 无 producer 报错；重复排序结果稳定。
- 析构函数先清空所有 OpEnv，再让成员逆序析构。

### 5. pybind11 绑定

- 只暴露需要的最小面：Graph、Tensor、枚举；默认不暴露 Op/Storage/Shape。
- Tensor 使用 `py::return_value_policy::reference`，不转移所有权。
- Python attr 转换顺序：bool -> int -> float -> str -> list/tuple；
  bool 必须先于 int 判断。
- 数组区分 int/double/str，拒绝数字与字符串混合；空数组按 int 数组处理。
- `Status` 失败抛 `std::runtime_error`（Python `RuntimeError`）；类型转换失败
  抛 `std::invalid_argument`（Python `ValueError`）。
- `create_constant` 只接收指针，不读取数据；文档中写明缓冲区生命周期。
- `dump_ir` 用 `op_list_` 输出 ops，段顺序、属性顺序和命名必须确定。

### 6. 构建与测试

- CMake 构建静态库、Python MODULE、C++ 测试，并注册 CTest。
- C++ 测试覆盖 Shape、构图、排序、GraphEnv/OpEnv 生命周期和错误路径。
- Python 测试覆盖正常构图、属性类型、多输出、legacy 输入、dump 和错误映射。
- 测试报告必须记录真实命令和输出，不能只写“测试通过”。
- 未覆盖项要明确列出，例如依赖环防御分支、真实 kernel、显存、序列化。

## 检查清单

- [ ] Runtime 头文件无 torch/ATen/c10 include，目标无 torch 链接。
- [ ] constant 不在普通 tensor 容器，无 producer，`reusable=false`。
- [ ] 普通 tensor 都有 producer；输入 tensor 的 producer 是 input op。
- [ ] 所有 `Create*` 失败时图状态不变。
- [ ] 输出 tensor 命名稳定，单输出和多输出规则一致。
- [ ] `SortOp` 结果 producer 在前、consumer 在后，constant 不出现。
- [ ] Graph 析构先清 OpEnv，再释放 GraphEnv。
- [ ] Python Tensor 句柄非拥有，文档写明 Graph 生命周期。
- [ ] `dump_ir` 不输出真实指针，属性顺序稳定。
- [ ] C++/Python 源码通过 Google Style 和 80 显示列检查。
- [ ] 每个模块/类/函数有中文注释，覆盖边界、输入输出、主要逻辑。
- [ ] 干净构建无警告，CTest 全部通过。
- [ ] 四件套齐全，测试报告含环境、命令、实际输出、未覆盖项。

## 常见陷阱

1. **GraphEnv 先于 OpEnv 析构**：若 `GraphEnv` 是 Graph 的最后一个成员，
   默认析构会先释放它，再析构 `op_map_` 中的 OpEnv。派生 OpEnv 的析构
   函数若访问 `graph_env_` 就是 use-after-free。应在 Graph 析构函数中先
   `ClearOpEnv()`。
2. **Python Tensor 悬空**：`py::return_value_policy::reference` 不延长
   Graph 生命周期；Python 先释放 Graph 再访问 Tensor 会崩溃。必须在模块
   docstring、公共包和测试中反复说明。
3. **constant 被当成普通 tensor**：如果 constant 写入普通 tensor 容器，
   `SortOp` 会要求 producer，或把常量错误地加入执行序列。应独立容器持有，
   并在遍历时按 `IsConstant()` 跳过。
4. **部分提交**：多输出 op 先创建一个输出、再发现第二个名称冲突，会留下
   半成品。必须预生成并校验全部输出名称后再创建对象。
5. **输出命名漂移**：`name.out` 和 `name.out.<i>` 规则不固定，dump、测试和
   外部加载器都会不稳定。固定规则并写进设计文档。
6. **dump 输出指针**：`StorageToString` 打印真实地址会让测试结果随进程变化，
   也无法做快照。只输出 `data=set`/`data=unset`。
7. **Zig/CMake 归档器丢失**：CMake 探测可能把 `CMAKE_AR` 置为
   `CMAKE_AR-NOTFOUND`。用 `GRAPH_AR` / `GRAPH_RANLIB` 缓存
   覆盖，指向 `python -m ziglang ar` 包装脚本。
8. **构建脚本路径大小写不一致**：`.cmd` 用反斜杠/大写盘符、`.sh` 用正斜杠/
   大写盘符，CMake 会认为编译器变化并重建缓存，导致归档器覆盖变量丢失。
   两个脚本统一正斜杠并把盘符转小写。
9. **`.cmd` 中文注释**：cmd.exe 按本地代码页解析批处理，中文注释可能导致
   “不是内部或外部命令”。`.cmd` 只使用 ASCII `rem` 注释和 CRLF。
10. **CTest 的 `python -I`**：`-I` 会忽略 `PYTHONUTF8`，中文输出在 GBK
    控制台乱码。测试可通过，但日志展示要另跑一次 `PYTHONUTF8=1`。
11. **空名称校验与注释不一致**：构造函数不能返回 `Status` 时，要么在注释中
    明确“允许为空”，要么在更上层校验；不要让注释承诺代码没有的行为。
12. **`Shape::GetByteSize` 命名误导**：它返回 pitch 元素数，不是字节数；
    真实字节数由 `Storage::byte_size` 表达，注释必须写清楚。
13. **Python bool 被当成 int**：`isinstance(True, int)` 为真；attr 转换必须
    先判断 `bool`，否则布尔属性会变成整数。
14. **数组类型混合**：`[1, 2.5]` 必须拒绝或明确提升规则；不要静默生成
    不可解释的 variant。
15. **递归 DFS 深度**：超深图可能栈溢出；大图场景改为显式栈迭代。
16. **测试只覆盖快乐路径**：至少覆盖空图排序、重复名称、外部 tensor、
    0 输出、空输出参数、非法维度、非法 attr 类型。

## 验证命令（本仓库）

```bash
# 干净构建 + CTest
rm -rf Build/graph
bash Graph/tools/build_with_zig.sh

# Windows cmd 构建
cmd //c "Graph\\tools\\build_with_zig.cmd"

# C++ 单测
Build/graph/graph_test.exe

# Python 单测
GRAPH_PYTHON_DIR="D:\Mobile_Net\Build\graph\python" \
  PYTHONUTF8=1 python Graph/tests/test_graph.py -v

# CTest
ctest --test-dir Build/graph --output-on-failure
```

正式构建（有 MSVC/系统 C++ 工具链时）：

```bash
cmake -S Graph -B Build/graph -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build Build/graph
ctest --test-dir Build/graph --output-on-failure
```

## 参考实现

- 设计、实施与测试：`Graph/.doc/Graph设计文档.md`
- 公共类型：`Graph/include/runtime_graph/types.h`、`shape.h`、
  `storage.h`、`attr.h`
- IR 对象：`Graph/include/runtime_graph/tensor.h`、`op.h`、
  `op_env.h`、`graph_env.h`、`graph.h`
- 实现：`Graph/src/`
- Python 绑定：`Graph/python/graph_python.cc`
- Python 公共包：`Graph/python/mobilenet_runtime_graph/__init__.py`
- 测试：`Graph/tests/graph_test.cc`、
  `Graph/tests/test_graph.py`
- 测试报告：`Graph/TestResults.md`
