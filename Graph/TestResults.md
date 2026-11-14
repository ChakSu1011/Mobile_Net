# Runtime Graph 模块测试结果

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-11-14 |
| 测试对象 | `Graph` C++ 静态库、`_graph` Python 扩展、公共模块 |
| 设计文档 | [Graph 设计文档](.doc/Graph设计文档.md) |
| 构建目录 | `Build/graph` |
| 结论 | 已执行的构建、C++ 单测、Python 单测、CTest、包导入和格式检查全部通过；未覆盖项见第 8 节 |

本报告只记录实际执行的命令、输出和结论。报告中引用的构建输出为原始终端
输出摘录，未修改测试结果。

## 1. 测试范围与环境

### 1.1 范围

| 范围 | 内容 |
|------|------|
| 已覆盖 | 公共类型、Shape/Storage、Graph 构图、拓扑排序、GraphEnv、OpEnv 生命周期、C++ 错误路径、pybind 绑定、属性转换、dump_ir、公共模块加载、构建脚本 |
| 未覆盖 | 真实 GPU/OCL kernel、显存分配、权重 H2D、序列化、并发、万级大图、Linux/macOS 原生构建 |

### 1.2 环境

| 项目 | 实际值 |
|------|--------|
| 操作系统 | Windows 11 Home China 10.0.26200 |
| Shell | Git Bash（POSIX sh） |
| Python | 3.13.3 |
| Python 路径 | `C:\Users\18626\AppData\Local\Programs\Python\Python313\python.exe` |
| pybind11 | 3.1.0 |
| CMake | 4.4.4 |
| Ninja | 1.13.2 |
| Zig | 0.17.1 |
| 编译器 | Zig 内置 Clang 22.1.8，通过 `python -m ziglang c++` 调用 |
| 构建类型 | Release |
| C++ 标准 | C++17 |

### 1.3 测试命令总览

```bash
# 1. 干净构建与 CTest
bash Graph/tools/build_with_zig.sh

# 2. C++ 单测直接运行
Build/graph/graph_test.exe

# 3. Python 单测
GRAPH_PYTHON_DIR="D:\Mobile_Net\Build\graph\python" \
  PYTHONUTF8=1 python Graph/tests/test_graph.py -v

# 4. 公共模块导入
python -c "import sys; sys.path.insert(0, r'Graph\python'); \
  import mobilenet_runtime_graph as m; print(m.Graph, m.Tensor)"

# 5. Windows cmd 构建脚本
cmd //c "Graph\\tools\\build_with_zig.cmd"
```

## 2. 构建验证（TR-01）

### 2.1 干净构建

删除 `Build/graph` 后执行 `bash Graph/tools/build_with_zig.sh`，实际输出：

```text
-- The CXX compiler identification is Clang 22.1.8
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: d:/Mobile_Net/Graph/tools/zig_cxx.cmd - skipped
-- Detecting CXX compile features - done
-- Found Python3: C:/Users/18626/AppData/Local/Programs/Python/Python313/python.exe (found suitable version "3.13.3", minimum required is "3.10") found components: Interpreter Development.Module
-- pybind11::lto disabled (problems with undefined symbols for MinGW for now)
-- pybind11::thin_lto disabled (problems with undefined symbols for MinGW for now)
-- Found pybind11: C:/Users/18626/AppData/Local/Programs/Python/Python313/Lib/site-packages/pybind11/include (found version "3.1.0")
-- Configuring done (24.3s)
-- Generating done (0.1s)
-- Build files have been written to: D:/Mobile_Net/Build/graph
[1/16] Building CXX object CMakeFiles/graph.dir/src/op_type.cc.obj
[2/16] Building CXX object CMakeFiles/graph.dir/src/types.cc.obj
[3/16] Building CXX object CMakeFiles/graph.dir/src/op_env.cc.obj
[4/16] Building CXX object CMakeFiles/graph.dir/src/shape.cc.obj
[5/16] Building CXX object CMakeFiles/graph.dir/D_/Mobile_Net/Runtime/common/status.cc.obj
[6/16] Building CXX object CMakeFiles/graph.dir/src/graph_env.cc.obj
[7/16] Building CXX object CMakeFiles/graph.dir/src/tensor.cc.obj
[8/16] Building CXX object CMakeFiles/graph.dir/src/attr.cc.obj
[9/16] Building CXX object CMakeFiles/graph.dir/src/storage.cc.obj
[10/16] Building CXX object CMakeFiles/graph.dir/src/op.cc.obj
[11/16] Building CXX object CMakeFiles/graph.dir/src/graph.cc.obj
[12/16] Building CXX object CMakeFiles/graph.dir/tests/graph_test.cc.obj
[13/16] Linking CXX static library libgraph.a
[14/16] Linking CXX executable graph_test.exe
[15/16] Building CXX object CMakeFiles/graph_python.dir/python/graph_python.cc.obj
[16/16] Linking CXX shared module python\_graph.pyd
```

结论：16/16 目标构建成功，输出中没有编译警告。产物为
`Build/graph/python/_graph.pyd`。

### 2.2 构建脚本切换验证

`.cmd` 与 `.sh` 脚本使用相同的正斜杠路径和盘符大小写后，交替执行不会再
触发 CMake 缓存重建：

```text
$ cmd //c "Graph\\tools\\build_with_zig.cmd"
-- Configuring done (0.5s)
-- Generating done (0.1s)
-- Build files have been written to: D:/Mobile_Net/Build/graph
ninja: no work to do.

$ bash Graph/tools/build_with_zig.sh
-- Configuring done (0.4s)
-- Generating done (0.1s)
-- Build files have been written to: D:/Mobile_Net/Build/graph
ninja: no work to do.
```

两种脚本随后都执行 CTest，均为 2/2 通过。归档器缓存实际值：

```text
CMAKE_AR:FILEPATH=d:/Mobile_Net/Graph/tools/zig_ar.cmd
CMAKE_CXX_COMPILER_AR:FILEPATH=d:/Mobile_Net/Graph/tools/zig_ar.cmd
CMAKE_RANLIB:FILEPATH=d:/Mobile_Net/Graph/tools/zig_ranlib.cmd
```

### 2.3 CTest 集成（TR-02）

```text
Test project D:/Mobile_Net/Build/graph
    Start 1: graph_unit
1/2 Test #1: graph_unit .......................   Passed    0.19 sec
    Start 2: graph_python
2/2 Test #2: graph_python .....................   Passed    0.32 sec

100% tests passed out of 2

Total Test time (real) =   0.66 sec
```

结论：CTest 2/2 通过，退出码 0。

## 3. C++ 单元测试（TR-03）

### 3.1 执行命令与输出

```bash
$ Build/graph/graph_test.exe
graph_test: ALL TESTS PASSED
$ echo $?
0
```

### 3.2 用例与实际覆盖

C++ 测试文件包含 4 个测试函数和 99 个 `CHECK` 断言：

| 用例 | 实际覆盖内容 | 结果 |
|------|--------------|------|
| `TestShape` | 默认 shape（`max_dim=0`、元素数 1）、dims/pitch 乘积、`max_dim` 越界、负维度、`int64_t` 溢出 | 通过 |
| `TestGraphConstruction` | constant 的不可复用标记和空 producer、显式 input 的 attr storage、linear/add/output 连接、排序顺序 `input -> linear -> add -> output`、constant 不进入执行序列、`SetAttr`、空属性名失败、`SetOpEnv(nullptr)` 失败、重复 `SortOp` 稳定 | 通过 |
| `TestEnvironment` | GraphEnv 注册/查询/重复名、Graph 接管所有权、MegaAttentionOpEnv 的 `NeedTempData`/`Init`/未实现 `Forward`、`ClearOpEnv`、Graph 析构自动清理 OpEnv | 通过 |
| `TestErrors` | 空图排序、空 op 名、0 输出、空输出参数、非法 op 类型、重复 op、空源 tensor、重复常量名、空常量输出、非法 shape、外部 tensor、空输出名 | 通过 |

### 3.3 结论

C++ 单测退出码 0，全部断言通过。测试只使用内存元数据，不加载模型、不访问
GPU，也不读取权重内容。

## 4. Python 单元测试（TR-04）

### 4.1 执行命令

```bash
$ GRAPH_PYTHON_DIR="D:\Mobile_Net\Build\graph\python" \
    PYTHONUTF8=1 python Graph/tests/test_graph.py -v
```

### 4.2 实际输出

```text
test_attribute_types (__main__.GraphPythonTest.test_attribute_types)
Python dict 的 bool/int/float/str/数组属性可写入并稳定输出。 ... ok
test_constant_dump (__main__.GraphPythonTest.test_constant_dump)
constant tensor 的 storage 文本包含精度、形状和不可复用标记。 ... ok
test_errors (__main__.GraphPythonTest.test_errors)
重复名称、外部 tensor、空图排序和非法参数抛出异常。 ... ok
test_graph_sections (__main__.GraphPythonTest.test_graph_sections)
dump_ir 的段标题、计数和顺序符合预期。 ... ok
test_input_attr_storage (__main__.GraphPythonTest.test_input_attr_storage)
显式 create_input 把 precision/layout 写入 Input attr storage。 ... ok
test_legacy_create_input (__main__.GraphPythonTest.test_legacy_create_input)
legacy create_input 写入默认 Storage，排序与 dump 仍可用。 ... ok
test_multiple_outputs (__main__.GraphPythonTest.test_multiple_outputs)
out_cnt > 1 时生成稳定命名的多个输出 tensor。 ... ok
test_public_module (__main__.GraphPythonTest.test_public_module)
公共模块可加载扩展，并导出与扩展相同的图类型和枚举。 ... ok
test_tensor_properties (__main__.GraphPythonTest.test_tensor_properties)
Tensor 暴露名称、constant 标记和复用标记，且不暴露 Storage 对象。 ... ok
test_topological_order (__main__.GraphPythonTest.test_topological_order)
op 序列为 input、linear、add、output，constant 不进入序列。 ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.002s

OK
```

### 4.3 覆盖点

| 用例 | 覆盖点 | 结果 |
|------|--------|------|
| `test_graph_sections` | dump 段标题、计数、graph 名称 | 通过 |
| `test_topological_order` | 拓扑顺序、src/dst、constant 不进入执行序列 | 通过 |
| `test_constant_dump` | constant Storage 文本、精度、形状、不可复用 | 通过 |
| `test_input_attr_storage` | 显式 input 的 attr storage | 通过 |
| `test_tensor_properties` | name、is_constant、reusable、set_reusable、不暴露 storage | 通过 |
| `test_legacy_create_input` | legacy input 默认 Storage、排序和 dump | 通过 |
| `test_attribute_types` | bool/int/float/str/数组属性 | 通过 |
| `test_multiple_outputs` | 多输出命名和 dst 列表 | 通过 |
| `test_public_module` | 公共模块加载和符号一致性 | 通过 |
| `test_errors` | 重复名称、外部 tensor、空图排序、维度/pitch/attr/out_cnt 错误 | 通过 |

### 4.4 结论

Python 单测 10/10 通过，退出码 0。错误路径中，C++ `Status` 失败转换为
`RuntimeError`，绑定层参数错误转换为 `ValueError`，与设计一致。

## 5. 公共模块与 dump_ir（TR-05）

### 5.1 公共模块导入

```bash
$ python -c "import sys; sys.path.insert(0, r'Graph\python'); \
    import mobilenet_runtime_graph as m; \
    print(m.Graph.__name__, m.Tensor.__name__, m.OpType.kLinear, \
    m.Precision.kFloat16)"
Graph Tensor OpType.kLinear Precision.kFloat16
```

`mobilenet_runtime_graph` 优先从 `sys.path` 加载 `_graph`；找不到时
使用 `GRAPH_PYTHON_DIR`，最后回退到源码树的
`Build/graph/python`。本次验证同时覆盖了环境变量方式和源码树默认
路径方式。

### 5.2 dump_ir 实际输出

构造 `input -> linear(weight) -> output` 后调用 `sort_op()` 和
`dump_ir()`，实际输出：

```text
graph report_graph
inputs 1
  input -> [input]
constants 1
  weight storage(precision=float16,layout=row_major,device=host,shape=[2,3],offset=0,bytes=12,data=set,reusable=false)
ops 3
  [0] input input src=[] dst=[input] attrs={storage=storage(precision=float16,layout=row_major,device=unknown,shape=[],offset=0,bytes=0,data=unset,reusable=true)}
  [1] linear linear src=[input,weight] dst=[linear.out] attrs={axis=1}
  [2] output output src=[linear.out] dst=[] attrs={storage=storage(precision=unknown,layout=row_major,device=unknown,shape=[],offset=0,bytes=0,data=unset,reusable=true)}
outputs 1
  output <- [linear.out]
```

结论：段顺序、计数、属性顺序和 tensor 连接符合设计；输出不含真实指针地址，
可重复断言。

## 6. 依赖与格式检查（TR-06）

### 6.1 禁止依赖扫描

对 `Graph/include`、`Graph/src`、`Graph/python` 和测试源
执行：

```bash
grep -RInE 'torch|ATen|c10|BR2XX|SUPA' \
  Graph/include Graph/src \
  Graph/python/graph_python.cc Graph/tests
```

实际命中 3 处，全部是注释：

```text
Graph/include/graph.h:95:  //   storage：权重 storage 描述，data_ptr 通常指向 Python torch tensor 的
Graph/include/types.h:10://   - 不依赖 ATen/torch/c10，也不包含 GPU/OCL 算子库头文件；
Graph/python/graph_python.cc:13://   - 不链接 torch/ATen，也不依赖 GPU/OCL 运行时。
```

源码中没有 torch/ATen/c10 的 include，CMake 也没有链接相关目标。`BR2XX` 和
`SUPA` 在源码中无命中；基础设计文档中的对应表述已替换为 `GPU/OCL`。

### 6.2 扩展 DLL 导入表

解析 `_graph.pyd` 的 PE32+ 导入表，实际依赖：

```text
KERNEL32.dll
api-ms-win-crt-convert-l1-1-0.dll
api-ms-win-crt-heap-l1-1-0.dll
api-ms-win-crt-locale-l1-1-0.dll
api-ms-win-crt-multibyte-l1-1-0.dll
api-ms-win-crt-private-l1-1-0.dll
api-ms-win-crt-runtime-l1-1-0.dll
api-ms-win-crt-stdio-l1-1-0.dll
api-ms-win-crt-string-l1-1-0.dll
api-ms-win-crt-time-l1-1-0.dll
python313.dll
```

没有 torch、ATen、c10、libc++ 或 GPU/OCL 运行库依赖。

### 6.3 80 列显示宽度检查

按东亚字符双宽度计算 C++ 和 Python 源码的显示列宽，实际输出：

```text
C++ display width <= 80: OK
Python display width <= 80: OK
```

### 6.4 注释检查

逐文件检查结果：

- 每个源文件有模块级中文注释，说明职责、边界、依赖；
- 每个类和结构体有类级注释；
- 每个函数（含成员函数、静态函数、绑定 lambda 所在函数）有中文注释；
- 注释覆盖功能边界、输入输出、主要逻辑和错误路径；
- 构建脚本 `.cmd` 使用 ASCII 注释，避免本地代码页导致命令解析错误。

## 7. 测试过程中发现并修复的问题

| 编号 | 现象 | 原因 | 修复 |
|------|------|------|------|
| P-01 | 新增 `CountingOpEnv` 后 C++ 测试编译失败：`expected class name`、`only virtual member functions can be marked 'override'`。 | 测试文件使用 `OpEnv` 但未添加 `using mobilenet::runtime_graph::OpEnv;`。 | 补充 using 声明后重新干净构建，16/16 通过。 |
| P-02 | 先运行 `.sh` 再运行 `.cmd` 时，CMake 删除缓存并重新配置，链接静态库时报 `CMAKE_AR-NOTFOUND`。 | 两个脚本传入的编译器路径盘符大小写/斜杠风格不同，CMake 认为编译器变化并重建缓存，归档器覆盖变量在该次配置中丢失。 | `.cmd` 统一为正斜杠，`.sh` 将盘符首字母转小写；复测两种脚本交替执行均为 `ninja: no work to do`，归档器缓存保持 `d:/.../zig_ar.cmd`。 |
| P-03 | `.cmd` 包装脚本包含中文注释时，cmd.exe 按本地代码页解析，出现“不是内部或外部命令”或 Zig 参数乱码。 | 批处理文件编码与系统代码页不一致。 | 所有 `.cmd` 改为 ASCII `rem` 注释和 CRLF 行尾。 |
| P-04 | CTest 的 Python 用例输出中文在 GBK 控制台显示乱码。 | CTest 使用 `python -I`，忽略 `PYTHONUTF8`/`PYTHONIOENCODING`。 | 测试本身通过；可读日志改用 `PYTHONUTF8=1` 且不使用 `-I` 运行。本报告第 4 节为可读输出。 |
| P-05 | 测试复核发现 `TestEnvironment` 在 `ClearGraphEnv()` 之后仍解引用已释放的 `graph_env_ptr`，属于测试自身的 use-after-free。 | 断言顺序错误：应先验证共享存储，再清理 GraphEnv。 | 将 `GetSharedStorage("kv_cache")` 断言移到 `ClearGraphEnv()` 之前，重新构建并执行 CTest，2/2 通过。 |

## 8. 未覆盖项与风险

| 编号 | 未覆盖项 | 影响 | 建议 |
|------|----------|------|------|
| U-01 | 依赖环、非 const tensor 无 producer 的 `VisitOp` 分支 | 公开构图 API 无法构造这些状态，防御性分支没有正向用例 | 增加内部构图表或反序列化夹具 |
| U-02 | 真实 GPU/OCL kernel、`Forward()` 数值结果 | 当前只能验证 OpEnv 接口和 `UNSUPPORTED_OP` | 接入具体 OpEnv 后增加端到端数值测试 |
| U-03 | 显存申请、workspace 复用、权重 H2D | 不属于 Graph 模块 | Manager/Engine 实现后补测 |
| U-04 | 序列化/反序列化 | 当前没有格式定义 | 定义格式后增加往返和版本兼容测试 |
| U-05 | 万级 op 大图和递归深度 | 未做压力测试；递归 DFS 可能受栈深限制 | 增加大图基准，必要时改为迭代 DFS |
| U-06 | 并发构图/排序 | 类明确不保证线程安全 | 如上层需要并发，增加同步或按图隔离 |
| U-07 | Python 错误码区分 | 只抛 `RuntimeError`/`ValueError`，不暴露 `ErrorCode` | 需要程序化处理时增加异常属性 |
| U-08 | 常量权重内容校验 | 只验证元信息和指针标记，不读取权重 | 权重加载接入后增加数值校验 |
| U-09 | Linux/macOS 原生构建 | 仅在 Windows 11 + Zig 上验证 | 增加 CI 构建矩阵 |
| U-10 | 对象式 `mark_input` 包装 | 基础设计文档提到但当前未实现 | 在 Python 薄包装层补充并测试 |
| U-11 | `byte_size` 与 shape/precision 的交叉校验 | 当前信任调用方，常量元信息可能不一致 | 在权重加载或 Finalizer 阶段校验 |
| U-12 | `SetGraphEnv`/`SetOpEnv` 的重复设置和清理顺序 | 替换/清理会让既有 OpEnv 的 `graph_env_` 悬空；当前靠调用方遵守“先 ClearOpEnv”前置条件 | Engine 接入时增加状态检查或调试断言 |

## 9. 结论

1. 干净构建 16/16 成功，无编译警告；
2. C++ 单测 99 个 `CHECK` 断言全部通过，输出
   `graph_test: ALL TESTS PASSED`；
3. Python 单测 10/10 通过；
4. CTest 2/2 通过；
5. `_graph` 与 `mobilenet_runtime_graph` 均可导入；
6. `dump_ir` 输出稳定、可重复；
7. 扩展不依赖 torch/ATen/c10/libc++，只依赖 Python 和 Windows 系统 DLL；
8. `.cmd` 与 `.sh` 构建脚本可交替使用，归档器覆盖生效；
9. C++ 和 Python 源码通过 80 列显示宽度检查。

本报告记录的是 IR、绑定和构建层面的验证结果。真实算子执行、显存管理、
序列化和跨平台构建仍未覆盖，相关风险已在第 8 节列出，不能据此宣称推理引擎
端到端可用。

## 附录 A：复现命令

```bash
# 干净构建 + CTest（Git Bash）
rm -rf Build/graph
bash Graph/tools/build_with_zig.sh

# 或者使用 Windows cmd
cmd //c "Graph\\tools\\build_with_zig.cmd"

# C++ 单测
Build/graph/graph_test.exe

# Python 单测（可读中文输出）
GRAPH_PYTHON_DIR="D:\Mobile_Net\Build\graph\python" \
  PYTHONUTF8=1 python Graph/tests/test_graph.py -v

# 公共模块导入
python -c "import sys; sys.path.insert(0, r'Graph\python'); \
  import mobilenet_runtime_graph as m; print(m.Graph, m.Tensor)"

# CTest
ctest --test-dir Build/graph --output-on-failure
```
