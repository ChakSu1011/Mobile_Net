# Graph 模块测试报告

- 报告日期：2026-11-08
- 被测对象：`Graph/`（Graph IR、RuntimeGraph、Builder/Validator、融合规则、TensorDesc 转换、Python 绑定）
- 对应设计：`Graph/Graph设计文档.md`
- 代码状态：工作区快照（仓库未初始化 git，无 commit 号）

## 1. 测试环境

| 项目 | 实际配置 |
| --- | --- |
| 操作系统 | Windows 11 10.0.26200（AMD64） |
| CPU | Intel64 Family 6 Model 197 Stepping 2 |
| Python | 3.13.3 |
| PyTorch | 2.12.0+cpu |
| C++ 编译器 | Zig 0.17.1（`python -m ziglang c++`），libc++ |
| 编译标准/优化 | `-std=c++17 -O2 -g -DNDEBUG -Wall -Wextra` |
| 构建脚本 | `Tools/build_graph_zig.py`（CI/本机复现） |
| 正式构建入口 | `Graph/CMakeLists.txt`（CMake 4.4.4 已安装；本机无 MSVC/LibTorch，未执行，见第 7 节） |
| 测试框架 | C++ 独立断言宏；Python 标准库 `unittest` |
| 代码格式化 | clang-format 22.1.6（`--style=Google`） |

说明：本机没有 MSVC 与 LibTorch CMake 包，因此使用设计文档允许的
`MOBILENET_GRAPH_HOST_TORCH_BRIDGE` 桥接方式编译 Python 扩展；C++ 目标
仍使用 C++17 与 `-O2`，与设计文档一致。

## 2. 构建与测试命令

```bash
python Tools/build_graph_zig.py --clean --run-tests
```

该命令依次执行：

1. 编译 `Graph/cpp/common`、`Graph/cpp/ir`、`Graph/cpp/runtime`、
   `Runtime/common`、`Runtime/kernel` 与 Python 绑定源文件；
2. 链接 `Build/graph/python/_graph.pyd`；
3. 链接并运行 `Build/graph/tests/test_runtime_graph.exe`；
4. 以 `-I` 隔离模式运行 Python 单元测试与 Qwen3 集成测试。

辅助命令：

```bash
# 只运行 C++ RuntimeGraph 单测
Build/graph/tests/test_runtime_graph.exe

# 只运行 Python 单元测试
PYTHONPATH="Build/graph/python;Graph/cpp/python;Tests/unit/graph" \
  python -I Tests/unit/graph/test_graph_ir.py

# 只运行双端一致性集成测试
PYTHONPATH="Build/graph/python;Graph/cpp/python;Tests/integration/graph" \
  python -I Tests/integration/graph/test_qwen3_cross_language.py
```

## 3. 测试结果汇总

| 测试集 | 用例数 | 结果 | 实际输出 |
| --- | --- | --- | --- |
| C++ RuntimeGraph 单测 | 7 个测试函数（含多条断言） | 通过 | `PASS runtime_graph tests`，退出码 0 |
| Python Graph IR 单测 | 22 | 通过 | `Ran 22 tests in 0.018s` / `OK` |
| Qwen3 双端一致性集成测试 | 9 | 通过 | `Ran 9 tests in 0.024s` / `OK` |
| 构建告警 | - | 通过 | 无 `warning:` / `error:` |

构建日志（关键行）：

```text
{"python": "3.13.3", "repository": "D:\\Mobile_Net"}
...
PASS runtime_graph tests
......................
----------------------------------------------------------------------
Ran 22 tests in 0.018s

OK
.........
----------------------------------------------------------------------
Ran 9 tests in 0.024s

OK

built D:\Mobile_Net\Build\graph\python\_graph.pyd
built D:\Mobile_Net\Build\graph\tests\test_runtime_graph.exe
```

完整日志保存在 `Build/graph_test_output.log`。

## 4. 测试用例执行明细

### 4.1 C++ RuntimeGraph 单测

文件：`Tests/unit/graph/test_runtime_graph.cc`

| 测试函数 | 覆盖设计用例 | 实际结果 |
| --- | --- | --- |
| `TestForwardCopiesData` | UT-GRAPH-012/013 | 通过：事件顺序为 `upload → execute:cast → download → synchronize`，输出字节与输入一致 |
| `TestExecutionValidationRejectsMissingBindings` | UT-GRAPH-014 | 通过：缺 Kernel 返回 `STATE_ERROR`；缺 I/O binding 返回 `INVALID_ARGUMENT` |
| `TestBindRules` | UT-GRAPH-015 | 通过：重复 BindBackend/BindArena 返回 `STATE_ERROR`，非法 arena 与 I/O 节点绑定 Kernel 被拒绝 |
| `TestArenaRangeValidation` | UT-GRAPH-015 | 通过：arena 越界返回 `BUFFER_ERROR` |
| `TestKernelErrorPropagation` | UT-GRAPH-014/IT-GRAPH-005 | 通过：Kernel 错误返回 `KERNEL_ERROR`，消息含节点名 `cast.0` |
| `TestReleaseDestroysAllKernels` | UT-GRAPH-015/IT-GRAPH-006 | 通过：单个 DestroyKernel 失败后继续释放其余 Kernel；重复 Release 返回 OK；Release 后 Forward 返回 `STATE_ERROR` |
| `TestCanonicalJsonDeterminism` | UT-GRAPH-017 | 通过：多次输出字节一致 |

C++ 测试入口输出：

```text
PASS runtime_graph tests
```

### 4.2 Python Graph IR 单测

文件：`Tests/unit/graph/test_graph_ir.py`，共 22 个用例：

| 用例 | 设计编号 | 结果 |
| --- | --- | --- |
| `test_minimal_graph_and_canonical_json` | UT-GRAPH-001/017 | 通过 |
| `test_multi_branch_topology` | UT-GRAPH-002 | 通过 |
| `test_cycle_rejected` | UT-GRAPH-003 | 通过 |
| `test_dangling_edge_rejected` | UT-GRAPH-004 | 通过 |
| `test_duplicate_names_rejected` | UT-GRAPH-005 | 通过 |
| `test_missing_io_nodes_rejected` | UT-GRAPH-006 | 通过 |
| `test_io_edge_rules_rejected` | UT-GRAPH-007 | 通过 |
| `test_io_dtype_mismatch_rejected` | UT-GRAPH-008 | 通过 |
| `test_weight_edge_rules_rejected` | UT-GRAPH-009 | 通过 |
| `test_dtype_and_shape_mismatch_rejected` | UT-GRAPH-010 | 通过 |
| `test_fusion_rule_mismatch_rejected` | UT-GRAPH-011 | 通过 |
| `test_version_mismatch` | UT-GRAPH-016 | 通过 |
| `test_sampling_constraints` | UT-GRAPH-021 | 通过 |
| `test_symbolic_shape_rules` | UT-GRAPH-022 | 通过 |
| `test_quantization_metadata_roundtrip` | UT-GRAPH-023 | 通过 |
| `test_fusion_rule_table` | UT-GRAPH-024 | 通过 |
| `test_to_tensor_desc_rules` | UT-GRAPH-025 | 通过 |
| `test_python_attribute_types` | UT-GRAPH-026 | 通过 |
| `test_runtime_conversion_json_equality` | UT-GRAPH-018 | 通过 |
| `test_unknown_io_binding_and_error_fields` | UT-GRAPH-017 | 通过 |
| `test_large_graph_validation` | UT-GRAPH-019 | 通过：1000 个计算节点、1002 节点/1001 边，校验成功 |
| `test_runtime_headers_have_no_torch_dependency` | UT-GRAPH-027 | 通过：runtime/common 头文件无 `ATen/`、`torch/`、`c10/` include |

实际输出：

```text
......................
----------------------------------------------------------------------
Ran 22 tests in 0.018s

OK
```

### 4.3 Qwen3-0.6B 双端一致性集成测试

文件：`Tests/integration/graph/test_qwen3_cross_language.py`，共 9 个用例：

| 用例 | 设计编号 | 结果 |
| --- | --- | --- |
| `test_counts` | IT-GRAPH-008 | 通过：204 节点 / 570 边 / 311 权重 |
| `test_canonical_json_byte_identical` | UT-GRAPH-018 / IT-GRAPH-001 | 通过：Python 与 C++ Canonical JSON 逐字节相等 |
| `test_python_graph_validates` | IT-GRAPH-001 | 通过：V1~V16 校验无问题 |
| `test_runtime_conversion_matches` | UT-GRAPH-018 | 通过：Graph IR 与 RuntimeGraph 规范 JSON 相等 |
| `test_native_torch_attributes` | UT-GRAPH-026 | 通过：`torch.Tensor`/`torch.dtype` 原生类型 |
| `test_weight_tie_shares_tensor` | IT-GRAPH-001 | 通过：`lm_head` 与 `embedding` 共享 `data_ptr` |
| `test_fusion_rules_hit_every_compute_node` | IT-GRAPH-001 | 通过：层 0/13/27 的全部计算节点规则可查询 |
| `test_golden_snapshot` | IT-GRAPH-008 | 通过：与金样逐字节一致 |
| `test_canonical_json_stable_across_calls` | UT-GRAPH-017 | 通过：重复输出稳定 |

实际输出：

```text
.........
----------------------------------------------------------------------
Ran 9 tests in 0.024s

OK
```

一致性数据：

| 指标 | 数值 |
| --- | --- |
| 节点数 | 204（3 个 Input + embedding + 28×7 层节点 + final_norm + lm_head + sampling + 1 个 Output） |
| 边数 | 570 |
| 权重数 | 311 |
| Canonical JSON 字节数 | 448,754 |
| 金样文件 | `Tests/golden/qwen3_0_6b_graph.json`（448,754 字节） |
| 金样 SHA256 | `162ffba7a78faae321fa7ae6c609bb6df83c0824460a753818122922e01057a2` |

### 4.4 无 LibTorch 依赖审计

头文件扫描（UT-GRAPH-027）：

```text
torch include hits: 0
```

`test_runtime_graph.exe` 的 DLL 导入清单：

```text
api-ms-win-crt-*.dll
KERNEL32.dll
```

不含 `torch.dll`、`torch_cpu.dll`、`c10.dll` 或 `torch_python.dll`。

### 4.5 性能记录（主机，非目标设备）

测量方式：Python `time.perf_counter()`，预热 1 次后重复 7 次，取最小值与中位数。
数据来源：`Build/graph_perf_output.log`。

| 指标 | 最小值 | 中位数 | 设计目标 | 结论 |
| --- | --- | --- | --- | --- |
| Qwen3 C++ 图构建 | 2.936 ms | 3.118 ms | PT-GRAPH-004 < 100 ms | 通过 |
| Qwen3 Python 图构建 | 5.611 ms | 5.983 ms | PT-GRAPH-004 < 100 ms | 通过 |
| Qwen3 图校验 | 0.593 ms | 0.598 ms | PT-GRAPH-001 ≤ 10 ms | 通过 |
| IR→RuntimeGraph JSON 转换 | 2.414 ms | 2.555 ms | PT-GRAPH-005 记录值 | 已记录 |

说明：以上为主机 Python 环境数据，不代表 Android arm64 目标设备；目标设备
指标仍需后续实测。

## 5. 测试中发现并修复的问题

### 5.1 meta 张量存储访问导致进程终止

- 现象：`GraphBuilder.finalize()` 在部分用例中触发
  `Unhandled exception caught in c10/util/AbortHandler.h`，堆栈指向
  `c10::TensorImpl::layout_impl` 与 `DescribeTensor`。
- 定位：符号化后确认 `StorageKey()` 对 meta 张量调用了 `tensor.storage()`；
  meta 张量的 `storage_access_should_throw_` 为真，会抛出
  “不可访问存储”的 c10 异常。
- 修复：`StorageKey()` 改用
  `tensor.unsafeGetTensorImpl()->unsafe_storage()` 并检查 StorageImpl 与
  数据指针，不再触发异常（`Graph/cpp/ir/src/tensor_desc.cc`）。
- 回归：修复后 `-O2` 干净构建下 22 + 9 个 Python 用例与 C++ 单测全部通过。
- 备注：该问题曾在 `-O0` 下偶然不触发，早期误判为 `-O2` ABI 问题；最终
  确认是真实代码缺陷，与优化级别无关。

### 5.2 编译告警清理

- `runtime_graph.cc`：`NDEBUG` 下去掉 `assert` 后 `has_kernel` 未使用；
  改为仅在调试构建中检查。
- `graph.cc`：删除未使用的 `WriteEdgeNames`。
- 回归后构建无告警。

## 6. 与设计文档的偏差

| 项目 | 设计文档 | 实际实现 | 原因与影响 |
| --- | --- | --- | --- |
| C++ 测试框架 | GoogleTest | 独立断言宏 | 本机无 GoogleTest 依赖；断言覆盖等价，后续接入 GoogleTest 不影响生产代码 |
| Python 测试框架（设计未限定） | - | `unittest` | 标准库，无额外依赖 |
| 正式构建入口 | CMake | 已提供 `Graph/CMakeLists.txt`；本机通过 Zig 脚本执行 | 本机无 MSVC/LibTorch CMake 包，CMake 目标未在本机编译；见风险 R1 |

## 7. 未覆盖项与风险

| 编号 | 未覆盖项 | 风险 | 建议 |
| --- | --- | --- | --- |
| R1 | `Graph/CMakeLists.txt` 未在本机实际配置/编译 | 正式 MSVC/LibTorch 构建可能存在链接或导出配置问题 | 在具备 MSVC 2022 + 同版本 LibTorch 的机器上执行第 2 节 CMake 命令并补测 |
| R2 | 真实 Qwen3-0.6B 权重与真实设备推理 | 图结构与真实权重数值未联调 | 模型文件到位后按 IT-GRAPH-001/002/003 补测 |
| R3 | UT-GRAPH-020 模糊测试 | 极端随机输入下的健壮性未验证 | 后续接入随机名称/边/dtype 的 fuzz 测试 |
| R4 | UT-GRAPH-028/029 双表示 Builder 对照 | 仅覆盖 RuntimeGraph 单测与转换一致性，未做同名非法结构的双表示错误码对照 | 增加 RuntimeGraphBuilder 对成环/悬空边的同名负向用例 |
| R5 | IT-GRAPH-002~007、IT-GRAPH-009 | 与 Loader/Executor/Serializer 的端到端集成未执行 | 依赖上下游模块到位后补测 |
| R6 | PT-GRAPH-002/003、ST-GRAPH-001 | 调度开销、元数据内存、1000 次 Bind/Release 未实测 | 增加基准与压力测试 |
| R7 | 性能数据仅为主机 Python 环境 | 不能代表移动端 | 目标设备到位后复测 |

## 8. 结论

Graph 模块代码、Python 绑定、C++ RuntimeGraph、构建脚本、测试与文档已完成。
在本次测试环境中：

- C++ RuntimeGraph 单测、22 个 Python 单元测试、9 个 Qwen3 双端集成测试
  全部通过；
- Qwen3-0.6B 图 Python 侧与 C++ 侧的 Canonical JSON 逐字节一致
  （448,754 字节，SHA256 与金样一致）；
- RuntimeGraph 未引入 LibTorch 依赖；
- 上述未覆盖项与风险已显式记录，不影响本次“功能代码 + 双端一致性”结论。
