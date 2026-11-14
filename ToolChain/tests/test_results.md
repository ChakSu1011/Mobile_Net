# Build 阶段测试结果

## 1. 测试环境

| 项目 | 值 |
| --- | --- |
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.13.3 |
| 解释器 | `C:\Users\18626\AppData\Local\Programs\Python\Python313\python.exe` |
| Graph 扩展 | `Graph/build/python/_graph.pyd`（2026-11-14 构建） |
| 字符编码 | `PYTHONUTF8=1` |
| 外部依赖 | 无 PyTorch / LibTorch / Protobuf；safetensors 由仓库内解析器读取 |
| 测试日期 | 2026-11-14 |

## 2. 测试命令与结果

### 2.1 模块级测试结果

| 模块 | 结果文档 |
|------|----------|
| Converter | [converter/tests/test_results.md](../converter/tests/test_results.md) |
| Optimizer | [optimizer/tests/test_results.md](../optimizer/tests/test_results.md) |
| Serializer | [serializer/tests/test_results.md](../serializer/tests/test_results.md) |

以下为阶段级测试。

### 2.2 Build 端到端流水线测试

命令：

```bash
PYTHONUTF8=1 python -m unittest ToolChain.test_pipeline -v
```

实际输出（末尾）：

```text
Ran 6 tests in 0.562s

OK
adapter qwen3
model qwen3_0_6b
tensors 735
ops 423
constants 310
weights_available 0
weights_total_bytes 0
ir_sha256 3578cbfd1ed37776f8f6384c5b84d5c1ea24589bac5eed39a381376d0a94da20
manifest_sha256 5f62384fe1fa14b0002b616c51309e9bf7e2b742ad41303d32ef4dd6a203f1df
package C:\Users\18626\AppData\Local\Temp\pipeline_ucncnwv7\package4
work_dir C:\Users\18626\AppData\Local\Temp\pipeline_ucncnwv7\work4
```

（临时目录名由 `tempfile.mkdtemp` 生成，每次运行不同。）

结论：通过。覆盖包计数、目录布局（仅 `model.mgf` 与
`compile_report.json`）、work 目录中间产物、字节级确定性、
关闭复用后 arena 增大、CLI 入口返回码。

### 2.3 Qwen3 组图回归测试

命令：

```bash
PYTHONUTF8=1 python -m unittest discover -s Models/Qwen3 -p 'test_*.py' -v
```

实际输出（末尾）：

```text
Ran 12 tests in 5.183s

OK
```

结论：通过。确认 Qwen3-0.6B 的 2 inputs / 310 constants / 423 ops /
1 output、310 个常量、tied embedding、Transformer 权重形状、
596,049,920 参数与 1,192,099,840 字节 BF16 常量，以及 IR 稳定性。

### 2.4 Graph C++ 测试

命令：

```bash
cd Graph/build && ./graph_test.exe
```

实际输出：

```text
graph_test: ALL TESTS PASSED
```

结论：通过。确认图构造、SortOp、常量为依赖叶子、I/O op、
dump_ir 等 Graph 基础行为未回归。

### 2.5 Qwen3-0.6B 模型包生成

命令：

```bash
PYTHONUTF8=1 python -m ToolChain.pipeline
```

实际输出：

```text
adapter qwen3
model qwen3_0_6b
tensors 735
ops 423
constants 310
weights_available 0
weights_total_bytes 0
ir_sha256 3578cbfd1ed37776f8f6384c5b84d5c1ea24589bac5eed39a381376d0a94da20
manifest_sha256 5f62384fe1fa14b0002b616c51309e9bf7e2b742ad41303d32ef4dd6a203f1df
package D:\Mobile_Net\Models\Qwen3\artifacts\qwen3_0_6b
work_dir D:\Mobile_Net\Models\Qwen3\artifacts\qwen3_0_6b_work
```

产物：

| 路径 | 大小 | 说明 |
| --- | --- | --- |
| `Models/Qwen3/artifacts/qwen3_0_6b/model.mgf` | 430,153 B | 112 B 头 + 规范化 JSON manifest，无权重区 |
| `Models/Qwen3/artifacts/qwen3_0_6b/compile_report.json` | 992 B | 构建摘要、IR 哈希、显存计划、验证结果 |
| `Models/Qwen3/artifacts/qwen3_0_6b_work/converter/model_graph.json` | 480,582 B | 转换后元数据 |
| `Models/Qwen3/artifacts/qwen3_0_6b_work/converter/conversion_report.json` | 83,948 B | 权重映射与转换摘要 |
| `Models/Qwen3/artifacts/qwen3_0_6b_work/optimizer/model_graph.json` | 518,748 B | 优化后元数据 |
| `Models/Qwen3/artifacts/qwen3_0_6b_work/optimizer/optimize_report.json` | 1,611 B | 精度/形状/显存与不变量报告 |

`compile_report.json` 关键字段：

```json
{
  "graph": {
    "activation_tensors": 423,
    "constants": 310,
    "ir_line_count": 744,
    "ir_sha256": "3578cbfd1ed37776f8f6384c5b84d5c1ea24589bac5eed39a381376d0a94da20",
    "ops": 423,
    "tensors": 735
  },
  "memory": {
    "alignment": 256,
    "arena_bytes": 81985536,
    "constants_bytes": 1192099840,
    "entries": 423,
    "peak_bytes": 78315520
  },
  "verification": {
    "manifest_sha256": "5f62384fe1fa14b0002b616c51309e9bf7e2b742ad41303d32ef4dd6a203f1df",
    "package_reloaded": true
  },
  "weights": {
    "available": 0,
    "sha256": null,
    "total_bytes": 0,
    "unavailable": 310
  }
}
```

结论：通过。模型包生成后已重新加载并通过 manifest SHA-256 与重建图
IR SHA-256 自检。

### 2.6 代码规范检查

```bash
PYTHONUTF8=1 python -m compileall -q ToolChain Models/Qwen3
```

实际输出：无错误。另以脚本检查 ToolChain 与 Models/Qwen3 下全部 Python 文件，
无超过 80 字符的行；AST 扫描未发现除 `__future__` 与 `__init__` 再导出
之外的未使用导入。

## 3. 用例覆盖清单

| 套件 | 用例数 | 主要覆盖 |
| --- | --- | --- |
| `ToolChain.converter.test_converter` | 13 | UT-CONV-001 ~ 012 |
| `ToolChain.optimizer.test_optimizer` | 18 | UT-OPT-001 ~ 016 |
| `ToolChain.serializer.test_serializer` | 13 | UT-SER-001 ~ 012 |
| `ToolChain.test_pipeline` | 6 | 端到端、确定性、复用开关、CLI |
| `Models/Qwen3/test_qwen3_graph.py` | 12 | 组图结构、权重形状、配置、IR 稳定性 |
| `graph_test.exe` | C++ 全量 | Graph/SortOp/I/O/dump_ir |

## 4. 未覆盖项与风险

1. **无真实 Qwen3-0.6B 权重**：仓库未包含 safetensors 权重，生成的
   `model.mgf` 为 metadata-only，310 个权重条目均为
   `available=false`、`reason="weights not supplied"`。权重打包路径仅由
   合成 safetensors 测试覆盖，未做真实模型数值验证。
2. **权重精度匹配约束**：Serializer 原样打包权重字节，不做 dtype 转换。
   默认优化目标为 FP16；若直接提供官方 BF16 safetensors，权重会因
   `dtype mismatch` 全部不可用。此时应使用
   `--target-precision bfloat16`，使图精度与源权重一致。
3. **未执行运行时推理**：Build 阶段不创建 OpEnv/Kernel，不验证端侧
   数值与性能；`OpEnv::Forward()` 首期只有 `MegaAttentionOpEnv` 骨架。
4. **显存计划为静态规划值**：`arena_bytes=81,985,536`、`peak_bytes=
   78,315,520` 基于 `max_num_tokens=256`、`B=1` 的规划上界，不是设备
   实测峰值。
5. **未做量化**：Precision Pass 当前只做浮点精度选择，INT8/INT4
   仅有扩展点，无实现与测试。
6. **测试环境为 Windows 主机**：未在 Android arm64 设备或 OpenCL
   后端上执行加载与推理测试。
