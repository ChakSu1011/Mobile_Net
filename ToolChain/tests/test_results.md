# Build 阶段测试结果

## 1. 测试环境

| 项目 | 值 |
| --- | --- |
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.13.3 |
| 解释器 | `C:\Users\18626\AppData\Local\Programs\Python\Python313\python.exe` |
| Graph 扩展 | `Graph/build/python/_graph.pyd`（2026-11-15 构建） |
| 字符编码 | `PYTHONUTF8=1` |
| 外部依赖 | 无 PyTorch / LibTorch / Protobuf；safetensors 由仓库内解析器读取 |
| 测试日期 | 2026-11-15 |

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
Ran 6 tests in 0.513s

OK
adapter qwen3
model qwen3_0_6b
tensors 623
ops 311
constants 310
weights_available 0
weights_total_bytes 0
ir_sha256 ebc16228db005814bbb331ab444d1a1ca501243102868ccb063f6463127c3483
manifest_sha256 0605cd1a754717b0e82bfc214a57d0cdecbe9720f9ff18b6afbdf834315b0b8f
package C:\Users\18626\AppData\Local\Temp\pipeline_czoxd9rs\package4
work_dir C:\Users\18626\AppData\Local\Temp\pipeline_czoxd9rs\work4
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
Ran 12 tests in 5.196s

OK
```

结论：通过。确认 Qwen3-0.6B 的 2 inputs / 310 constants / 311 ops /
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
PYTHONUTF8=1 python -m ToolChain.pipeline \
  --config Models/Qwen3/config.json \
  --weights Models/Qwen3/model.safetensors \
  --target-precision float16
```

实际输出：

```text
adapter qwen3
model qwen3_0_6b
tensors 623
ops 311
constants 310
weights_available 310
weights_total_bytes 1192099840
ir_sha256 ebc16228db005814bbb331ab444d1a1ca501243102868ccb063f6463127c3483
manifest_sha256 dd4c69c96d9a935dd8968771748c825867bf62fb5d295cc1d3fe4024e0d41b50
package D:\Mobile_Net\Models\Qwen3\artifacts\qwen3_0_6b
work_dir D:\Mobile_Net\Models\Qwen3\artifacts\qwen3_0_6b_work
```

产物：

| 路径 | 大小 | 说明 |
| --- | --- | --- |
| `Models/Qwen3/artifacts/qwen3_0_6b/model.mgf` | 384,873 B | 112 B 头 + 规范化 JSON manifest，含权重表 |
| `Models/Qwen3/artifacts/qwen3_0_6b/weights.bin` | 1,192,099,840 B | FP16 权重打包，按 256 B 对齐 |
| `Models/Qwen3/artifacts/qwen3_0_6b/compile_report.json` | 1,204 B | 构建摘要、IR 哈希、显存计划、验证结果 |
| `Models/Qwen3/artifacts/qwen3_0_6b_work/converter/model_graph.json` | 418,551 B | 转换后元数据 |
| `Models/Qwen3/artifacts/qwen3_0_6b_work/converter/conversion_report.json` | 83,637 B | 权重映射与转换摘要 |
| `Models/Qwen3/artifacts/qwen3_0_6b_work/optimizer/model_graph.json` | 443,557 B | 优化后元数据 |
| `Models/Qwen3/artifacts/qwen3_0_6b_work/optimizer/optimize_report.json` | 1,819 B | 精度/形状/显存、权重转换与不变量报告 |
| `Models/Qwen3/artifacts/qwen3_0_6b_work/optimizer/weights_fp16.safetensors` | 1,192,134,752 B | Precision Pass 流式转换的 FP16 权重文件 |

`compile_report.json` 关键字段：

```json
{
  "graph": {
    "activation_tensors": 311,
    "constants": 310,
    "ir_line_count": 632,
    "ir_sha256": "ebc16228db005814bbb331ab444d1a1ca501243102868ccb063f6463127c3483",
    "ops": 311,
    "tensors": 623
  },
  "memory": {
    "alignment": 256,
    "arena_bytes": 81985536,
    "constants_bytes": 1192099840,
    "entries": 311,
    "peak_bytes": 78315520
  },
  "verification": {
    "manifest_sha256": "dd4c69c96d9a935dd8968771748c825867bf62fb5d295cc1d3fe4024e0d41b50",
    "package_reloaded": true
  },
  "weights": {
    "available": 310,
    "sha256": "23b5812e9d7ec965738e4fcaaf81b3f67fbf598cce5fc5054318b0d2a47dd629",
    "total_bytes": 1192099840,
    "unavailable": 0
  }
}
```

权重转换统计（`optimize_report.json`）：

```json
{
  "converted_count": 310,
  "copied_count": 0,
  "output_name": "weights_fp16.safetensors",
  "source_count": 310,
  "target_precision": "float16"
}
```

结论：通过。Precision Pass 将 310 个 BF16 权重流式转换为 FP16 并写出
`weights_fp16.safetensors`，Serializer 打包 310 个 FP16 权重到
`weights.bin`；模型包生成后已重新加载并通过 manifest SHA-256 与重建图
IR SHA-256 自检。另抽样验证：`model.layers.0.self_attn.q_proj.weight`
的源 BF16→FP16 结果与转换文件、包内字节三者完全一致，包内 SHA-256 与
manifest 一致；`embed_tokens.weight` 前 1 MiB 同样一致。

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
| `ToolChain.optimizer.test_optimizer` | 21 | UT-OPT-001 ~ 019，含权重转换 |
| `ToolChain.serializer.test_serializer` | 13 | UT-SER-001 ~ 012 |
| `ToolChain.test_pipeline` | 6 | 端到端、确定性、复用开关、CLI |
| `Models/Qwen3/test_qwen3_graph.py` | 12 | 组图结构、权重形状、配置、IR 稳定性 |
| `graph_test.exe` | C++ 全量 | Graph/SortOp/I/O/dump_ir |

## 4. 未覆盖项与风险

1. **权重数值未验证**：本地已使用 `Models/Qwen3/model.safetensors`
   生成 310 个 FP16 权重并打包到 `weights.bin`，但只对
   `q_proj.weight` 全量与 `embed_tokens.weight` 前 1 MiB 做了字节级
   比对，未对 310 个张量逐一核对，也未与 Transformers 做数值基线对比
   或执行运行时推理。`Models/` 仅本地保留，不提交 Git。
2. **FP16 舍入损失未量化**：BF16→FP16 采用 round-to-nearest-even，
   BF16 的 8 位指数范围收窄到 FP16 的 5 位；当前 Qwen3 权重未触发
   `65504` 溢出检查，但转换后的数值误差与对生成质量的影响尚未评估。
3. **溢出策略为硬失败**：有限值绝对值超过 `65504` 时 Precision Pass
   直接抛 `BuildError`，不做 clamp 或缩放；接入其他模型时可能因个别
   极端权重中断构建。
4. **未执行运行时推理**：Build 阶段不创建 OpEnv/Kernel，不验证端侧
   数值与性能；`OpEnv::Forward()` 首期只有 `MegaAttentionOpEnv` 骨架。
5. **显存计划为静态规划值**：`arena_bytes=81,985,536`、`peak_bytes=
   78,315,520` 基于 `max_num_tokens=256`、`B=1` 的规划上界，不是设备
   实测峰值。
6. **未做量化**：Precision Pass 已定位为后续量化统一入口，INT8/INT4
   仅有设计扩展点，无实现与测试。
7. **测试环境为 Windows 主机**：未在 Android arm64 设备或 OpenCL
   后端上执行加载与推理测试。
