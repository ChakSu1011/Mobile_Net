# Optimizer 模块测试结果

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-11-15 |
| 测试对象 | `ToolChain.optimizer` |
| 设计文档 | [optimizer_design.md](../.doc/optimizer_design.md) |
| 完整阶段测试 | [ToolChain/tests/test_results.md](../../tests/test_results.md) |

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.13.3 |
| 解释器 | `C:\Users\18626\AppData\Local\Programs\Python\Python313\python.exe` |
| Graph 扩展 | `Graph/build/python/_graph.pyd`（2026-11-15 构建） |
| 字符编码 | `PYTHONUTF8=1` |
| 外部依赖 | 无 PyTorch / LibTorch / Protobuf |

## 2. 测试命令与结果

```bash
PYTHONUTF8=1 python -m unittest ToolChain.optimizer.test_optimizer -v
```

实际输出（末尾）：

```text
Ran 22 tests in 0.114s

OK
```

结论：通过，退出码 0。

### 2.1 真实权重转换验证

命令：

```bash
PYTHONUTF8=1 python -m ToolChain.pipeline \
  --config Models/Qwen3/config.json \
  --weights Models/Qwen3/model.safetensors \
  --target-precision float16
```

实际输出（末尾）：

```text
weights_available 310
weights_total_bytes 1192099840
ir_sha256 4326aaa8c2505f020184f721df6a2ac947fc2d1d4959edd5d78ddda53f95adaf
manifest_sha256 9c817f7a601424fd1a10155edae527a02d8a52e98e2d7a8233b5f6ece9d06c9a
```

`optimize_report.json` 的 `weight_conversion`：

```json
{
  "converted_count": 310,
  "copied_count": 0,
  "output_name": "weights_fp16.safetensors",
  "source_count": 310,
  "target_precision": "float16"
}
```

补充校验（同一环境执行）：

```text
dtype BF16 shape (151936, 1024)
numpy==python True
first fp16 values (-0.00927734375, 0.03369140625, -0.07470703125, -0.01806640625)
1.0 -> 003c 1.0
converted==expected True
packed==converted True
packed_sha_matches True
embed sample match True
```

结论：通过。numpy 加速路径与纯 Python 路径对真实 BF16 数据输出一致；
`model.layers.0.self_attn.q_proj.weight` 的源转换结果、转换文件字节与
模型包内字节完全一致，包内 SHA-256 与 manifest 一致；embedding 前
1 MiB 抽样一致。

## 3. 覆盖范围

- BF16 → FP16 等宽精度转换、`target_precision=keep`、非法精度转换报错；
- BF16 权重字节转换、转换文件 dtype/数值、溢出报错、BF16 目标跳过转换；
- linear、gather、RMSNorm、Add、GQA 的形状契约；
- 312 个 activation 张量的静态形状完整性；
- Sample 输出 `[B,S]`、精度 `int64`，整数输出精度属性不被改写；
- 生命周期计算与 256 B 对齐；
- 复用区间无重叠、关闭复用后 arena 增大；
- 输入/常量不进入 arena，输出不可复用；
- 优化报告字段与确定性。

## 4. 未覆盖项与风险

1. **显存计划为静态规划值**：`arena_bytes=78,317,568`、
   `peak_bytes=78,315,520` 基于
   `B=1`、`max_num_tokens=256` 的规划上界，不是设备实测峰值。
2. **未做量化**：Precision Pass 已定位为量化统一入口，INT8/INT4 仅有
   扩展点，无实现与测试。
3. **无全量数值基线**：仅对 1 个完整张量与 1 个 1 MiB 抽样做字节级
   比对，未对 310 个张量逐一核对，也未与 Transformers 做数值基线对比；
   FP16 舍入对生成质量的影响未评估。
4. **溢出策略为硬失败**：有限值绝对值超过 `65504` 时抛 `BuildError`，
   不 clamp、不缩放；接入其他模型时可能因个别极端权重中断构建。
5. **不分配真实显存**：只输出逻辑偏移计划，未接入 Manager/Device。
6. **仅 Windows 主机验证**：未在目标设备上执行内存规划与 kernel 测试。
