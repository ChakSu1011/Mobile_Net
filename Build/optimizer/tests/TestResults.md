# Optimizer 模块测试结果

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-11-14 |
| 测试对象 | `Build.optimizer` |
| 设计文档 | [optimizer_Design.md](../.doc/optimizer_Design.md) |
| 完整阶段测试 | [Build/tests/TestResults.md](../../tests/TestResults.md) |

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.13.3 |
| 解释器 | `C:\Users\18626\AppData\Local\Programs\Python\Python313\python.exe` |
| Graph 扩展 | `Build/graph/python/_graph.pyd`（2026-11-14 构建） |
| 字符编码 | `PYTHONUTF8=1` |
| 外部依赖 | 无 PyTorch / LibTorch / Protobuf |

## 2. 测试命令与结果

```bash
PYTHONUTF8=1 python -m unittest Build.optimizer.test_optimizer -v
```

实际输出（末尾）：

```text
Ran 18 tests in 0.172s

OK
```

结论：通过，退出码 0。

## 3. 覆盖范围

- BF16 → FP16 等宽精度转换、`target_precision=keep`、非法精度转换报错；
- linear、gather、RMSNorm、Add、GQA 的形状契约；
- 423 个 activation 张量的静态形状完整性；
- 生命周期计算与 256 B 对齐；
- 复用区间无重叠、关闭复用后 arena 增大；
- 输入/常量不进入 arena，输出不可复用；
- 优化报告字段与确定性。

## 4. 未覆盖项与风险

1. **显存计划为静态规划值**：`arena_bytes`、`peak_bytes` 基于
   `B=1`、`max_num_tokens=256` 的规划上界，不是设备实测峰值。
2. **未做量化**：Precision Pass 只做浮点精度选择，INT8/INT4 仅有扩展点。
3. **无真实权重**：优化对象是 Converter 记录的元数据，未验证真实权重
   转换后的数值一致性。
4. **不分配真实显存**：只输出逻辑偏移计划，未接入 Manager/Device。
5. **仅 Windows 主机验证**：未在目标设备上执行内存规划与 kernel 测试。
