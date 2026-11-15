# Graph 显存排布可视化工具测试结果

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-11-15 |
| 测试对象 | `graph_memory` |
| 设计文档 | [graph_memory_design.md](../.doc/graph_memory_design.md) |
| skill | [SKILL.md](../.skill/SKILL.md) |

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.13.3 |
| 解释器 | `C:\Users\18626\AppData\Local\Programs\Python\Python313\python.exe` |
| 输入模型包 | `Models/Qwen3/artifacts/qwen3_0_6b/model.mgf` |
| 外部依赖 | 无；仅使用 Python 标准库与仓库内 ToolChain 代码 |

## 2. 单元与集成测试

命令：

```bash
PYTHONUTF8=1 python Graph/tools/graph_memory/tests/test_memory_visualizer.py -v
```

实际输出（末尾）：

```text
Ran 8 tests in 0.710s

OK
```

退出码：0。

覆盖点：

| 编号 | 用例 | 结果 |
|------|------|------|
| UT-MEM-001 | Qwen3 Activation/权重计数与 summary | 通过 |
| UT-MEM-002 | Activation owner/shape/pitch/offset/size | 通过 |
| UT-MEM-003 | 权重 owner/shape/pitch/实际 offset | 通过 |
| UT-MEM-004 | tied embedding 多 owner | 通过 |
| UT-MEM-006 | Qwen3 Layer 折叠 ×28 | 通过 |
| UT-MEM-007 | 合成层签名差异标记 | 通过 |
| UT-MEM-008 | 代表层明细行数 | 通过 |
| UT-MEM-009 | 两次渲染 SVG 字节一致 | 通过 |

## 3. Qwen3 显存图生成

命令：

```bash
PYTHONUTF8=1 python Graph/tools/graph_memory/visualize_memory.py \
  --package Models/Qwen3/artifacts/qwen3_0_6b/model.mgf
```

实际输出：

```text
wrote Models\Qwen3\artifacts\qwen3_0_6b_memory.svg
activations 312
weights 310
planned_weights False
```

产物：

| 项目 | 值 |
|------|-----|
| 路径 | `Models/Qwen3/artifacts/qwen3_0_6b_memory.svg` |
| 大小 | 74,556 B |
| SHA-256 | `99e12d654a81a62541d75befe2af482f82913ee5d8900f896b5dd1e779050095` |
| 画布 | 1600 × 2796 |
| Activation arena | 78,317,568 B |
| Activation peak | 78,315,520 B |
| Alignment | 256 B |
| Activation entries | 312 |
| 常量逻辑字节 | 1,192,099,840 B |
| 权重布局 | 实际 `weights.bin` offset |
| Activation 明细行 | 15（4 非层 + 11 代表层） |
| 权重明细行 | 13（2 非层 + 11 代表层） |
| Layer 复用条 | 28 行 |
| 权重分组 | 3（LayerWeights ×28、embedding、final_norm） |

图中包含：

- Activation 面板：arena 地址轴、28 层复用条、代表层明细；
- 每个 Activation 显示 owner、tensor、Shape、Pitch、offset、size、
  lifespan 和复用标记；
- 权重面板：权重 arena、`LayerWeights ×28`、embedding、final_norm
  的总览和明细；
- 每个权重显示 owner、参数名、Shape、Pitch、size、offset；
- tied `embedding.weight` 显示 `embedding, lm_head` 两个 owner；
- 模型包包含真实权重，权重 offset 来自 `weights.bin`（256 B 对齐）；
- Sample 的 `sample.out [1×256] int64` 作为非层 Activation 显示。

## 4. 人工验收

待用户打开 SVG 后确认：

- [ ] Activation 显示 owner、Shape、Pitch、offset、size。
- [ ] 28 层复用条与代表层布局一致。
- [ ] REUSE 区块共享 offset。
- [ ] 权重显示 owner、参数名、Shape、Pitch、size、offset。
- [ ] Sample 的 `sample.out` 显示在非层 Activation 中。
- [ ] 重复 Layer 只画代表层并标注 ×28。
- [ ] 两个面板、图例、哈希完整。

## 5. 未覆盖项与风险

1. **人工视觉验收未完成**：SVG 已生成，可读性需用户确认。
2. **权重 offset 未运行时校验**：当前使用 `weights.bin` 的打包 offset，
   未验证设备端实际地址与 256 B 对齐后的映射。
3. **逻辑地址**：Activation offset 来自 MemoryPass 的逻辑规划，
   不是设备实测地址。
4. **复杂布局**：区块较多时标签可能拥挤；当前 Qwen3 视图已人工检查。
5. **不验证数值**：不读取权重内容、不执行 kernel。
6. **仅 Windows 验证**：未在 Linux/macOS 或目标设备验证。
