# Graph 静态图可视化工具测试结果

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-11-14 |
| 测试对象 | `graph_visualizer` |
| 设计文档 | [graph_visualizer_design.md](../.doc/graph_visualizer_design.md) |
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
PYTHONUTF8=1 python Graph/tools/graph_visualizer/tests/test_visualizer.py -v
```

实际输出（末尾）：

```text
Ran 11 tests in 0.082s

OK
```

退出码：0。

覆盖点：

| 编号 | 用例 | 结果 |
|------|------|------|
| UT-VIS-001 | Qwen3 包计数（2/310/423/1/735） | 通过 |
| UT-VIS-002 | 视图模型包含 embedding、Layer ×28、final norm、LM Head | 通过 |
| UT-VIS-003 | `linear_swi_glu` 融合组成 | 通过 |
| UT-VIS-004 | 未知 OpType 不崩溃 | 通过 |
| UT-VIS-005 | 参数边存在且只有精度/shape 元数据 | 通过 |
| UT-VIS-006 | shape 使用 `×` 连接 | 通过 |
| UT-VIS-007 | 三层同构折叠为 ×3；Qwen3 折叠为 ×28 | 通过 |
| UT-VIS-008 | 三层中一层不同，差异层单独标记 | 通过 |
| UT-VIS-009 | 长属性列表截断 | 通过 |
| UT-VIS-010 | 两次渲染 SVG 字节一致 | 通过 |

## 3. Qwen3 静态图生成

命令：

```bash
PYTHONUTF8=1 python Graph/tools/visualize_graph.py \
  --package Models/Qwen3/artifacts/qwen3_0_6b/model.mgf
```

实际输出：

```text
wrote Models\Qwen3\artifacts\qwen3_0_6b_graph.svg
nodes 21
edges 39
```

产物：

| 项目 | 值 |
|------|-----|
| 路径 | `Models/Qwen3/artifacts/qwen3_0_6b_graph.svg` |
| 大小 | 33 KB |
| SHA-256 | `6a7ec8c9ffad090b86e3b1850e86c21e9ebb73c0e368900ea3530e8bd4227a69` |
| 画布 | 1570 × 3045 |
| 节点数 | 21 |
| 边数 | 39 |
| Layer 折叠 | `DecoderLayer ×28 (layer.0–layer.27)` |
| 参数边 | 14 条，仅显示精度与 shape |

图中包含：

- 2 个输入：`input.input_ids`、`input.position_ids`；
- embedding；
- 一个 Decoder Layer 代表子图，包含 15 个融合算子；
- final norm、LM Head、1 个输出；
- 14 条参数边，标签仅为 `[fp16, shape]`；
- 无权重数值、统计值或哈希。

## 4. 人工验收

待用户打开 SVG 后确认：

- [ ] 节点是融合算子名称，显示散算子组成、类别、精度、属性；
- [ ] 激活/输入/输出边为实线箭头；
- [ ] 参数边为虚线，仅显示精度和 shape；
- [ ] 重复 Layer 只画代表性子图并标注 `×28`；
- [ ] 自上而下布局，箭头方向清晰；
- [ ] 图例、标题、manifest/IR 哈希完整。

## 5. 未覆盖项与风险

1. **人工视觉验收未完成**：SVG 已生成，但最终可读性需用户确认。
2. **非 `layer.<n>` 命名的重复块**：首期只折叠 Layer 模式，其他重复结构
   不自动折叠。
3. **复杂布局**：纯 Python 分层布局无法保证零交叉；当前 Qwen3 视图已
   人工检查，极端模型可能需要调整。
4. **SVG 字体差异**：使用通用无衬线字体栈，不同浏览器的中文字体宽度可能
   略有差异。
5. **不验证权重内容**：工具不读取 `weights.bin`，参数边只验证 manifest
   的 shape 与 precision。
6. **不提供完整展开视图**：按需求只显示代表层和差异层。
