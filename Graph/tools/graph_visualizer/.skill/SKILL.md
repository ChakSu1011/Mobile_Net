---
name: graph_visualizer
description: 把序列化后的 model.mgf 渲染为自上而下的静态 SVG 流程图，节点只显示融合算子名称/类型/类别/精度/散算子类别，输入、输出、激活和参数都以平行四边形边显示精度与 shape，并折叠重复 Layer。
---

# Graph 静态图可视化

## 适用场景

- 已有 Serializer 生成的 `model.mgf`，需要人工确认融合算子图。
- 需要查看每个融合算子的类型、类别、精度和散算子类别。
- 需要确认输入、输出、激活和参数平行四边形的精度、shape 与串接顺序。
- 需要折叠重复的 Decoder Layer，并发现与其他层结构不同的差异层。

## 不适用场景

- 需要查看权重数值、统计值或采样值。
- 需要完整展开所有 Layer（本工具只画代表性子图和差异层）。
- 需要运行时性能、显存实测或数值推理。
- 需要编辑图、优化图或导出 ONNX/Protobuf。
- 需要 HTML 交互视图。

## 输入

1. `model.mgf` 文件或包含它的包目录。
2. 对应的 `ModelGraph` manifest；工具不读取 `weights.bin` 内容。
3. 可选 `--output` 输出路径。

## 输出

- 一个自包含 SVG 文件。
- 终端摘要：节点数、平行四边形边数、连接数、警告信息。
- 默认输出路径：`Models/<模型目录>/artifacts/<model-name>_graph.svg`。

## 标准流程

```bash
PYTHONUTF8=1 python Graph/tools/visualize_graph.py \
  --package Models/Qwen3/artifacts/qwen3_0_6b/model.mgf
```

指定输出：

```bash
PYTHONUTF8=1 python Graph/tools/visualize_graph.py \
  --package Models/Qwen3/artifacts/qwen3_0_6b/model.mgf \
  --output Models/Qwen3/artifacts/qwen3_0_6b_graph.svg
```

1. 加载并校验 `model.mgf` 头部与 manifest 哈希。
2. 从 manifest 构造 `ModelGraph`。
3. 识别 `layer.<n>.*`，按结构签名合并连续相同层。
4. 构建显示节点：融合算子、输入/输出/激活/参数 tensor 平行四边形。
5. 计算自上而下 rank、节点/平行四边形尺寸和 Layer 背景框。
6. 渲染 SVG 并输出警告。

## 核心规则

- 节点为融合算子，只显示名称、`OpType`、类别、精度和散算子类别；
  不显示 `attrs`。
- 输入、输出、激活、参数全部绘制为串在流程中的平行四边形。
- 平行四边形内只显示精度和 shape；参数平行四边形使用虚线边框。
- 输入/输出平行四边形外部显示 tensor 名；参数平行四边形不显示权重名称、数值或哈希。
- shape 元素数 ≤ 1 的常量不绘制参数平行四边形。
- 连续相同 Layer 只画一个代表性子图，标注 `×N (layer.a–layer.b)`。
- 与基准层签名不同的 Layer 单独绘制，红色边框并标记“差异”。
- 不提供完整展开模式。
- 相同输入和选项生成字节一致的 SVG。

## 检查清单

- [ ] SVG 顶部有模型名、manifest/IR 哈希和图例。
- [ ] 节点只显示名称、类型、类别、精度和散算子类别，无 attrs。
- [ ] 输入/输出/激活/参数都是平行四边形边。
- [ ] 平行四边形只显示精度和 shape；参数平行四边形为虚线。
- [ ] 重复 Layer 标注 `×N`，没有重复绘制 N 份。
- [ ] 差异 Layer 单独显示并标记。
- [ ] 输出位于模型 artifacts 目录。
- [ ] 两次运行生成的 SVG 字节一致。

## 常见陷阱

1. **把 `Graph/build` 当成本工具目录**：工具在
   `Graph/tools/graph_visualizer/`，构建输出在 `Graph/build/`。
2. **读取 `weights.bin`**：可视化不需要权重内容，参数平行四边形只用 manifest
   的 shape 和 precision。
3. **把第一层的 embedding 输入误判为差异**：Layer 签名必须把外部
   activation 归一化为角色占位，而不是比较张量全名。
4. **用 op 数量判断层是否相同**：必须比较归一化名称、类型、属性、连接
   和对外角色。
5. **参数平行四边形显示权重名称或数值**：当前规则只显示精度和 shape，避免泄露
   权重内容并保持图面简洁。
6. **输出到 ToolChain 或仓库根目录**：产物应写入
   `Models/<模型目录>/artifacts/`。
