---
name: graph_memory
description: 从 model.mgf 的 memory_plan 生成 Activation 与参数/权重显存排布 SVG，显示 Shape、Pitch、offset、大小、复用和重复 Layer 折叠，用于人工确认显存布局。
---

# Graph 显存排布可视化

## 适用场景

- 已有 Serializer 生成的 `model.mgf`，需要人工确认 Activation arena。
- 需要查看每个算子输出 Activation 的 Shape、Pitch、offset、size。
- 需要确认哪些 Activation 复用同一段显存、生命周期是否重叠。
- 需要查看权重属于哪个算子、Shape、Pitch、size 和相对 offset。
- 需要折叠重复 Decoder Layer，避免 28 层重复绘制。

## 不适用场景

- 真实设备显存分配或性能测量。
- 读取权重数值、数值推理或 kernel profiling。
- 修改图或显存计划。
- 完整展开 28 层的逐层审阅。

## 输入

1. `model.mgf` 文件或包目录。
2. manifest 中的 `graph` 与 `optimization.memory_plan`。
3. `weights.entries` 的实际 offset（可选）。

## 输出

- 一个自包含 SVG，两个面板：
  1. Activation Memory Arena；
  2. Parameter / Weight Memory。
- 默认路径：`Models/<模型目录>/artifacts/<model-name>_memory.svg`。

## 标准流程

```bash
PYTHONUTF8=1 python Graph/tools/graph_memory/visualize_memory.py \
  --package Models/Qwen3/artifacts/qwen3_0_6b/model.mgf
```

指定输出：

```bash
PYTHONUTF8=1 python Graph/tools/graph_memory/visualize_memory.py \
  --package Models/Qwen3/artifacts/qwen3_0_6b/model.mgf \
  --output Models/Qwen3/artifacts/qwen3_0_6b_memory.svg
```

1. 加载并校验 `model.mgf`。
2. 读取 `memory_plan` 的 entries 和 summary。
3. 构造 Activation 区块：owner、Shape、Pitch、offset、size、lifetime。
4. 构造权重区块：owner、Shape、Pitch、offset、size。
5. 按层签名折叠重复 Layer，标记差异层。
6. 布局两个地址轴和明细行，渲染 SVG。

## 核心规则

- Activation Shape 来自 `ModelGraph`；Pitch 为空时按 row-major contiguous
  推导，并标记 `(contiguous)`。
- Activation offset/size/lifetime 来自 `memory_plan.entries`。
- 同一 offset/size 的 Activation 显示 `REUSE` 和斜线纹理。
- 权重 owner 来自消费该常量的算子；tied embedding 显示多个 owner。
- 权重无实际 offset 时按常量顺序、256 B 对齐生成 planned layout，
  并标注 `planned`。
- 重复 Layer 只画代表层，标注 `×N`；差异层单独显示。
- 不读取 `weights.bin` 内容。

## 检查清单

- [ ] Activation 显示 owner、Shape、Pitch、offset、size。
- [ ] 28 层复用条与代表层布局一致。
- [ ] REUSE 区块共享 offset。
- [ ] 权重显示 owner、参数名、Shape、Pitch、size、offset。
- [ ] planned layout 有明确标注。
- [ ] 重复 Layer 标注 ×N，差异层单独显示。
- [ ] 两个面板、图例、哈希完整。
- [ ] 两次运行 SVG 字节一致。

## 常见陷阱

1. **把 Activation 的空 pitch 当错误**：ShapePass 不保留 activation pitch，
   按 contiguous shape 推导并标注。
2. **比较绝对 offset 判断层差异**：分配器状态会让相同层的 offset 漂移；
   只比较 shape/pitch/size/参数签名。
3. **把 planned offset 当实际权重地址**：metadata-only 包没有物理
   offset，图中必须标注 planned。
4. **把常量算进 Activation arena**：MemoryPass 不把常量放入 arena；
   权重应在单独面板。
5. **绘制 28 层全部 Activation**：默认代表层 + 28 层复用条。
6. **读取 weights.bin**：本工具只读 manifest 元数据。
