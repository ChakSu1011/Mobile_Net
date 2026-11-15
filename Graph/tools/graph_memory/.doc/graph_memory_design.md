# Graph 显存排布可视化工具设计文档

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 日期 | 2026-11-15 |
| 文档状态 | 已确认，随代码同步 |
| 模块目录 | `Graph/tools/graph_memory/` |
| 输入 | `model.mgf` |
| 输出 | 模型 artifacts 目录下的 `*_memory.svg` |
| 配套 skill | `Graph/tools/graph_memory/.skill/SKILL.md` |

## 已确认决策

| 决策 | 结论 |
|------|------|
| 展示内容 | Activation Shape、Pitch、offset、size、复用；权重 owner、Shape、Pitch、size、相对 offset |
| Activation 粒度 | 代表层 + 28 层复用条 + 非层结构 |
| 输出布局 | 一个 SVG，上下两个面板：Activation / 权重 |
| 重复 Layer | 与拓扑图一致，代表层 + `×N`，布局不同的层单独显示 |
| 输出目录 | `Models/<模型目录>/artifacts/` |
| 输出文件 | `<model-name>_memory.svg` |

## 1. 背景与目标

### 1.1 背景

`model.mgf` manifest 中已有 `optimization.memory_plan`：

- `alignment`、`arena_bytes`、`peak_bytes`、`constants_bytes`；
- 每个 Activation 的 `offset`、`size`、`birth`、`last_use`、`reusable`；
- 每个权重的 shape、dtype、byte_size、offset（有真实权重时）。

但 JSON 无法直观看出 Activation 在显存池中的位置、复用关系，以及权重
属于哪个算子、相对位置如何。工具从 `model.mgf` 生成 SVG 显存排布图。

### 1.2 目标

1. Activation 面板显示：
   - 所属算子、输出 tensor；
   - Shape 与 Pitch；
   - 显存池 offset、size；
   - 生命周期与复用关系；
   - arena 总量、峰值、对齐。
2. 权重面板显示：
   - 所属算子、参数名；
   - Shape、Pitch、size；
   - 相对 offset 与顺序；
   - 常量区总量。
3. 重复 Layer 折叠为代表层并标注 `×N`；布局不同的层单独显示。
4. 输出自包含 SVG，离线可打开，字节确定。

### 1.3 非目标

- 不做真实显存分配或设备侧测量。
- 不读取权重数值。
- 不做性能/带宽分析。
- 不修改 `model.mgf` 或显存计划。
- 不提供完整 28 层展开视图（调试开关除外）。

### 1.4 交付物与位置

| 交付物 | 位置 |
|--------|------|
| 设计文档 | `Graph/tools/graph_memory/.doc/graph_memory_design.md` |
| 工具代码 | `Graph/tools/graph_memory/` |
| CLI | `Graph/tools/graph_memory/visualize_memory.py` |
| 测试 | `Graph/tools/graph_memory/tests/test_memory_visualizer.py` |
| 测试结果 | `Graph/tools/graph_memory/tests/test_results.md` |
| skill | `Graph/tools/graph_memory/.skill/SKILL.md` |
| 示例产物 | `Models/Qwen3/artifacts/qwen3_0_6b_memory.svg` |

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收方式 |
|------|------|----------|
| FR-MEM-01 | 读取并校验 `model.mgf`，解析 graph 与 memory_plan | 损坏包报错 |
| FR-MEM-02 | Activation 显示 owner op、tensor、Shape、Pitch、offset、size | 与 manifest 一致 |
| FR-MEM-03 | 地址轴展示整个 arena，区块宽度与字节数成比例 | arena 总宽对应 `arena_bytes` |
| FR-MEM-04 | 展示复用：共享 offset 的区块可辨认，生命周期重叠不误判 | 与 memory plan 一致 |
| FR-MEM-05 | 显示 arena/peak/alignment/entries 和复用收益 | 数值与计划一致 |
| FR-MEM-06 | 权重显示 owner op、参数名、Shape、Pitch、size、offset | 每个常量有 owner |
| FR-MEM-07 | 权重按算子/层分组，展示相对位置与总大小 | 总宽对应 `constants_bytes` |
| FR-MEM-08 | 连续相同 Layer 折叠为 `×N` | Qwen3 显示代表层 ×28 |
| FR-MEM-09 | 布局不同的 Layer 单独显示并标记差异 | 合成测试图验证 |
| FR-MEM-10 | 输出自包含 SVG，含标题、图例、哈希、警告 | 浏览器离线打开 |

### 2.2 非功能需求

| 编号 | 需求 | 落地方式 |
|------|------|----------|
| NFR-MEM-01 | 无第三方依赖 | Python 标准库 + ToolChain 解析 |
| NFR-MEM-02 | 确定性 | 稳定排序，无时间戳/随机数/绝对路径 |
| NFR-MEM-03 | 性能 | Qwen3 折叠视图 < 3 s |
| NFR-MEM-04 | 可读性 | 地址轴、lane、图例、分组颜色、中文标签 |
| NFR-MEM-05 | 内存 | 不读取 `weights.bin` |
| NFR-MEM-06 | 离线 | SVG 无外部资源 |

### 2.3 约束条件

- Activation arena 不包含输入和常量；与 MemoryPass 一致。
- `memory_plan.entries` 的 offset 是逻辑偏移。
- Activation 的 `TensorMeta.pitch` 在 ShapePass 后为空；可视化时按
  row-major contiguous 推导 `pitch = shape`，并标注“pitch=contiguous”。
- 常量使用 `TensorMeta.pitch`；若 `pitch` 与 `shape` 不同，图中同时显示。
- 当前 Qwen3 包 metadata-only，权重无实际 offset；按常量顺序 + 256 B
  对齐计算 planned offset，并标注 `planned layout`。
- 输出遵守模型 artifacts 目录规则。

### 2.4 验收标准

| 编号 | 验收标准 |
|------|----------|
| AC-MEM-01 | Qwen3 Activation 面板显示 arena/peak/alignment、代表层和 28 层复用条 |
| AC-MEM-02 | 每个 Activation 显示 owner、Shape、Pitch、offset、size |
| AC-MEM-03 | 复用区块共享 offset 区间；生命周期重叠的区块分开 |
| AC-MEM-04 | 权重面板显示 owner、参数名、Shape、Pitch、size、offset |
| AC-MEM-05 | 重复 Layer 只画代表层并标注 ×28；差异层单独显示 |
| AC-MEM-06 | 相同输入与选项生成字节一致 SVG |
| AC-MEM-07 | 输出位于 `Models/Qwen3/artifacts/qwen3_0_6b_memory.svg` |

## 3. 概要设计

### 3.1 数据流

```text
model.mgf
  ├─ manifest["graph"] → ModelGraph（shape/pitch/tensor/op）
  ├─ manifest["optimization"]["memory_plan"] → entries
  └─ manifest["weights"]["entries"] → weight metadata
          │
          ▼
   MemoryViewModelBuilder
          │ activation blocks / weight blocks / layer runs
          ▼
   RepeatCollapser
          │ representative layer + ×N + variants
          ▼
   LayoutEngine
          │ address axes / lanes / groups
          ▼
   SvgRenderer
          ▼
   Models/<模型目录>/artifacts/<model-name>_memory.svg
```

### 3.2 模块划分

```text
Graph/tools/graph_memory/
├── __init__.py
├── package.py              # 读取 model.mgf
├── view_model.py           # ActivationBlock / WeightBlock / LayerRun
├── repeats.py              # 重复层识别与差异检测
├── layout.py               # 两面板布局
├── svg.py                  # 自包含 SVG 渲染
├── cli.py                  # 参数解析与错误码
├── visualize_memory.py     # CLI 启动器
├── .doc/graph_memory_design.md
├── .skill/SKILL.md
└── tests/
    ├── test_memory_visualizer.py
    └── test_results.md
```

### 3.3 视图模型

```python
@dataclass(frozen=True)
class ActivationBlock:
    tensor: str
    owner_op: str
    precision: str
    shape: tuple[int | str, ...]
    pitch: tuple[int, ...]
    pitch_derived: bool
    offset: int
    size: int
    birth: int
    last_use: int
    reusable: bool
    repeat_count: int

@dataclass(frozen=True)
class WeightBlock:
    name: str
    owner_ops: tuple[str, ...]
    precision: str
    shape: tuple[int | str, ...]
    pitch: tuple[int, ...]
    offset: int
    size: int
    source: str | None
    available: bool
    repeat_count: int

@dataclass(frozen=True)
class MemoryViewModel:
    title: str
    model_name: str
    manifest_sha256: str
    ir_sha256: str
    alignment: int
    arena_bytes: int
    peak_bytes: int
    constants_bytes: int
    activation_blocks: tuple[ActivationBlock, ...]
    weight_blocks: tuple[WeightBlock, ...]
    activation_runs: tuple[LayerRunView, ...]
    weight_runs: tuple[LayerRunView, ...]
    warnings: tuple[str, ...]
```

### 3.4 技术选型与取舍

| 主题 | 选择 | 理由 | 代价 |
|------|------|------|------|
| 输出 | 一个 SVG 两面板 | 便于整体审阅 | 文件较高 |
| Activation | 代表层 + 28 层条 | 避免 312 条标签 | 细节需看代表层 |
| 复用 | 生命周期 lane + 共享 offset | 直观 | 区块可能有交叉 |
| 权重 | planned offset | 无权重时仍可看相对布局 | 不是实际物理地址 |

## 4. 详细设计

### 4.1 模型包读取

- 调用 `ToolChain.serializer.load_package(package_dir, verify=True)`。
- 从 `manifest["graph"]` 构造 `ModelGraph`。
- 从 `manifest["optimization"]["memory_plan"]` 读取 summary 与 entries。
- 从 `manifest["weights"]["entries"]` 读取权重 offset/available/source。
- 不读取 `weights.bin`。

### 4.2 Activation 区块

对每条 memory entry：

1. 查 `TensorMeta`；
2. producer map 找 owner op；
3. Shape 来自 tensor.shape；
4. Pitch：
   - tensor.pitch 非空时使用；
   - 否则按 row-major contiguous 推导 `pitch = shape`，
     `pitch_derived = True`；
5. offset/size/birth/last_use/reusable 来自 memory entry；
6. 校验 `size` 与 tensor.byte_size；不一致时警告，布局使用 entry.size。

### 4.3 Activation 面板

- **Arena overview**：地址轴 `0..arena_bytes`；显示 28 层复用条。
  - 每层一条细带，按该层 Activation 的 offset/size 绘制区块；
  - 代表层可高亮；
  - 条带左侧标注 `Layer 0..27`，右侧标注 `×28`。
- **Representative detail**：代表层的每个 Activation 一个区块，
  lane 分配避免生命周期重叠；标签：
  ```text
  layer.0.q_path.out
  shape 1×256×2048
  pitch 1×256×2048 (contiguous)
  offset 1,048,576  size 1,048,576 B
  ```
- 非层 Activation（embedding/final norm/lm head）单独显示在 detail 中。
- 复用区块：相同 offset 区间的区块用相同底色 + 斜线纹理；生命周期
  重叠的区块不在同一 lane。
- Arena summary：
  ```text
  arena TBD B | peak TBD B | alignment 256 B | entries 312
  ```

### 4.4 权重区块

1. 每个常量找消费 op：
   - 遍历 ops 的 inputs，建立 constant → owners 映射；
   - tied embedding 显示多个 owner。
2. Shape/Pitch：
   - shape 来自 tensor.shape；
   - pitch 来自 tensor.pitch（常量有 pitch）；
   - size 来自 tensor.byte_size。
3. Offset：
   - 若 `weights.entries` 有实际 offset，使用实际值；
   - 否则按 `ModelGraph.constants` 顺序、256 B 对齐计算 planned offset；
   - 图中标注 `planned layout`。
4. 权重面板：
   - **Overview**：权重 arena `0..constants_bytes`；按 offset 画区块；
     重复层折叠为 `LayerWeights ×28` 大区块；
   - **Detail**：代表层每个权重一个区块，按输入顺序排列；标签：
     ```text
     owner layer.0.q_path
     layer.0.q_proj.weight
     shape 2048×1024  pitch 2048×1024
     offset 0  size 4,194,304 B
     ```
   - 非层权重（embedding/final norm/lm head）单独显示。
   - 多 owner 的常量显示 `shared by: embedding, lm_head`。

### 4.5 重复层识别

1. 按 `layer.<n>.` 分组；
2. 层内存签名比较：
   - Activation：归一化 op/tensor 名、shape、pitch、size，不比较绝对 offset
     （分配器状态会造成 offset 漂移）；
   - 权重：参数名、shape、pitch、size、相对顺序；
3. 连续相同签名合并为 `LayerRun`；
4. 代表层显示详细区块，标注 `×N`；
5. 非基准签名的层单独显示，红色边框并标记“差异”；
6. 不提供完整 28 层展开（保留调试开关）。

### 4.6 布局与 SVG

- 一个 SVG，两个纵向面板：
  1. Activation（overview + detail）
  2. Weight（overview + detail）
- 宽度按字节比例；超长 arena 使用统一缩放。
- 颜色：
  - Activation 按 layer/lifetime 分组；
  - 复用区块斜线纹理；
  - 权重按 owner/category 分组；
  - 差异层红色。
- 图例：Activation、复用、权重、共享权重、差异层、planned layout。
- 标题/页脚：模型名、manifest/IR 哈希、arena/peak/constants/alignment。
- 无时间戳、无绝对路径，确定性输出。

### 4.7 CLI 与 API

```bash
PYTHONUTF8=1 python Graph/tools/graph_memory/visualize_memory.py \
  --package Models/Qwen3/artifacts/qwen3_0_6b/model.mgf \
  --output Models/Qwen3/artifacts/qwen3_0_6b_memory.svg
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--package` | 必填 | `model.mgf` 或包目录 |
| `--output` | 由包路径推导 | 输出 SVG |

API：

```python
view = build_memory_view(package_path)
render_memory_svg(view, output_path)
```

### 4.8 错误处理

| 场景 | 行为 |
|------|------|
| 包/头部/manifest 错误 | `BuildError`，退出码 3 |
| 缺少 memory_plan | `BuildError`，指出字段 |
| Activation 无 producer | 警告，owner=unknown |
| 权重无 owner | 警告，owner=unused |
| 无实际权重 offset | planned layout 并标注 |
| Layer 布局不一致 | 拆分差异层并警告 |
| 输出目录不可写 | 报错，非零退出码 |

### 4.9 性能

- 折叠视图：代表层 11 个 Activation + 14 个权重组；
- 28 层复用条仅绘制小矩形；
- 不加载权重；
- SVG 预估 < 300 KB。

### 4.10 输出命名

```text
Models/<模型目录>/artifacts/<model-name>_memory.svg
```

Qwen3：

```text
Models/Qwen3/artifacts/qwen3_0_6b_memory.svg
```

## 5. 实施步骤

| 阶段 | 任务 | 交付 |
|------|------|------|
| S1 | 设计文档定稿 | 本文档 |
| S2 | 包读取与 ViewModel | `package.py`、`view_model.py` |
| S3 | Activation 区块/Pitch/复用 | `view_model.py` |
| S4 | 权重 owner/planned offset | `view_model.py` |
| S5 | 重复层识别 | `repeats.py` |
| S6 | 两面板布局与 SVG | `layout.py`、`svg.py` |
| S7 | CLI 与输出 | `visualize_memory.py` |
| S8 | 测试与 Qwen3 出图 | `tests/` |
| S9 | skill | `.skill/SKILL.md` |

风险：

| 编号 | 风险 | 应对 |
|------|------|------|
| R-01 | 复用区块重叠 | lane + 纹理 + 代表层 |
| R-02 | 无真实权重 offset | planned layout 标注 |
| R-03 | 层 offset 漂移导致误判差异 | 比较 shape/pitch/size，不比较绝对 offset |
| R-04 | 312 条 Activation 拥挤 | 28 层条 + 代表层 |
| R-05 | SVG 字体差异 | 通用字体栈，人工验收 |

## 6. 测试与验证设计

### 6.1 测试策略

- 单元：包读取、Activation/Weight 字段、Pitch 推导、planned offset、
  重复层、差异层。
- 集成：Qwen3 生成 SVG，检查两个面板。
- 确定性：两次渲染字节一致。
- 人工验收。

### 6.2 用例

| 编号 | 用例 | 预期 |
|------|------|------|
| UT-MEM-001 | Qwen3 包 | 312 Activation、310 常量 |
| UT-MEM-002 | Activation 字段 | owner/shape/pitch/offset/size 一致 |
| UT-MEM-003 | Activation Pitch | 空 pitch 按 shape 推导并标记 |
| UT-MEM-004 | 权重 owner | 每个常量映射到消费算子 |
| UT-MEM-005 | planned offset | 256 B 对齐、顺序正确 |
| UT-MEM-006 | 3 层同构 | 折叠 ×3 |
| UT-MEM-007 | 3 层中一层不同 | 差异层单独显示 |
| UT-MEM-008 | 两次渲染 | SVG 字节一致 |
| IT-MEM-001 | Qwen3 Activation 面板 | arena/peak/alignment/复用条 |
| IT-MEM-002 | Qwen3 权重面板 | owner/shape/pitch/size/offset/×28 |
| IT-MEM-003 | 输出路径 | `qwen3_0_6b_memory.svg` |

### 6.3 环境与命令

```bash
PYTHONUTF8=1 python Graph/tools/graph_memory/tests/test_memory_visualizer.py -v
```

Windows 11、Python 3.13。

### 6.4 通过标准

- 单元/集成测试全部通过；
- Qwen3 两个面板信息完整；
- 复用与重复层折叠正确；
- SVG 字节一致；
- 用户人工确认。

### 6.5 人工验收清单

- [ ] Activation 显示 owner、Shape、Pitch、offset、size。
- [ ] 能看出哪些区块复用同一 offset。
- [ ] 28 层复用条与代表层一致。
- [ ] 权重显示 owner、参数名、Shape、Pitch、size、offset。
- [ ] 权重相对位置和顺序正确。
- [ ] 重复 Layer 只画代表层并标注 ×28。
- [ ] 布局不同的层单独显示。
- [ ] 两个面板、图例、哈希完整。
