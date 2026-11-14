# Graph 静态图可视化工具设计文档

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 日期 | 2026-11-14 |
| 文档状态 | 已确认，随代码同步 |
| 模块目录 | `Graph/tools/graph_visualizer/` |
| 输入 | `model.mgf` |
| 输出 | 模型 artifacts 目录下的 `*_graph.svg` |
| skill | `Graph/tools/graph_visualizer/.skill/SKILL.md` |

## 已确认决策

| 决策 | 结论 |
|------|------|
| 输出格式 | 仅 SVG |
| 权重展示 | 不显示权重值；每个显示算子消费的常量作为虚线参数边，仅标注精度和 shape |
| 重复 Layer | 绘制代表性子图，标注重复次数 |
| 差异 Layer | 与基准层结构不同的 Layer 单独显示并标记 |
| 完整图 | 不提供完整展开视图 |
| 布局方向 | 自上而下 |
| 输出目录 | `Models/<模型目录>/artifacts/` |
| 输出文件 | `<model-name>_graph.svg` |

## 1. 背景与目标

### 1.1 背景

模型经 Converter、Optimizer、Serializer 后生成 `model.mgf`。manifest 包含
优化后的 `ModelGraph`、算子属性、张量精度、显存计划和权重表，但结构化 JSON
不便于人工确认融合算子的数据流。`dump_ir()` 适合机器校验，不适合整体审阅。

### 1.2 目标

1. 从 `model.mgf` 生成类似算子流程图的静态 SVG。
2. 节点表示融合算子，显示名称、`OpType`、类别、精度、散算子组成和关键属性。
3. 输入、输出、激活边为实线箭头；参数边为虚线；边标注精度和 shape。
4. 连续相同 Layer 折叠为代表性子图并标注 `×N`。
5. 与基准层结构不同的 Layer 单独绘制并标记差异。
6. 输出自包含 SVG，离线可打开，相同输入生成字节一致结果。

### 1.3 非目标

- 不显示权重数值、统计值、采样值或权重哈希。
- 不提供完整展开的全部 Layer 视图。
- 不做运行时性能、显存实测或数值推理。
- 不做图编辑、图优化、ONNX/Protobuf 导出或 HTML 交互。
- 不替代 `dump_ir()` 的机器校验。

### 1.4 交付物与位置

| 交付物 | 位置 |
|--------|------|
| 设计文档 | `Graph/tools/graph_visualizer/.doc/graph_visualizer_design.md` |
| CLI 启动器 | `Graph/tools/visualize_graph.py` |
| 工具代码 | `Graph/tools/graph_visualizer/` |
| 测试代码 | `Graph/tools/graph_visualizer/tests/test_visualizer.py` |
| 测试结果 | `Graph/tools/graph_visualizer/tests/test_results.md` |
| skill | `Graph/tools/graph_visualizer/.skill/SKILL.md` |
| 示例产物 | `Models/Qwen3/artifacts/qwen3_0_6b_graph.svg` |

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收方式 |
|------|------|----------|
| FR-VIS-01 | 读取 `model.mgf`，校验头部、manifest SHA-256 与版本 | 损坏包报错，不生成图 |
| FR-VIS-02 | 节点显示融合算子名称、`OpType`、类别、精度、散算子组成、属性 | Qwen3 节点信息完整 |
| FR-VIS-03 | 输入、输出、激活边为实线箭头，标注 tensor 名、精度、shape | 抽查与 manifest 一致 |
| FR-VIS-04 | 每个显示算子消费的常量作为虚线参数边，仅标注精度和 shape | 折叠视图只显示代表层参数边 |
| FR-VIS-05 | 参数边不显示权重名称、数值、统计值、可用性或哈希 | SVG 中无权重内容 |
| FR-VIS-06 | 连续相同 Layer 折叠为代表性子图，标注 `×N` | Qwen3 显示一个 Decoder Layer `×28` |
| FR-VIS-07 | 与基准层结构不同的 Layer 单独绘制并标记差异 | 合成测试图中差异层单独出现 |
| FR-VIS-08 | 输出自包含 SVG，含标题、图例、模型/IR 哈希 | 浏览器离线打开正常 |
| FR-VIS-09 | 输出路径默认位于模型 artifacts 目录 | Qwen3 输出到 `Models/Qwen3/artifacts/` |
| FR-VIS-10 | 未知 OpType、包损坏、输出失败时给出明确错误或警告 | 不静默丢弃节点 |

### 2.2 非功能需求

| 编号 | 需求 | 落地方式 |
|------|------|----------|
| NFR-VIS-01 | 无第三方依赖 | Python 标准库 + 现有 ToolChain 解析代码 |
| NFR-VIS-02 | 确定性 | 固定节点/边排序，无时间戳、随机数、绝对路径 |
| NFR-VIS-03 | 性能 | Qwen3 视图生成时间 < 3 s |
| NFR-VIS-04 | 可读性 | 自上而下分层布局、箭头、颜色图例、中文标签 |
| NFR-VIS-05 | 内存 | 不读取 `weights.bin` 内容 |
| NFR-VIS-06 | 离线 | SVG 不引用外部 JS/CSS/字体 |
| NFR-VIS-07 | 可维护 | 融合算子组成和类别独立成表 |

### 2.3 约束条件

- 输入是 Serializer 生成的 `model.mgf` v1。
- 当前 Qwen3 包为 metadata-only；工具不依赖权重内容。
- 精度以 `ModelGraph` 中 `TensorMeta.precision` 为准。
- 参数边来自 `ModelGraph.constants` 与 `manifest["weights"]["entries"]`。
- 不读取 `weights.bin`，不依赖 `_graph` 扩展。
- 输出遵守模型产物目录规则；代码遵守 Google Style 与中文注释要求。

### 2.4 验收标准

| 编号 | 验收标准 |
|------|----------|
| AC-VIS-01 | Qwen3 图包含 2 个输入、embedding、Decoder Layer 代表子图 `×28`、final norm、LM Head、1 个输出 |
| AC-VIS-02 | 每个融合算子节点显示散算子组成、类别、精度和属性 |
| AC-VIS-03 | 激活边为实线箭头，参数边为虚线，均标注精度和 shape |
| AC-VIS-04 | SVG 中不出现权重值、统计值、采样值或权重哈希 |
| AC-VIS-05 | 所有层一致时只画一个代表层；某层不同时该层单独显示并标记 |
| AC-VIS-06 | 相同输入与选项生成字节一致的 SVG |
| AC-VIS-07 | 输出文件位于 `Models/Qwen3/artifacts/qwen3_0_6b_graph.svg` |
| AC-VIS-08 | 单元/集成测试通过，用户人工审阅确认图与 manifest 一致 |

## 3. 概要设计

### 3.1 数据流

```text
model.mgf
  ├─ 112 字节头 → PackageHeader
  ├─ canonical JSON manifest → ModelGraph / WeightEntry
  └─ weights.bin（不读取内容）
          │
          ▼
   ViewModelBuilder
          │ nodes / edges / layer runs
          ▼
   RepeatAnalyzer
          │ 连续相同层折叠；差异层单独保留
          ▼
   LayoutEngine
          │ 自上而下分层、端口分配
          ▼
   SvgRenderer
          ▼
   Models/<模型目录>/artifacts/<model-name>_graph.svg
```

### 3.2 模块划分

```text
Graph/tools/
├── visualize_graph.py                  # CLI 启动器
└── graph_visualizer/                   # 功能模块根目录 / Python 包
    ├── __init__.py
    ├── package.py                      # 读取并校验 model.mgf
    ├── view_model.py                   # Node / Edge / LayerRun / ViewModel
    ├── composition.py                  # 融合算子 → 散算子、类别、颜色
    ├── formatting.py                   # 属性、shape、精度、标签格式化
    ├── repeats.py                      # 连续 Layer 分组与差异检测
    ├── layout.py                       # 自上而下 DAG 布局
    ├── svg.py                          # 自包含 SVG 渲染
    ├── cli.py                          # 参数解析与错误码
    ├── .doc/
    │   └── graph_visualizer_design.md
    ├── .skill/
    │   └── SKILL.md
    └── tests/
        ├── test_visualizer.py
        └── test_results.md
```

### 3.3 视图模型

```python
@dataclass(frozen=True)
class Node:
    id: str
    name: str
    op_type: str
    category: str
    precision: str
    attrs: dict[str, Any]
    composition: tuple[str, ...]
    node_type: str  # compute/input/output

@dataclass(frozen=True)
class Edge:
    id: str
    src: str | None          # None 表示参数边
    dst: str
    tensor: str
    precision: str
    shape: tuple[int, ...]
    kind: str                # input/activation/output/parameter

@dataclass(frozen=True)
class LayerRun:
    start: int
    end: int
    signature: str
    repeat_count: int
    is_variant: bool
```

### 3.4 技术选型与取舍

| 主题 | 选择 | 理由 | 代价 |
|------|------|------|------|
| 渲染格式 | 自包含 SVG | 矢量、可离线打开、无 JS 依赖 | 交互能力弱 |
| 绘图实现 | 纯 Python 生成 SVG | 无 Graphviz/浏览器依赖，确定性好 | 布局算法需自行实现 |
| 布局 | 自上而下分层 DAG | 适合算子流程图 | 极端大图可能仍需人工调整 |
| 输入解析 | 复用 `ToolChain.serializer.load_package` | 头部、manifest、哈希校验已实现 | 依赖 Serializer 模块 |
| 重复处理 | 连续 Layer 结构签名比较 | 保留层顺序，容易发现差异层 | 非 `layer.<n>` 命名块不自动折叠 |

## 4. 详细设计

### 4.1 模型包读取

1. 调用 `ToolChain.serializer.load_package(package_dir, verify=True)`，校验
   magic、版本与 manifest SHA-256。
2. 从 `manifest["graph"]` 构造 `ModelGraph.from_dict()`。
3. 从 `manifest["weights"]["entries"]` 读取常量 shape 与 dtype。
4. 不读取 `weights.bin`，不要求 `_graph` 扩展。

### 4.2 节点：融合算子

节点内容：

```text
layer.0.mlp_gate_up
kLinearSwiGlu  [mlp / fp16]
组成: linear(gate) → linear(up) → silu → mul
attrs: hidden=1024, inter=3072, act=silu
```

融合算子组成表覆盖 Qwen3 使用类型；未知类型显示“未登记融合组成”。

| OpType | 散算子组成 |
|--------|------------|
| `kGather` | `gather` |
| `kRmsNorm` | `pow`、`mean`、`add(eps)`、`rsqrt`、`mul(weight)` |
| `kLinear` | `matmul`，可选 `add(bias)` |
| `kRope` | `cos/sin`、`rotate_half`、`mul`、`add` |
| `kMegaAttention` | `repeat_kv`、`QK^T`、`scale`、`causal_mask`、`softmax`、`PV`、`merge_heads` |
| `kLinearSwiGlu` | `linear(gate)`、`linear(up)`、`silu`、`mul` |
| `kAdd` | `add` |
| `kLinearRmsNorm` | `linear`、`rms_norm` |
| `kLinearRmsNormRope` | `linear`、`rms_norm`、`rope` |
| `kLinearRmsNormRopeQuantPagedKVStore` | `linear`、`rms_norm`、`rope`、`quant`、`paged_kv_store` |
| `kAddRmsNorm` | `add`、`rms_norm` |
| `kFfn` | `gate/up`、`activation`、`down` |
| 其他/未知 | 显示 `op_type`，标注“未登记融合组成” |

类别映射：

| 类别 | OpType 示例 |
|------|-------------|
| `io` | `kInput`、`kOutput` |
| `embedding` | `kGather`、`kExpandEmbedding` |
| `norm` | `kRmsNorm`、`kAddRmsNorm` |
| `linear` | `kLinear`、`kLinearMul` |
| `attention` | `kMegaAttention`、`kSparseMla` |
| `mlp` | `kLinearSwiGlu`、`kFfn` |
| `elementwise` | `kAdd` |
| `position` | `kRope`、`kRotate` |
| `routing` | 各 Router |
| `quant` | `kQuant`、`kKvCompress` |
| `other` | 未归类类型 |

### 4.3 边：输入、输出、参数

| 边类型 | 样式 | 可见标签 |
|--------|------|----------|
| 图输入 | 蓝色实线箭头 | `input_ids [int64, 1×256]` |
| 激活 | 深灰实线箭头 | `hidden.0 [fp16, 1×256×1024]` |
| 图输出 | 绿色实线箭头 | `logits [fp16, 1×256×151936]` |
| 参数 | 橙色虚线箭头，带 `W` 标记 | `[fp16, 1024×1024]` |

参数边规则：

- 每个显示算子消费的常量生成一条参数边，从目标算子左侧的 `W` 标记
  进入；折叠层只显示代表层的参数边；
- 可见标签仅显示精度和 shape；
- 不显示权重名称、数值、统计值、可用性或 SHA-256；
- 同一常量被多个 op 消费时，为每个消费 op 各生成一条参数边。

### 4.4 属性展示

- bool/int/float/string 直接显示；
- list 长度 ≤ `--max-label-items`（默认 4）时内联，否则显示 `[N items]`；
- dict 只显示键和截断值；
- 每个节点最多显示 4 行属性，超出用 `...` 省略。

### 4.5 重复 Layer 与差异 Layer

1. 用 `^(?P<prefix>layer)\.(?P<index>\d+)\.` 识别层内 op。
2. 每层构造签名：归一化 op 名称、op_type、归一化 attrs、层内边、对外角色。
3. 将连续且签名相同的层合并为一个 `LayerRun`。
4. 每个 run 绘制一个代表性子图：
   - `repeat_count > 1` 标注 `×N (layer.a–layer.b)`；
   - 非基准 run 标记为差异，使用红色边框和 `差异` 徽标；
   - 非基准但重复次数 > 1 的 run 标注 `变体 ×N`。
5. 所有层一致时只显示一个代表层 `×28`。
6. 某层不同时按顺序显示多个 run，例如：

```text
[ Layer 0 (差异) ] → [ DecoderLayer ×27 (layer.1–layer.27) ]
```

7. 不提供完整展开模式。

### 4.6 布局

- 自上而下：输入在顶部，输出在底部。
- 仅输入/激活/输出边参与层级计算；参数边不改变 rank。
- 每个 `LayerRun` 作为整体参与分层。
- 同层节点按首次出现顺序加重心排序。
- 节点宽度最大 360 px，超长文本换行。
- 边使用正交折线或贝塞尔曲线，箭头指向目标端口。
- 图顶部放标题和图例，底部放 manifest/IR 哈希。

### 4.7 SVG 规范

- 自包含 `<svg>`，`<defs>` 定义箭头 marker。
- 无外部 JS/CSS/字体引用。
- 配色：输入 `#2563eb`、输出 `#16a34a`、激活 `#334155`、参数
  `#d97706`、attention `#7c3aed`、norm `#0f766e`、mlp `#b45309`、
  linear `#475569`、差异层 `#dc2626`。
- 节点/边 ID 使用稳定序号；无时间戳、随机数、绝对路径。
- 相同输入和选项生成字节一致的 SVG。

### 4.8 CLI 与 API

```bash
PYTHONUTF8=1 python Graph/tools/visualize_graph.py \
  --package Models/Qwen3/artifacts/qwen3_0_6b/model.mgf \
  --output Models/Qwen3/artifacts/qwen3_0_6b_graph.svg
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--package` | 必填 | `model.mgf` 文件或包目录 |
| `--output` | 由包路径推导 | 输出 SVG 路径 |
| `--max-label-items` | 4 | 节点属性显示上限 |

默认输出：

```text
Models/<模型目录>/artifacts/<model-name>_graph.svg
```

API：

```python
view = build_view_model(package_path)
render_svg(view, output_path)
```

### 4.9 错误处理

| 场景 | 行为 |
|------|------|
| 包不存在/头部损坏/manifest 哈希不符 | 抛 `BuildError`，退出码 3 |
| manifest 缺少 `graph` | 抛 `BuildError`，指出缺失字段 |
| 未知 OpType | 正常绘制，标注“未登记融合组成”，stderr 警告 |
| Layer 结构不一致 | 按 run 拆分，差异层单独显示并警告 |
| 输出目录不可写 | 报错，返回非零退出码 |

### 4.10 性能与内存

- 默认视图约 20–40 个节点，不读取权重内容。
- 不构建完整 423 op 视图；SVG 预估小于 300 KB。

### 4.11 输出命名

- 输入：`Models/<模型目录>/artifacts/<model-name>/model.mgf`
- 输出：`Models/<模型目录>/artifacts/<model-name>_graph.svg`
- Qwen3：`Models/Qwen3/artifacts/qwen3_0_6b_graph.svg`

## 5. 实施步骤

| 阶段 | 任务 | 依赖 | 交付 |
|------|------|------|------|
| S1 | 设计文档定稿 | 用户确认 | 本文档 |
| S2 | 包读取与 ViewModel | Serializer loader | `package.py`、`view_model.py` |
| S3 | 组成表、类别、属性格式化 | S2 | `composition.py`、`formatting.py` |
| S4 | 连续 Layer 分组与差异检测 | S2 | `repeats.py` |
| S5 | 自上而下布局与 SVG 渲染 | S2–S4 | `layout.py`、`svg.py` |
| S6 | CLI 与输出路径 | S5 | `visualize_graph.py`、`cli.py` |
| S7 | 测试与人工审阅 | S6 | `tests/` |
| S8 | skill 与交付 | S7 | `.skill/SKILL.md` |

风险：

| 编号 | 风险 | 应对 |
|------|------|------|
| R-01 | 纯 Python 布局交叉 | 折叠视图节点少；稳定排序与重心排序 |
| R-02 | 融合算子组成不准确 | 组成表逐项对照 Graph 设计文档与 Qwen3 Builder |
| R-03 | 差异层判断遗漏 | 用完整结构签名比较，不只比较 op 数量 |
| R-04 | SVG 字体/中文显示差异 | 通用无衬线字体栈；测试报告记录查看环境 |

## 6. 测试与验证设计

### 6.1 测试策略

- 单元测试：包读取、视图模型、组成表、属性格式化、Layer 分组。
- 集成测试：Qwen3 `model.mgf` 生成 SVG，检查关键节点和边。
- 差异测试：合成 3 层图，其中一层不同，确认差异层单独显示。
- 确定性测试：同一输入两次生成 SVG 字节一致。
- 人工验收：用户打开 SVG 确认布局、箭头、精度、参数边、`×28`。

### 6.2 用例设计

| 编号 | 用例 | 预期 |
|------|------|------|
| UT-VIS-001 | 读取合法 Qwen3 包 | 返回 735 tensor、423 计算 op |
| UT-VIS-002 | 损坏 manifest | 抛 `BuildError` |
| UT-VIS-003 | `kLinearSwiGlu` 组成 | 返回 4 个散算子 |
| UT-VIS-004 | 未知 OpType | 不崩溃，标注“未登记” |
| UT-VIS-005 | 常量生成参数边 | 参数边标签仅 shape/精度 |
| UT-VIS-006 | 激活边标签 | tensor 名、精度、shape 与 manifest 一致 |
| UT-VIS-007 | 3 层同构 | 折叠为 `×3` |
| UT-VIS-008 | 3 层中一层不同 | 差异层单独显示并标记 |
| UT-VIS-009 | `--max-label-items` | 属性截断符合预期 |
| UT-VIS-010 | 两次渲染 | SVG 字节一致 |
| IT-VIS-001 | Qwen3 默认视图 | 含 embedding、Layer ×28、final norm、LM Head、output |
| IT-VIS-002 | 输出路径 | 生成 `Models/Qwen3/artifacts/qwen3_0_6b_graph.svg` |

### 6.3 测试环境与命令

```bash
PYTHONUTF8=1 python Graph/tools/graph_visualizer/tests/test_visualizer.py -v
```

环境：Windows 11、Python 3.13。输入为
`Models/Qwen3/artifacts/qwen3_0_6b/model.mgf`。

### 6.4 通过标准

- 单元/集成测试全部通过。
- Qwen3 图节点、边、精度、参数边、`×28` 正确。
- 差异层测试通过。
- 相同输入 SVG 字节一致。
- 用户人工确认图与 manifest / 设计意图一致。

### 6.5 人工验收清单

- [ ] 节点是融合算子名称，显示散算子组成、类别、精度、属性。
- [ ] 激活/输入/输出边为实线箭头。
- [ ] 参数边为虚线，仅显示精度和 shape。
- [ ] SVG 中没有任何权重值、统计值、采样值或权重哈希。
- [ ] 重复 Layer 只画代表性子图并标注 `×N`。
- [ ] 结构不同的 Layer 单独显示并标记“差异”。
- [ ] 自上而下布局，箭头方向清晰。
- [ ] 图例、标题、manifest/IR 哈希完整。
- [ ] 输出位于 `Models/Qwen3/artifacts/`。
