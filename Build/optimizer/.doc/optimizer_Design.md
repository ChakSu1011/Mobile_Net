# Optimizer 模块设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v1.0 |
| 文档状态 | 已实现，随 Build 阶段代码同步 |
| 编写日期 | 2026-11-13 |
| 所属阶段 | Build（离线工具链） |
| 模块目录 | `Build/optimizer/` |
| 上游依赖 | `Build/converter` 输出的 `ModelGraph` |
| 下游模块 | `Build/serializer` |
| 权威图设计 | [Graph/.doc/Graph设计文档.md](../../../Graph/.doc/Graph设计文档.md) |

## 1. 背景与目标

### 1.1 背景

`Graph` 只负责图结构与依赖关系，不推导 shape、precision，也不做显存
规划；其 Python 绑定不暴露 Op/Storage 访问器。因此 Build 阶段的优化对象
是 Converter 旁路记录的 `ModelGraph`，而不是试图修改不可变的 Graph
图对象。优化结果写回 `ModelGraph`，Serializer 再据此重建可执行图并序列化。

### 1.2 目标

1. 实现 Precision Infer：把 BF16 激活/权重策略转换为 FP16，记录 FP32
   累加策略，保持 int64 输入不变。
2. 实现 Shape Infer：以 `B=1`、`S=max_num_tokens` 为规划上界，推导所有
   激活张量形状并校验算子形状契约。
3. 实现 Memory Manage：基于拓扑序生命周期做 Activation 复用排版，输出
   偏移、生命周期、峰值与常量区大小。
4. 输出结构化优化报告（JSON 可序列化），包含精度表、形状表、显存计划与
   不变量检查结果。
5. 保证优化可重复：同一输入与配置产生完全一致的 `ModelGraph` 与显存计划。

### 1.3 非目标

- 不修改 Graph 的图对象；不新增/删除 op。
- 不实现量化（INT8/INT4）；只在策略结构中预留扩展字段。
- 不做 Dummy Run 真实执行；生命周期由静态依赖分析得到。
- 不分配真实显存；只输出逻辑偏移计划。
- 不处理 KV Cache、padding mask 与动态 batch > 1。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收要点 |
| --- | --- | --- |
| FR-OPT-01 | Precision Pass 必须把所有浮点激活与常量的目标精度统一为配置值（默认 `float16`） | 所有 activation/constant 的 `precision` 为目标值；int64 输入保持 `int64` |
| FR-OPT-02 | Precision Pass 必须在数值敏感算子属性中写入 `accumulate_precision=float32` | `rms_norm`、`linear`、`linear_swi_glu`、`mega_attention`、`rope` 均带该属性 |
| FR-OPT-03 | 精度转换不得改变张量字节数；源/目标元素大小不同的转换必须报错 | BF16→FP16 字节数不变；FP32→FP16 抛 `BuildError` |
| FR-OPT-04 | Shape Pass 必须为每个 activation 推导出完整静态形状 | 不存在空 shape 或符号维 |
| FR-OPT-05 | Shape Pass 必须校验算子形状契约（K 维、GQA 比例、RMSNorm 整除、Add 同形） | 非法形状抛 `BuildError` 并包含 op 名 |
| FR-OPT-06 | Shape Pass 必须以 `max_num_tokens` 绑定 `S`，并校验 `max_num_tokens <= context_length` | 规划上下界写入报告与 `ModelGraph.metadata` |
| FR-OPT-07 | Memory Pass 必须以拓扑序计算每个 activation 的首次定义与末次使用 | 每个 activation 有 `birth`、`last_use` |
| FR-OPT-08 | Memory Pass 必须按 256 字节对齐做复用排版，活跃区间重叠的张量不得共享字节 | 任意两个重叠生命周期张量的 `[offset, offset+size)` 不相交 |
| FR-OPT-09 | 图输入与图输出张量不得参与复用 | 二者 `reusable=false` |
| FR-OPT-10 | 报告必须包含常量字节数、激活区字节、理论峰值与复用后峰值 | 字段可 JSON 序列化且可脚本校验 |
| FR-OPT-11 | Optimizer 必须输出不变量检查结果，任一失败即报错 | `invariants` 全部 passed；失败抛 `BuildError` |
| FR-OPT-12 | Optimizer 必须支持 `target_precision="keep"` 关闭精度转换 | 关闭后精度与形状仍被推导，属性精度不变 |

### 2.2 非功能需求

| 编号 | 类别 | 要求 |
| --- | --- | --- |
| NFR-OPT-01 | 确定性 | 两次运行产生逐字段一致的 `ModelGraph`、显存计划与报告（不含时间戳） |
| NFR-OPT-02 | 性能 | Qwen3-0.6B 单次优化 < 5 s，内存增量 < 50 MB |
| NFR-OPT-03 | 可诊断 | 错误包含 pass 名、op 名、张量名与实际/期望值 |
| NFR-OPT-04 | 可组合 | 三个 Pass 可独立运行，也可按 Precision→Shape→Memory 顺序运行 |
| NFR-OPT-05 | 规范 | Google Style 与中文注释；不依赖 torch/transformers |

### 2.3 约束条件

- 输入 `ModelGraph` 必须已通过 `validate()`。
- 符号维只允许 `B` 与 `S`；Qwen3 prefill 图固定 `B=1`。
- 常量不参与 Activation 复用区，单独统计。
- 对齐默认 256 B，可按后端能力配置。
- 图输出由 `ModelGraph.outputs` 指定，输出 op 不产生新张量。

### 2.4 验收标准

| 编号 | 验收项 | 通过条件 |
| --- | --- | --- |
| AC-OPT-01 | 精度完整 | 无 `unknown`/源精度残留（`keep` 除外）；int64 输入不变 |
| AC-OPT-02 | 形状完整 | 423 个 Qwen3 activation 全部有静态形状 |
| AC-OPT-03 | 显存安全 | 生命周期重叠张量零字节重叠；峰值与报告一致 |
| AC-OPT-04 | 可重复 | 连续两次优化结果 `to_dict()` 相等 |
| AC-OPT-05 | 报告 | JSON 可解析；不变量全部 passed |

## 3. 概要设计

### 3.1 总体架构

```text
ModelGraph ──▶ PrecisionPass ──▶ ShapePass ──▶ MemoryPass ──▶ OptimizeResult
                    │                │              │              │
                    ▼                ▼              ▼              ▼
                精度表            形状表         显存计划       优化报告
```

Pipeline 在 `ModelGraph` 的深拷贝上执行，保证输入对象不被修改；任一 Pass
失败时不返回半成品。

### 3.2 模块划分

| 文件 | 职责 |
| --- | --- |
| `Build/optimizer/__init__.py` | 导出 `Optimizer`、`OptimizeConfig`、`OptimizeResult` |
| `Build/optimizer/precision.py` | Precision Pass |
| `Build/optimizer/shape.py` | Shape Pass |
| `Build/optimizer/memory.py` | 生命周期分析与显存排版 |
| `Build/optimizer/report.py` | 报告结构与不变量检查 |
| `Build/optimizer/optimizer.py` | Pass 编排与结果封装 |
| `Build/optimizer/__main__.py` | 命令行入口（读取 model_graph.json，输出优化结果） |

### 3.3 关键流程

1. 深拷贝输入 `ModelGraph`。
2. Precision Pass：确定目标精度，改写张量精度与算子精度属性，
   写入累加精度；校验字节数不变。
3. Shape Pass：按拓扑序逐 op 推导输出形状并写回张量；校验形状契约。
4. Memory Pass：计算生命周期、排序、first-fit 复用、生成 `MemoryPlan`。
5. Report：汇总统计并执行不变量检查。
6. 返回 `OptimizeResult`，供 Serializer 使用。

### 3.4 技术选型与取舍

| 决策点 | 选择 | 理由 | 代价 |
| --- | --- | --- | --- |
| 优化对象 | `ModelGraph` sidecar | Graph 不可变、元数据不可见 | 必须保证 sidecar 与图同步 |
| 形状推导 | 静态规则表 + 拓扑序 | 确定性、可测试、无需执行 | 新算子需补规则 |
| 动态维 | 固定 `B=1`、`S=max_num_tokens` 规划 | 与首期 prefill 图一致 | 不支持动态 batch |
| 显存复用 | 生命周期区间 + first-fit | 实现简单、可验证 | 非全局最优 |
| 精度策略 | BF16→FP16，FP32 累加 | 与引擎主路径一致 | 数值需后续真机对齐 |
| 报告格式 | JSON | 可审查、可脚本校验 | 体积大于二进制日志 |

## 4. 详细设计

### 4.1 数据结构

```python
@dataclass(frozen=True)
class OptimizeConfig:
    target_precision: str = "float16"   # 或 "keep"
    accumulate_precision: str = "float32"
    batch_size: int = 1
    max_num_tokens: int = 256
    context_length: int = 40960
    alignment: int = 256
    enable_reuse: bool = True
```

`MemoryEntry` 与 `MemoryPlan` 定义在 `Build/common/model_ir.py`：

```python
@dataclass(frozen=True)
class MemoryEntry:
    name: str
    offset: int
    size: int
    birth: int
    last_use: int
    reusable: bool

@dataclass(frozen=True)
class MemoryPlan:
    alignment: int
    arena_bytes: int
    peak_bytes: int
    constants_bytes: int
    entries: tuple[MemoryEntry, ...]
```

`OptimizeResult`：

```python
@dataclass(frozen=True)
class OptimizeResult:
    model_graph: ModelGraph
    config: OptimizeConfig
    memory_plan: MemoryPlan
    report: dict[str, Any]
```

### 4.2 接口定义

```python
class Optimizer:
    def __init__(self, config: OptimizeConfig | None = None) -> None: ...
    def run(self, model_graph: ModelGraph) -> OptimizeResult: ...

class PrecisionPass:
    def run(self, model_graph: ModelGraph) -> dict[str, Any]: ...

class ShapePass:
    def run(self, model_graph: ModelGraph) -> dict[str, Any]: ...

class MemoryPass:
    def run(self, model_graph: ModelGraph) -> MemoryPlan: ...
```

### 4.3 Precision Pass

#### 4.3.1 精度映射

| 张量类别 | 处理 |
| --- | --- |
| `input.input_ids`、`input.position_ids` | 保持 `int64` |
| constant（BF16） | 改为目标精度（FP16），`byte_size` 不变 |
| activation | 改为目标精度 |
| 图输出张量 | 跟随其生产 op |

目标精度只接受 `float16`、`bfloat16`、`float32`、`keep`。当目标与源元素
字节数不同（如 FP32→FP16）时抛 `BuildError`：本阶段不重排权重，体积变化
会破坏序列化权重表。

#### 4.3.2 累加属性

对以下 op 写入 `accumulate_precision`：

- `rms_norm`
- `linear`
- `linear_swi_glu`
- `mega_attention`
- `rope`

同时把 op 的 `precision` 属性改为目标精度；保留 `bias=false`、
`transpose_weight` 等既有属性。

#### 4.3.3 不变量

- 所有非 int64 张量精度 == 目标精度（`keep` 时 == 源精度）；
- `byte_size` 在转换前后相等；
- op 属性中的 `precision` 与输出张量精度一致。

### 4.4 Shape Pass

#### 4.4.1 符号绑定

- `B = batch_size`（首期固定 1）；
- `S = max_num_tokens`（默认 256）；
- `context_length` 只作为合法性上界：`max_num_tokens <= context_length`。

选择 256 作为默认规划上界，避免按 40960 全上下文规划 logits
（`1 × 40960 × 151936 × 2 B ≈ 12.4 GB`）导致不可用计划；该值可配置。

#### 4.4.2 算子形状规则

| op_type | 输入约束 | 输出 |
| --- | --- | --- |
| `gather` | indices rank=2，weight rank=2，axis=0 | `indices.shape + weight.shape[1:]` |
| `linear` | 输入末维 == `in_features`；权重 `[out, in]` | `输入[:-1] + [out_features]` |
| `linear_swi_glu` | 输入末维 == `hidden_size`；两个权重 `[intermediate, hidden]` | `输入[:-1] + [intermediate_size]` |
| `rms_norm` | 输入末维可被 `normalized_shape` 整除 | 与输入同形 |
| `rope` | 输入末维 == `num_heads * head_dim` | 与输入同形 |
| `mega_attention` | q 末维 == `num_heads*head_dim`；k/v 末维 == `num_kv_heads*head_dim`；比例匹配 | 与 q 同形 |
| `add` | 两个输入同形 | 与输入同形 |
| 其他 | 失败 | 抛 `BuildError` |

#### 4.4.3 精度补全

若 activation 的精度仍为 `unknown`，从生产 op 的 `precision` 属性继承；
仍无法确定时报错。

#### 4.4.4 输出绑定

输出 op 不产生新张量；Shape Pass 校验每个 `IOBinding.tensor` 已有形状，
并把输出张量标记为 `reusable=false`。

### 4.5 Memory Pass

#### 4.5.1 生命周期

- 入口先校验每个 activation 的 `static_shape` 已解析；任一 activation 形状为空
  或仍含符号维时立即抛出 `BuildError`（`run shape inference first`），禁止
  在未完成 Shape Pass 的图上做显存规划。
- 计算 op 拓扑序下标 `0..N-1`；输出绑定视为下标 `N` 的消费者。
- 图输入不进入生命周期表，也不参与 arena；常量同样不参与。
- activation `birth = 生产 op 下标`；
  `last_use = max(消费 op 下标, 输出绑定下标)`。
- 图输出张量 `reusable=false`。

#### 4.5.2 排版算法

1. 按 `(birth, -size, name)` 排序待分配张量，保证确定性。
2. 维护已分配列表 `(offset, end, last_use, name)`。
3. 对每个张量，从 offset=0 开始按 `alignment` 向上取整，检查与所有
   `last_use >= birth` 的已分配区间是否重叠；重叠则跳到该区间末尾继续。
4. 分配后更新 `arena_bytes = max(arena_bytes, offset + size)`。
5. `peak_bytes = max over op index of 活跃张量大小之和`（不含常量）。

`enable_reuse=false` 时每个张量独占新区间，用于对照与调试。

#### 4.5.3 不变量

- 任意两个生命周期重叠的张量字节区间不相交；
- 所有偏移按 `alignment` 对齐；
- `arena_bytes >= peak_bytes`；
- 常量字节数 = 所有 constant `byte_size` 之和；
- 无重复张量名。

### 4.6 报告

报告顶层字段：

```json
{
  "model": {"name": "...", "adapter": "..."},
  "bounds": {"batch": 1, "sequence": 256, "context_length": 40960},
  "precision": {"target": "float16", "accumulate": "float32",
                "converted_tensors": 733, "precision_histogram": {...}},
  "shape": {"resolved_tensors": 423, "unresolved": []},
  "memory": {"alignment": 256, "arena_bytes": ..., "peak_bytes": ...,
             "constants_bytes": 1192099840, "entries": 423},
  "invariants": [{"name": "...", "passed": true, "detail": "..."}]
}
```

报告不包含时间戳，保证可重复。

### 4.7 异常与错误处理

| 场景 | 异常 | 信息 |
| --- | --- | --- |
| 目标精度不支持 | `BuildError` | 目标值、允许值 |
| 元素大小变化 | `BuildError` | 张量名、源/目标精度 |
| 形状契约不满足 | `BuildError` | op 名、输入形状、期望 |
| 未知 op_type | `BuildError` | op 名、类型 |
| 张量形状未解析 | `BuildError` | 张量名、生产 op |
| 不变量失败 | `BuildError` | 不变量名、反例 |
| 配置非法 | `BuildError` | 字段与范围 |

## 5. 实施步骤

| 阶段 | 任务 | 依赖 | 产出 |
| --- | --- | --- | --- |
| P1 | 定义 `OptimizeConfig`、`MemoryPlan` 与报告结构 | common | 数据结构 |
| P2 | 实现 Precision Pass 与单测 | P1 | `precision.py` |
| P3 | 实现 Shape Pass 与单测 | P1 | `shape.py` |
| P4 | 实现 Memory Pass 与重叠校验 | P3 | `memory.py` |
| P5 | 编排 Optimizer 与 CLI | P2-P4 | `optimizer.py` |
| P6 | Qwen3 全图优化与报告 | P5 | `optimize_report.json` |
| P7 | 测试结果与风险记录 | P6 | `tests/TestResults.md` |

风险与应对：

- **形状规则遗漏**：未知 op 直接报错，不静默跳过；新增算子先补规则与用例。
- **全上下文显存爆炸**：默认 `max_num_tokens=256` 作为规划上界并写入文档。
- **复用导致正确性问题**：只复用 `reusable=true` 的 activation；图输入/输出
  永不复用；任何重叠由测试逐一校验。

## 6. 测试与验证设计

### 6.1 测试策略

- 单元测试：三个 Pass 分别覆盖正常、边界与错误路径。
- 集成测试：Qwen3 全图优化，校验形状数量、精度直方图与显存不变量。
- 确定性测试：重复运行逐字段比较。

### 6.2 用例设计

| 编号 | 用例 | 输入 | 通过标准 |
| --- | --- | --- | --- |
| UT-OPT-001 | 精度转换 | 小型 bf16 图 | 目标精度全部为 fp16；字节数不变 |
| UT-OPT-002 | int64 保持 | Qwen3 输入 | `input_ids`/`position_ids` 仍为 int64 |
| UT-OPT-003 | 精度关闭 | `target_precision="keep"` | 精度不变，形状仍推导 |
| UT-OPT-004 | 元素大小变化 | fp32 图转 fp16 | 抛 `BuildError` |
| UT-OPT-005 | gather 形状 | `[1,8]` indices + `[100,16]` weight | 输出 `[1,8,16]` |
| UT-OPT-006 | linear 形状 | `[1,8,16]` × `[32,16]` | 输出 `[1,8,32]` |
| UT-OPT-007 | GQA 校验 | q=16 heads、kv=7 heads | 抛 `BuildError` |
| UT-OPT-008 | RMSNorm 整除 | 末维 10、normalized 4 | 抛 `BuildError` |
| UT-OPT-009 | Add 同形 | 不同 shape | 抛 `BuildError` |
| UT-OPT-010 | 生命周期 | 线性链 a→b→c | birth/last_use 正确 |
| UT-OPT-011 | 复用无重叠 | Qwen3 全图 | 所有重叠区间不相交 |
| UT-OPT-012 | 关闭复用 | `enable_reuse=false` | 偏移单调增长，arena 更大 |
| UT-OPT-013 | 对齐 | 尺寸非 256 倍数 | 所有 offset % 256 == 0 |
| UT-OPT-014 | 确定性 | 连续两次运行 | `to_dict()` 相等 |
| UT-OPT-015 | Qwen3 形状完整性 | Qwen3 图 | 423 个 activation 全部有形状 |
| UT-OPT-016 | 报告 JSON | Qwen3 图 | 可 json 序列化；不变量 passed |

### 6.3 测试环境

- Windows 11 / Linux，Python 3.10+；
- Graph 原生扩展已构建；
- 命令：`PYTHONUTF8=1 python -m unittest Build.optimizer.test_optimizer -v`。

### 6.4 通过标准

- 全部用例通过；
- Qwen3 激活形状完整、显存计划无重叠；
- 报告可被 `json.loads` 解析；
- 测试报告记录实际输出、未覆盖项与风险。
