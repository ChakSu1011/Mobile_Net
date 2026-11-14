# Serializer 模块设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v1.0 |
| 文档状态 | 已实现，随 Build 阶段代码同步 |
| 编写日期 | 2026-11-13 |
| 所属阶段 | Build（离线工具链） |
| 模块目录 | `Build/serializer/` |
| 上游依赖 | `Build/converter`、`Build/optimizer` |
| 下游消费 | Runtime Loader（后续实现）、本模块 Loader（校验用） |
| 权威图设计 | [Graph/.doc/Graph_design.md](../../../Graph/.doc/Graph_design.md) |

## 1. 背景与目标

### 1.1 背景

`Graph` 不提供序列化能力，也不定义文件格式；项目没有 `protoc`，
`Proto/` 目录为空。因此不能沿用原架构文档中的 Protobuf `model.pb` 方案。
Serializer 需要定义一个自描述、可版本化、可确定性生成、并能重建
Graph的模型包格式。

### 1.2 目标

1. 定义 `model.mgf` 二进制容器：固定头 + 规范化 JSON manifest + 可选权重区。
2. 把优化后的 `ModelGraph`、显存计划、权重表与统计信息写入 manifest。
3. 把可用权重按 256 B 对齐打包为 `weights.bin`，记录逐权重 SHA-256。
4. 生成 `compile_report.json`，供人工与 Agent 审查。
5. 实现 Loader：校验文件头、版本与哈希，解析 manifest，并按 manifest
   重建 Graph；重建结果支持 `dump_ir()` 校验。
6. 保证无权重时也能生成 metadata-only 模型包。

### 1.3 非目标

- 不实现 Protobuf schema 或生成代码。
- 不实现运行时设备加载、权重上传与 Kernel 创建。
- 不实现权重格式转换（BF16 字节原样打包）。
- 不实现加密、签名、压缩与分片下载。
- 不实现旧版本迁移工具；首版只有 version 1。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收要点 |
| --- | --- | --- |
| FR-SER-01 | `model.mgf` 必须包含 8 字节 magic、版本号、manifest 长度与 SHA-256、权重长度与 SHA-256 | 头部字段可解析；magic 为 `MNETMGF\0` |
| FR-SER-02 | manifest 必须是规范化 JSON（key 排序、紧凑分隔符、UTF-8） | 同一输入两次生成字节一致 |
| FR-SER-03 | manifest 必须包含完整 `ModelGraph`、显存计划、优化配置与权重表 | Loader 可据此重建图 |
| FR-SER-04 | 权重可用时必须写入 `weights.bin`，每个权重按 256 B 对齐并记录 offset/size/SHA-256 | 偏移对齐；哈希与字节一致 |
| FR-SER-05 | 权重不可用时必须生成 metadata-only 包，不得报错 | manifest 中 `available=false`，`weights.bin` 可省略 |
| FR-SER-06 | `compile_report.json` 必须包含格式版本、图统计、显存摘要、权重可用性与文件哈希 | JSON 可解析 |
| FR-SER-07 | Loader 必须校验 magic、版本、manifest 哈希、权重哈希与边界 | 篡改任一字节即抛 `BuildError` |
| FR-SER-08 | `rebuild_graph()` 必须按 manifest 重建输入、常量、计算 op、输出并调用 `sort_op()` | 重建图的 `dump_ir()` SHA-256 与 manifest 记录一致 |
| FR-SER-09 | 包目录必须只包含预期文件 | 不产生临时文件残留 |
| FR-SER-10 | 序列化必须拒绝未优化的 `unknown` 精度与未解析形状 | 抛 `BuildError` 并指出张量名 |

### 2.2 非功能需求

| 编号 | 类别 | 要求 |
| --- | --- | --- |
| NFR-SER-01 | 确定性 | 同一输入与配置生成逐字节一致的 `model.mgf` 与 `weights.bin` |
| NFR-SER-02 | 完整性 | manifest 与权重区均使用 SHA-256；Loader 默认校验 |
| NFR-SER-03 | 可扩展性 | 头部 version 与 manifest `format_version` 分离；未知可选字段忽略 |
| NFR-SER-04 | 性能 | metadata-only 包生成 < 2 s；哈希为流式读取 |
| NFR-SER-05 | 规范 | Google Style 与中文注释；只依赖标准库 |

### 2.3 约束条件

- 权重字节必须与源 safetensors 完全一致，不做 dtype 转换。
- `weights.bin` 只包含 `weight_available=true` 的常量；tied 权重不重复打包。
- manifest 不包含时间戳，保证确定性；时间戳只写入 `compile_report.json`。
- 重建必须使用记录时的 op 创建顺序，不能使用拓扑序替代。
- 输入/输出张量的 `data_ptr` 不序列化；重建时固定为 0。
- Graph 的 `create_constant` 要求 `dims`、`pitch` 为整数列表。

### 2.4 验收标准

| 编号 | 验收项 | 通过条件 |
| --- | --- | --- |
| AC-SER-01 | 头部可解析 | magic/version/flags/长度/hash 全部正确 |
| AC-SER-02 | 确定性 | 两次生成 `model.mgf` SHA-256 相同 |
| AC-SER-03 | 重建一致 | 未优化图重建后 `dump_ir()` 逐字节一致 |
| AC-SER-04 | 权重正确 | 合成权重包解包后字节与源一致 |
| AC-SER-05 | 错误路径 | 截断、坏 magic、坏版本、坏 hash 均抛 `BuildError` |
| AC-SER-06 | Qwen3 包 | `Build/artifacts/qwen3_0_6b/` 可加载并重建图 |

## 3. 概要设计

### 3.1 总体架构

```text
OptimizeResult ──▶ Serializer ──▶ model.mgf
                        │            ├─ 固定头（112 B）
                        │            ├─ manifest（canonical JSON）
                        │            └─ weights.bin（可选）
                        ├──▶ compile_report.json
                        └──▶ Loader ──▶ Graph.Graph + ModelGraph
```

### 3.2 模块划分

| 文件 | 职责 |
| --- | --- |
| `Build/serializer/__init__.py` | 导出 `serialize_package`、`load_package`、`rebuild_graph` |
| `Build/serializer/format.py` | 头部常量、manifest 组装、字节读写 |
| `Build/serializer/serializer.py` | 包写出、权重打包、报告生成 |
| `Build/serializer/loader.py` | 包读取、校验、图重建 |
| `Build/serializer/__main__.py` | CLI：`inspect` 与 `rebuild` |
| `Build/common/safetensors.py` | safetensors 头信息与原始字节读取 |

### 3.3 文件格式

```text
model.mgf
┌──────────────────────────────────────────────┐
│ magic[8] = "MNETMGF\0"                       │
│ version u32 = 1                              │
│ flags u32（bit0 = 有 weights.bin）            │
│ manifest_offset u64 = 112                    │
│ manifest_length u64                          │
│ manifest_sha256[32]                          │
│ weights_offset u64（无权重时 0）              │
│ weights_length u64                           │
│ weights_sha256[32]                           │
├──────────────────────────────────────────────┤
│ manifest（canonical JSON UTF-8）              │
├──────────────────────────────────────────────┤
│ weights.bin（可选，独立文件）                 │
└──────────────────────────────────────────────┘
```

`weights.bin` 作为独立文件存放，头部记录其长度与哈希；manifest 中的每个
权重条目记录相对偏移、字节数与哈希。

### 3.4 技术选型与取舍

| 决策点 | 选择 | 理由 | 代价 |
| --- | --- | --- | --- |
| 容器 | 固定头 + JSON manifest | 自描述、可 diff、无 protoc 依赖 | 体积大于纯二进制 |
| manifest 编码 | canonical JSON | 确定性、易审查 | 需固定 key 排序与分隔符 |
| 权重文件 | 独立 `weights.bin` | 便于分片与校验 | 多一个文件 |
| 权重读取 | 自实现 safetensors 头解析 | 不依赖 torch/numpy，支持 BF16 | 需维护 8 字节头部逻辑 |
| 图重建 | 按 manifest 重放构图 API | 唯一可行路径；Graph 无反序列化 | 需保证属性类型往返一致 |

## 4. 详细设计

### 4.1 数据结构

```python
@dataclass(frozen=True)
class WeightEntry:
    name: str
    source: str | None
    shape: tuple[int, ...]
    dtype: str
    byte_size: int
    offset: int | None
    sha256: str | None
    available: bool
    reason: str | None = None  # 不可用原因，可用时不写入 manifest

@dataclass(frozen=True)
class ModelPackage:
    path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    weights_path: Path | None
    weights_sha256: str | None
```

### 4.2 接口定义

```python
def serialize_package(
    result: OptimizeResult,
    output_dir: Path,
    *,
    weights_path: Path | None = None,
    overwrite: bool = False,
) -> SerializationResult: ...

def load_package(
    package_dir: Path,
    *,
    verify: bool = True,
) -> ModelPackage: ...

def rebuild_graph(
    package: ModelPackage,
    *,
    verify_ir: bool = True,
) -> Any: ...
```

`SerializationResult` 包含 `package_dir`、`model_file`、`weights_file`、
`report_file`、`manifest_sha256`、`weights_sha256`、`graph_ir_sha256`。

### 4.3 manifest 结构

```json
{
  "format": "MobileNet.ModelPackage",
  "format_version": 1,
  "model": {"name": "qwen3_0_6b", "adapter": "qwen3", "config": {}},
  "graph": {
    "schema_version": 1,
    "name": "qwen3_0_6b",
    "adapter": "qwen3",
    "inputs": ["input.input_ids", "input.position_ids"],
    "constants": ["embedding.weight", "..."],
    "outputs": [{"name": "output.logits", "tensor": "lm_head.out"}],
    "tensors": [],
    "ops": [],
    "metadata": {},
    "ir_sha256": "...",
    "ir_line_count": 744
  },
  "optimization": {
    "config": {"target_precision": "float16", "...": "..."},
    "memory_plan": {"alignment": 256, "arena_bytes": 0, "entries": []}
  },
  "weights": {
    "file": "weights.bin",
    "alignment": 256,
    "total_bytes": 0,
    "entries": [
      {"name": "embedding.weight", "source": "model.embed_tokens.weight",
       "shape": [151936, 1024], "dtype": "bfloat16", "byte_size": 311164928,
       "offset": null, "sha256": null, "available": false}
    ]
  },
  "statistics": {
    "tensors": 735, "ops": 423, "constants": 310, "outputs": 1,
    "activation_tensors": 423, "constant_bytes": 1192099840
  }
}
```

`graph` 内容直接来自 `ModelGraph.to_dict()`，附加 `ir_sha256` 与
`ir_line_count`。`ir_sha256` 由“按该 manifest 重建后的图”计算，保证
Loader 自校验闭环。

### 4.4 序列化流程

1. 校验 `OptimizeResult`：
   - 所有非 int64 张量精度不是 `unknown`；
   - 所有 activation 形状无符号维；
   - 显存计划存在。
2. 深拷贝 `ModelGraph.to_dict()`，写入 manifest 的 `graph` 字段。
3. 用 `rebuild_graph_from_model` 在内存中重建 Graph，计算
   `dump_ir()` 的 SHA-256 与行数，写入 `graph.ir_sha256` /
   `graph.ir_line_count`。
4. 扫描 `weights_path`（safetensors 文件或目录），按 constant 的
   `weight_source` 查找权重：
   - 找到且 dtype/shape 匹配：`available=true`；
   - 未找到或形状不匹配：`available=false`，不报错；
   - 找到但读取失败：抛 `BuildError`。
5. 按常量登记顺序打包可用权重：offset 按 256 B 对齐，逐项计算 SHA-256。
6. 组装 manifest，使用 canonical JSON 编码；计算 manifest SHA-256。
7. 写 `model.mgf`：头部 + manifest；若有可用权重，写 `weights.bin`。
8. 写 `compile_report.json`（含生成时间，non-deterministic）。
9. 重新读取包并调用 `load_package(verify=True)` 自检。

### 4.5 权重读取

`Build/common/safetensors.py` 直接解析 safetensors：

- 前 8 字节 = 小端 u64 头长度 `N`；
- 读取 `N` 字节 JSON，键为张量名，值为
  `{"dtype": "BF16", "shape": [...], "data_offsets": [begin, end]}`；
- 数据区起点 = `8 + N`；`begin/end` 相对于数据区起点；
- 支持目录下多个 `*.safetensors`，按文件名排序；
- 重复键抛 `BuildError`。

dtype 映射：

| safetensors | manifest |
| --- | --- |
| `F16` | `float16` |
| `BF16` | `bfloat16` |
| `F32` | `float32` |
| `I64` | `int64` |
| `I32` | `int32` |
| `I8` | `int8` |
| `U8` | `uint8` |
| `BOOL` | `bool` |

形状与 dtype 必须与常量元数据一致；否则标记不可用。

### 4.6 图重建

1. `ModelGraph.from_dict(manifest["graph"])`。
2. `Graph(name)`。
3. 按 `constants` 顺序调用 `create_constant`：
   - `dims`/`pitch` 必须是整数；
   - `precision`/`layout` 由字符串映射到枚举；
   - `byte_size` 与 `data_ptr=0`。
4. 按 `inputs` 顺序调用 `create_input(name, precision, layout)`。
5. 按 `ops` 记录顺序调用 `create_op`：
   - `op_type` 字符串映射回 OpType；
   - `src_tensors` 按名称映射到已创建的 Tensor；
   - `out_cnt = len(outputs)`；
   - `attrs` 原样传入。
6. 按 `outputs` 顺序调用 `create_output(name, tensor)`。
7. `sort_op()`。
8. 若 `verify_ir`，计算 `sha256(dump_ir())` 并与 manifest `graph.ir_sha256`
   比较；不一致抛 `BuildError`，信息中包含期望哈希与实际哈希。manifest
   不保存 IR 原文，因此不做逐行差异定位。

### 4.7 异常与错误处理

| 场景 | 异常 | 信息 |
| --- | --- | --- |
| 文件不存在 | `BuildError` | 路径 |
| magic 错误 | `BuildError` | 实际前 8 字节 |
| 版本不支持 | `BuildError` | 实际/支持版本 |
| manifest 越界 | `BuildError` | offset/length/文件大小 |
| manifest 哈希不符 | `BuildError` | 期望/实际 |
| 权重哈希不符 | `BuildError` | 文件名 |
| manifest JSON 非法 | `BuildError` | 解析位置 |
| 图重建失败 | `BuildError` | op 名、阶段 |
| IR 哈希不符 | `BuildError` | 期望/实际 SHA-256 |
| 输出目录已存在且未授权覆盖 | `BuildError` | 目录路径 |

### 4.8 性能与内存

- manifest 约 1-2 MB（Qwen3），一次性读入内存。
- 权重打包使用流式读取（1 MiB chunk）与增量 SHA-256。
- metadata-only 包不创建 `weights.bin`。
- Loader 重建图的峰值内存与图规模线性相关。

## 5. 实施步骤

| 阶段 | 任务 | 依赖 | 产出 |
| --- | --- | --- | --- |
| P1 | 定义头部常量、canonical JSON 与哈希工具 | common | `format.py` |
| P2 | 实现 safetensors 读取器与测试 | common | `safetensors.py` |
| P3 | 实现 Serializer 与 metadata-only 包 | P1 | `serializer.py` |
| P4 | 实现 Loader 与图重建 | P1、converter | `loader.py` |
| P5 | 实现权重打包与合成权重测试 | P2、P3 | 权重测试 |
| P6 | CLI 与 Qwen3 模型包生成 | P3、P4 | `Build/artifacts/qwen3_0_6b/` |
| P7 | 测试结果与风险记录 | P6 | `tests/test_results.md` |

风险与应对：

- **属性往返不一致**：用 `dump_ir()` 哈希闭环；发现浮点格式差异时记录并
  修正类型转换。
- **无真实权重**：生成 metadata-only 包并在报告与 manifest 中显式标记。
- **格式演进**：version 与 format_version 分离；Loader 拒绝未知主版本。
- **大文件哈希耗时**：流式读取；当前 1.2 GB 权重可接受。

## 6. 测试与验证设计

### 6.1 测试策略

- 单元测试：头部字段、canonical JSON、哈希、错误路径。
- 往返测试：小型图 → 包 → 加载 → 重建 → `dump_ir()` 一致。
- 权重测试：脚本生成小型 safetensors → 打包 → 解包字节一致。
- 集成测试：Qwen3 → 优化 → 包 → 加载 → 重建 → IR 哈希一致。

### 6.2 用例设计

| 编号 | 用例 | 输入 | 通过标准 |
| --- | --- | --- | --- |
| UT-SER-001 | 头部解析 | 正常包 | magic/version/flags 正确 |
| UT-SER-002 | manifest 哈希 | 篡改 manifest | 抛 `BuildError` |
| UT-SER-003 | 权重哈希 | 篡改 weights.bin | 抛 `BuildError` |
| UT-SER-004 | 截断文件 | 截断头部/manifest | 抛 `BuildError` |
| UT-SER-005 | 版本不符 | version=99 | 抛 `BuildError` |
| UT-SER-006 | metadata-only | 无权重 | 包可加载；`weights.bin` 不存在 |
| UT-SER-007 | 小型图往返 | 合成图 | 重建 `dump_ir()` 一致 |
| UT-SER-008 | 确定性 | 两次序列化 | `model.mgf` SHA-256 相同 |
| UT-SER-009 | 合成权重 | 手动写 safetensors | 打包字节与源一致；offset 对齐 |
| UT-SER-010 | IR 校验失败 | 修改 manifest 图 | 重建后哈希不符抛错 |
| UT-SER-011 | 未优化图 | 含 unknown 精度 | 抛 `BuildError` |
| UT-SER-012 | Qwen3 包 | 全流程 | 包可加载；IR 行数 744；统计 735/423/310 |

### 6.3 测试环境

- Windows 11 / Linux，Python 3.10+；
- Graph 原生扩展已构建；
- 不需要 GPU/torch；合成 safetensors 由测试代码生成；
- 命令：`PYTHONUTF8=1 python -m unittest Build.serializer.test_serializer -v`。

### 6.4 通过标准

- 全部用例通过；
- Qwen3 包可加载、可重建、IR 哈希一致；
- 确定性测试字节一致；
- 测试报告记录实际输出、未覆盖项与风险。
