---
name: build-stage-pipeline
description: 把模型 config（首期 Qwen3-0.6B）经 Converter → Optimizer → Serializer 固化为 Graph_Fixed 模型包 model.mgf，并用加载重建与 IR SHA-256 自校验；适用于离线构图、精度/形状/显存规划、权重打包与模型包交付。
---

# Build 阶段流水线：Convert → Optimize → Serialize

## 适用场景

- 已有 Graph_Fixed（C++17 runtime 静态图）与 Python 最小构图绑定，需要把
  模型结构转换为可执行的 `Graph`，同时记录优化/序列化所需的元数据。
- 需要在不链接 PyTorch/LibTorch 的前提下完成精度推导、Shape 推导、
  激活显存复用规划与模型包打包。
- 需要生成可被 Runtime Loader 重建的 `model.mgf`，并验证“重建图 IR 哈希
  与 manifest 记录一致”。
- 需要给新模型族（非 Qwen3）接入同一套 Build 流程。

## 不适用场景

- 需要执行真实推理或数值对齐：Build 阶段只构图、规划与打包，不创建
  OpEnv/Kernel，不做 Dummy Run。
- 需要训练、量化校准或 INT8/INT4 权重生成：当前只有精度扩展点。
- 需要通用 ONNX/Protobuf 导出：本项目使用自定义 `model.mgf`。
- 模型结构无法用现有 `OpType` 表达：应先扩展 Graph_Fixed 的 OpType 与
  运行时 OpEnv/Kernel，而不是在 Converter 中静默降级。

## 输入

1. 模型 `config.json`（Transformers 风格），首期
   `Models/Qwen3/config.json`。
2. Graph_Fixed Python 扩展（`Build/graph_fixed/python/`）。
3. 可选的 safetensors 权重文件或目录；不提供时生成 metadata-only 包。
4. 优化配置：`target_precision`、`max_num_tokens`、`context_length`、
   `alignment`、`enable_reuse`。
5. 设计文档：六段式（背景与目标、需求分析、概要设计、详细设计、实施步骤、
   测试与验证设计）。

## 输出（四件套）

1. 设计文档：总架构文档对应章节 + converter/optimizer/serializer 模块文档。
2. 代码：`Build/common`、`Build/converter`、`Build/optimizer`、
   `Build/serializer`、`Build/pipeline.py`。
3. 测试报告：`Build/TestResults.md`，含环境、命令、实际输出、结论、
   未覆盖项与风险。
4. 本 skill。

模型包产物：

- `model.mgf`：112 字节固定头 `<8sIIQQ32sQQ32s>` + 规范化 JSON manifest；
- `weights.bin`：可选，按 256 B 对齐，独立 SHA-256；
- `compile_report.json`：构建摘要、IR 哈希、显存计划与自检结果；
- work 目录：`converter/`、`optimizer/` 的 `model_graph.json` 与报告。

## 核心不变量

- **单一可执行图**：只有 Graph_Fixed 的 `Graph/Op/Tensor/Storage`；
  `ModelGraph` 是非执行 sidecar，不得被当作第二套图。
- **RecordingGraph 透明**：代理必须透传 `create_input`、`create_constant`、
  `create_op`、`create_output`、`sort_op`、`dump_ir`；代理图与原生图
  `dump_ir()` 逐字节一致。
- **常量是叶子**：常量不进入 `SortOp()` 执行序列；Qwen3 的 310 个常量
  在 IR 的 `constants` 段，不在 `ops` 执行序中。
- **I/O op 计数**：Qwen3 `ModelGraph.ops=423`（计算 op），IR `ops` 段为
  426（额外 2 个 Input op + 1 个 Output op）；不要混用两个口径。
- **精度只做等宽转换**：BF16→FP16 允许（2 B→2 B）；FP32→FP16 改变元素
  大小时必须报错，禁止静默截断。
- **Shape 先于 Memory**：Memory Pass 入口必须拒绝未解析 activation 形状，
  不允许在符号维上做显存规划。
- **显存不重叠**：生命周期重叠的张量 `[offset, offset+size)` 不得相交；
  所有 offset 按 alignment 对齐；图输入与常量不进 arena，图输出不可复用。
- **序列化不改字节**：Serializer 只打包权重原始字节，不做 dtype 转换；
  因此源权重 dtype 必须与优化后图精度一致。
- **IR 哈希闭环**：manifest 记录重建图 `dump_ir()` 的 SHA-256 与行数；
  Loader 重建后必须复算并比对。
- **确定性**：相同输入生成字节一致的 `model.mgf`；manifest 用 canonical
  JSON，权重按常量顺序与固定对齐打包。

## 标准流程

### 1. 确认图契约

- 阅读 `Graph_Fixed/Graph设计文档.md`，确认 `create_input`、
  `create_constant`、`create_op`、`create_output`、`sort_op`、
  `GetOrderlyOPList()` 的签名与语义。
- 确认输出命名规则：`out_cnt == 1` 为 `<op>.out`，否则为
  `<op>.out.<i>`。
- 确认常量 `pitch`、`precision`、`layout`、`byte_size` 的记录方式，
  避免 Shape Pass 误清常量 pitch。

### 2. Converter

- 用 `RecordingGraph` 包装原生 `Graph`，在调用原生 API 的同时记录
  `TensorMeta`/`OpMeta`/`IOBinding`。
- 复用模型 Builder（如 `Models/Qwen3/qwen3_graph_builder.py`），通过
  `graph_factory` 注入代理，不复制层结构。
- 记录每个常量的 Hugging Face 权重名、shape、dtype 与来源；
  `tie_word_embeddings=true` 时记录 `tied_weights`，不创建独立 LM Head 常量。
- 输入统一注记 `[B,S]`、`int64`、`row_major`，清空 pitch。
- 可选 safetensors 探测只读头部（8 字节头长度 + JSON 头），不读数据区，
  不依赖 torch/numpy；dtype/shape 不匹配写入 `weight_issues`。

### 3. Optimizer

- Precision Pass：按目标精度转换浮点张量；输入保持 `int64`；未知精度从
  生产 op 的 `precision` 属性继承；元素大小变化报错。
- Shape Pass：把 `B`、`S` 绑定为规划值，按 op 规则推导输出形状；
  校验 linear K 维、GQA 比例、RMSNorm 整除、Add 同形；标记输出不可复用。
- Memory Pass：按拓扑序计算 `birth`/`last_use`；输出绑定视为下标 `N`
  的消费者；按 `(birth, -size, name)` 排序做 first-fit 复用；校验对齐、
  重叠、峰值与常量字节。
- Report：输出精度/形状/显存/生命周期与不变量检查结果，字段必须可 JSON
  序列化且确定。

### 4. Serializer

- 校验 `OptimizeResult`：精度已确定、activation 形状无符号维、显存计划存在。
- 用优化后的 `ModelGraph` 重建 Graph_Fixed 图，计算 `dump_ir()` 的
  SHA-256 与行数，写入 manifest。
- 扫描常量与可选权重，生成 `WeightEntry`：不可用时写 `reason`
  （`weights not supplied`、`source tensor not found`、`shape mismatch`、
  `dtype mismatch`、`byte_size mismatch`）。
- 按常量顺序把可用权重写入 `weights.bin`，每个张量 256 B 对齐，
  记录 offset 与单张量 SHA-256。
- 写 `model.mgf`（头 + manifest）与 `compile_report.json`，随后重新加载
  自检 manifest 哈希与权重哈希。

### 5. 流水线串联

```bash
PYTHONUTF8=1 python -m Build.pipeline
```

自定义参数：

```bash
PYTHONUTF8=1 python -m Build.pipeline \
  --config Models/Qwen3/config.json \
  --output Build/artifacts/qwen3_0_6b \
  --work-dir Build/artifacts/qwen3_0_6b_work \
  --weights /path/to/safetensors \
  --target-precision bfloat16 \
  --max-num-tokens 256 \
  --context-length 40960
```

流水线最后一步必须执行 `load_package(verify=True)` 与
`rebuild_graph(verify_ir=True)`；任一失败即构建失败。

### 6. 测试与交付

```bash
PYTHONUTF8=1 python -m unittest Build.converter.test_converter -v
PYTHONUTF8=1 python -m unittest Build.optimizer.test_optimizer -v
PYTHONUTF8=1 python -m unittest Build.serializer.test_serializer -v
PYTHONUTF8=1 python -m unittest Build.test_pipeline -v
PYTHONUTF8=1 python -m unittest discover -s Models/Qwen3 -p 'test_*.py' -v
```

测试报告必须写实际输出，不得只写“测试通过”。

## 检查清单

- [ ] `RecordingGraph.dump_ir()` 与原生 `Graph.dump_ir()` 逐字节一致。
- [ ] `ModelGraph` 的 inputs/constants/ops/outputs 与 IR 计数口径已解释。
- [ ] 每个常量都有 HF 权重名；tied LM Head 记录别名。
- [ ] 所有 activation 有静态形状；无符号维进入 Memory Pass。
- [ ] 精度转换后 `byte_size` 与元素大小一致；非法转换报错。
- [ ] 生命周期重叠张量字节区间不相交；offset 全部对齐。
- [ ] `model.mgf` 头字段（magic/version/flags/offset/length/hash）自洽。
- [ ] manifest 是 canonical JSON；两次构建字节一致。
- [ ] 重建图 IR SHA-256 与行数和 manifest 一致。
- [ ] 无权重时生成 metadata-only 包且不残留 `weights.bin`。
- [ ] 包目录只含 `model.mgf` 与 `compile_report.json`，中间产物在 work 目录。
- [ ] 全部代码通过 Google Style 与中文注释要求。

## 常见陷阱

1. **把 IR `ops=426` 当成计算 op 数**：其中 3 个是 I/O op；
   `ModelGraph.ops` 是 423。测试断言要写清口径。
2. **用 torch 读 safetensors**：Build 不依赖 torch。用
   `Build/common/safetensors.py` 直接解析头部，否则会在无 PyTorch 环境失败，
   且 BF16 无法用 numpy 直接表达。
3. **Serializer 期待 BF16 源权重**：默认优化目标是 FP16。直接给官方 BF16
   权重会得到 `dtype mismatch` 且全部不可用；改用
   `--target-precision bfloat16`，或先将权重转换为 FP16。
4. **忘记清输入 pitch**：Shape Pass 只应清输入 pitch；常量 pitch 必须保留，
   否则重建 Graph 时 `create_constant` 的 pitch 与原始图不一致，IR 哈希漂移。
5. **报告里写时间戳**：serializer 的 `model.mgf` 和确定性测试要求字节稳定；
   时间戳只能放在 `compile_report.json`，不能进入 manifest。
6. **覆盖非空输出目录**：`serialize_package` 默认拒绝；流水线显式传
   `overwrite=True`。手工调用时不要误删用户目录。
7. **只比较 manifest 哈希**：IR 哈希必须由“重建图”计算，不能复用转换期
   缓存值，否则无法发现重建路径的图漂移。
8. **metadata-only 包当成可用模型**：`weights_available=0` 的包只能验证
   图、显存计划与加载流程，不能用于推理。
9. **在 `__init__.py` 之外做再导出**：本仓库用 `Build/common/__init__.py`
   统一再导出；新增公共符号时同步更新，避免循环导入。
10. **Windows 编码**：含中文的 JSON/日志统一 `PYTHONUTF8=1`，manifest
    使用 UTF-8 canonical JSON。
