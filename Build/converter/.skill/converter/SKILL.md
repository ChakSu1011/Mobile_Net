---
name: converter
description: 把模型配置（首期 Qwen3-0.6B）转换为 Graph 可执行静态图，并旁路记录 Optimizer/Serializer 所需的 ModelGraph 元数据；适用于离线构图、权重来源映射与转换报告生成。
---

# Converter：模型到 Graph 与 ModelGraph

## 适用场景

- 已有 `Graph` C++17 静态图与 pybind11 最小构图接口，需要把模型结构
  转换为可执行图。
- 需要在不修改 Graph 对象的前提下，额外记录 shape、precision、权重来源、
  tied embedding 等优化/序列化元数据。
- 需要为新的 Decoder-only 模型族接入 Adapter，并保持 Graph 为唯一图表示。

## 不适用场景

- 精度/Shape 推导与显存规划：属于 Optimizer。
- 模型包序列化、权重打包与加载：属于 Serializer。
- 数值推理、权重内容校验、训练或量化校准。
- 模型结构无法用现有 Graph `OpType` 表达时，应先扩展 Graph，
  不应在 Converter 中静默降级为多个低层 op。

## 输入

1. 模型 `config.json`（Transformers 风格）。
2. Graph Python 扩展 `_graph` 与公共模块 `mobilenet_runtime_graph`。
3. 可选 safetensors 文件或目录；只读取头部索引，不读取数据区。
4. 模型 Builder：首期为 `Models/Qwen3/qwen3_graph_builder.py`。

## 输出

- `ConversionResult`：`graph`（已排序的原生 Graph）、`model_graph`
  （旁路记录的 `ModelGraph`）、`adapter`。
- `model_graph.json`：转换后元数据，供 Optimizer 使用。
- `conversion_report.json`：图计数、IR 哈希、权重映射与问题列表。

## 核心不变量

- **单一可执行图**：只创建 Graph 的 `Graph/Op/Tensor/Storage`，
  `ModelGraph` 是非执行 sidecar。
- **代理透明**：`RecordingGraph` 必须完整转发
  `create_input`、`create_constant`、`create_op`、`create_output`、
  `sort_op`、`dump_ir`，代理图与原生图 IR 逐字节一致。
- **常量是叶子**：常量不进入 `SortOp()` 执行序列；Qwen3 的 310 个常量
  在 IR 的 `constants` 段，不在执行序中。
- **计数口径**：Qwen3 `ModelGraph.ops=423` 是计算 op；IR `ops=426`
  额外包含 2 个 Input op 与 1 个 Output op，不能混用。
- **输入约定**：`input_ids`、`position_ids` 记录为 `[B, S]`、`int64`、
  `row_major`，并清空 pitch。
- **权重来源**：每个常量必须有 Hugging Face 权重名；tied embedding 时
  记录 `lm_head.weight -> embedding.weight` 别名，不创建独立常量。
- **不读权重内容**：`data_ptr` 固定为 0；safetensors 只读头信息，
  dtype/shape 不匹配写入 `weight_issues`。

## 标准流程

1. 读取并校验配置；加载 `mobilenet_runtime_graph`。
2. 创建原生 `Graph`，用 `RecordingGraph` 包装并注入 Builder 的
   `graph_factory`。
3. 复用模型 Builder 构图，不复制层结构；调用 `finish()` 生成
   `ModelGraph`。
4. 补充输入符号形状、常量权重来源、权重可用性与 tied 别名。
5. 写 `model_graph.json` 与 `conversion_report.json`，并校验计数字段。

## 检查清单

- [ ] `RecordingGraph.dump_ir()` 与直接使用 Graph 完全一致。
- [ ] `ModelGraph` 通过引用完整性、无环与输出可达校验。
- [ ] 所有常量都有 `weight_source` 与静态形状。
- [ ] tied / untied 两种配置的常量数量与别名正确。
- [ ] 权重缺失时不报错，全部 `weight_available=false`。
- [ ] 报告中的 IR SHA-256、行数与实际 `dump_ir()` 一致。
- [ ] 输入 pitch 已清空，常量 pitch 保留。
- [ ] 代码符合 Google Style 与中文注释要求。

## 常见陷阱

1. **把 IR `ops=426` 当成计算 op 数**：其中 3 个是 I/O op。
2. **用 torch 读 safetensors**：Build 不依赖 torch；使用
   `Build/common/safetensors.py` 解析头部。
3. **忘记 tied 别名**：图里只有一个 `embedding.weight`，事后无法从图
   推断 `lm_head.weight` 是共享还是缺失，必须在 Converter 记录。
4. **输出命名假设错误**：单输出为 `<op>.out`，多输出为
   `<op>.out.<i>`；RecordingGraph 必须按此校验。
5. **Graph 先于 Tensor 销毁**：Tensor 是非拥有句柄，调用方必须保持
   Graph 存活。
6. **属性不可 JSON 序列化**：`create_op` 的 attr 必须是 bool/int/float/
   str/list/dict；否则在记录阶段报 `BuildError`。
