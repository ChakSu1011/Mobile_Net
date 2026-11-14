---
name: model-graph
description: 把 Decoder-only Transformer 配置转换为 Graph 可执行静态图，按模型参数路径登记常量、连接融合算子并调用 sort_op；首期实例为 Qwen3-0.6B。
---

# 模型组图：从配置到 Graph 静态图

## 适用场景

- 已有 Graph Python 绑定，需要为具体模型族创建静态图。
- 需要按 Transformers 参数路径登记权重常量，并与图内名称建立可追溯映射。
- 需要复用统一的 Decoder Layer 结构，支持 Qwen3 等 Dense 模型。
- 需要生成可重复的图、IR 摘要和组图测试。

## 不适用场景

- 读取权重内容、数值推理或 kernel 执行。
- 精度/Shape 推导、显存规划、序列化（分别属于 Optimizer、Serializer）。
- MoE、VL、KV Cache、padding mask 或采样逻辑。
- 模型结构无法用现有 Graph `OpType` 表达时，应先扩展 Graph。

## 输入

1. 模型 `config.json`：层数、hidden、head、KV head、head_dim、词表、
   RoPE、dtype、tie_word_embeddings。
2. Graph 公共模块 `mobilenet_runtime_graph`。
3. 可选的 Transformers 参考模型，仅用于测试权重形状；运行时不依赖 torch。

## 输出

- 已调用 `sort_op()` 的 `Graph`。
- `dump_ir()` 稳定的 IR 文本与图摘要。
- 常量形状表与逻辑权重名映射。
- 组图测试结果。

## 核心不变量

- **输入固定**：Qwen3 prefill 图只有 `input.input_ids` 与
  `input.position_ids`，均为 `[B, S]`、`int64`。
- **常量只登记元信息**：`data_ptr=0`，不读权重内容；`reusable=false`。
- **计数口径**：Qwen3 `ModelGraph.ops=423` 是计算 op；IR `ops=426`
  额外包含 2 个 Input op 与 1 个 Output op。
- **结构固定**：embedding → 28 × Decoder Layer → final RMSNorm → LM Head
  → output；每层 15 个 op。
- **Qwen3 特性**：q/k 做 per-head RMSNorm，q/k 做 NeoX RoPE，
  attention 为 causal GQA，MLP 为 SwiGLU。
- **tied embedding**：`tie_word_embeddings=true` 时 LM Head 复用
  `embedding.weight`，不登记独立 `lm_head.weight`。
- **确定性**：常量按 Transformers 参数顺序登记，op/tensor 名称固定，
  重复构建与重复 `dump_ir()` 一致。

## 标准流程

1. 读取并校验配置；非法 dtype、activation、attention_bias 直接报错。
2. 按参数顺序登记常量：embedding、每层 11 个权重、final norm，
   tied 时只登记一个 embedding。
3. 创建两个输入；用 `kGather` 构造 embedding。
4. 逐层构造：attn norm → q/k/v 投影 → q/k norm → RoPE →
   `kMegaAttention` → o_proj → residual → mlp norm →
   `kLinearSwiGlu` → down_proj → residual。
5. 构造 final RMSNorm、LM Head 和 output，调用 `sort_op()`。
6. 输出 IR 摘要；测试图计数、权重形状、tied/untied 与错误路径。

## 检查清单

- [ ] 2 个输入、310 个常量、423 个计算 op、1 个输出。
- [ ] IR `ops` 段为 426，且能解释 3 个 I/O op。
- [ ] 每层 15 个 op，拓扑顺序从 embedding 到 lm_head 连续。
- [ ] 所有常量形状与 Transformers 参考模型一致。
- [ ] tied / untied LM Head 行为正确。
- [ ] 常量 `data_ptr=0`、`reusable=false`，不读取权重。
- [ ] `dump_ir()` 重复调用输出稳定。
- [ ] 代码符合 Google Style 与中文注释要求。

## 常见陷阱

1. **把 IR `ops=426` 当成计算 op 数**：计算 op 是 423。
2. **q/k norm 维度错误**：Qwen3 的 q_norm/k_norm 是 per-head
   RMSNorm，归一化维度是 `head_dim`，不是 hidden。
3. **RoPE 布局错误**：Qwen3 使用 NeoX 风格 `rotate_half`，不是 GPT-J
   的交错布局。
4. **GQA 头数不匹配**：`num_attention_heads` 必须能被
   `num_key_value_heads` 整除。
5. **tied embedding 重复登记**：tied 时图中只有一个 embedding 常量，
   别名关系由调用方记录，不能事后从图恢复。
6. **Tensor 生命周期**：Tensor 是非拥有句柄，必须保持 Graph 存活。
7. **权重常量被当成输入**：常量没有 producer op，是 `SortOp()` 的
   依赖叶子，不进入执行序列。
