# Skill: OCL MegaAttention (Causal GQA Prefill)

## 适用场景

- 实现/优化 causal GQA attention（prefill，无 KV cache）；
- Mali/Adreno 上 tiled online softmax、GQA K/V 复用；
- 与 Mock Attention 数值/性能对比。

## 输入

- 契约：`ops/mega_attention/mega_attention_spec.cc`；
- Mock：`mock/kernels/mega_attention_kernel.cc`；
- 形状：B=1,S≤256,H=16,KV=8,D=128；q/k/v/out FP16；
- 设备：Mali-G925，12 CU，half8，2 MB L2，WG≤1024。

## 步骤

1. `mega_attention.cl` baseline：一 Q head/WG，QT=8，TPQ=16；
2. Q tile 读入寄存器；K/V tile 流式；dot 用 subgroup 归约；
3. online softmax：running max/sum，FP32；acc D 维；
4. causal：key 上限按 query tile；tile 内 mask；
5. 与 Mock 对齐（S=1/8，small H/KV/D）；
6. 真实 S=256 性能；
7. GQA pair（2 Q head/WG）复用 K/V；调 QT/KT；
8. 记录最优参数、时间、FLOPs；更新文档/skill。

## 输出

- `.cl` + 宿主包装；
- 数值阈值内一致；
- kernel/Mock 时间、加速比、有效 FLOPs。

## 常见陷阱

1. causal：key j≤i；按 query tile 限制 key 范围，tile 内 mask；不要全量算再 mask。
2. GQA：KV head = q_head / (H/KV)；ratio=2 时两个 Q head 共享。
3. online softmax 的 `alpha=exp(m_old-m_new)`、`p=exp(s-m_new)`；FP32。
4. 不要物化 S×S score；S=256、H=16 时 score 1M 元素，global 带宽代价大。
5. Q tile 寄存器化；K/V 流式；QT 越大复用越高但寄存器压力越大。
6. 输出 merge_heads：`[B,S,H*D]`；不要输出 `[B,H,S,D]`。
7. D=128 时每 lane 分 8 维（TPQ=16）；half8 读。
8. Mali local 是 CL_GLOBAL；不要用大 local 存 K/V tile。
9. 不要 fast-relaxed-math；softmax/acc FP32。
10. `use_kv_cache=false` 走全量 q/k/v；`true` 走 paged 两段式 kernel
    （见下方 Paged Attention 章节）。
11. P4.5 起 `seq` 是有效长度、`seq_stride` 是静态行跨度；q/k/v/out 地址必须用
    `seq_stride`。把 `seq` 同时当 stride 会在 packed 布局下读错 K/V 行。

## 性能数据（Intel Arc 130T，2026-11-20）

S=256、H=16、KV=8、D=128；warmup 1 + 3 次 `Execute+Synchronize`
平均；基线与优化后交替运行 5 组、取中位数：

| 实现 | kernel+同步时间 | Mock | vs Mock |
| --- | --- | --- | --- |
| P4.3 两遍 softmax | 39.855 ms | 587.705 ms | 14.7x |
| P4.5 单遍 online softmax | 32.687 ms | 587.705 ms | 18.0x |

单遍实现相对两遍约 1.22x；早期非交替单次曾测得 22.19 ms，但交替复测稳定在
32.7 ms 附近，本表采用保守的交替中位数。Mali-G925 已断开，真机数据待重连后补测。

## P4.3 实现状态

- `mega_attention` OCL 基线 Kernel 已实现，并在主机 Intel Arc 上通过 `ocl_vs_mock_test` 与 Mock 数值对齐；
- 真实形状性能见 `ocl/tests/test_results.md` 11.5；多 seed 误差待补。
- 手机 Mali-G925 已断开，历史结果保留。

## Paged Attention（2026-11-21）

**适用场景**：阶段一 KV 推理，`use_kv_cache=true`，需要 chunked prefill、
decode 或混合 batch。

**输入**：
- `AttentionContext`：batch、token_count、kv_block_size、layer_index、KvCacheView；
- `AttentionMetadataView`：seq_lens/query_lens/context_lens/request_indices/
  positions/slots/block_table；
- q/k/v：本步 token，形状仍为静态 `[1,S,...]`。

**步骤**：
1. 宿主 `Prepare` 编译 `paged_kv_store` 与 `mega_attention_paged` 两个 entry；
2. `Execute` 先下发 `paged_kv_store`（token × kv_head），按 slot 写 NHD 池；
3. 再下发 `mega_attention_paged`（token × q_head），按块表读历史 KV；
4. causal 用绝对位置 `kv_pos <= q_pos`，chunked prefill 与 decode 同一公式；
5. GQA 用 `kv_head = q_head / (heads/kv_heads)`；
6. 元数据数组 offset 必须显式传参；KV 池偏移用 `long`。

**输出**：`out[token, head, d]`，只写有效 token 行。

**常见陷阱**：
1. `FP16` 在 Mock/主机侧是 16 位存储位型，必须用 `HalfToFloat` 转换；
   直接 `static_cast<float>(uint16_t)` 会把位型当整数，得到巨大错误值。
2. `slots` 是扁平 token 下标（`block*block_size+offset`），不是块号。
3. 块表尾部补 0，但只读到 `ceil(seq_lens/block_size)`，不要多读。
4. `paged_kv_store` 与 attention 必须在同一 in-order 队列顺序入队；
   跨队列需 event。
5. 不要用全局 `is_prefill` 判断；混合 batch 由每请求的
   `query_lens/context_lens` 表达。
6. 静态 `S` 仍是行跨度上限；`token_count` 才是有效行数。

## Paged 性能优化（2026-11-21）

- prefill（token_count≥8）使用 64 work-item workgroup，按 token/head
  分组，利用 L1 复用相邻 head（GQA pair）的 K/V；decode 保持 local=1，
  避免 4 个 token 被压到一个 CU。
- 显式一个 work-item 处理两个 Q head 的版本会因 2×128 累加器导致私有
  内存/占用率下降，decode 明显回退，不要默认启用。
- 当前 4×256 prompt 基准：prefill 约 110 token/s，decode 约 12.3 token/s。
