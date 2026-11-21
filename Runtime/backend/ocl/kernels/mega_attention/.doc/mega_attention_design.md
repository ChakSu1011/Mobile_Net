# MegaAttention OCL 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.2 |
| 模块 | `Runtime/backend/ocl/kernels/mega_attention/` |
| 权威位置 | 本文件（含 Kernel 文档章节） |
| 上游文档 | [`ocl_design.md`](../../../.doc/ocl_design.md)、[`backend_design.md`](../../../../.doc/backend_design.md) |
| 算子契约 | `Runtime/backend/ops/mega_attention/mega_attention_spec.cc` |
| Mock 参考 | `Runtime/backend/mock/kernels/mega_attention_kernel.cc` |
| Kernel 文档 | 本文件第 4.3 节 |

## 1. 背景与目标

MegaAttention 提供两条路径：

1. **全量 prefill**（`use_kv_cache=false`）：`out = softmax(Q·Kᵀ·scale + causal_mask) · V`，
   q/k/v 为完整序列；
2. **Paged attention**（`use_kv_cache=true`）：q/k/v 只含本步 token；先把 K/V
   写入 KV 池，再按块表读取历史 KV，支持 chunked prefill、decode 与混合 batch。

Qwen3：B=1，S≤256，H=16，KV=8（GQA ratio 2），D=128；Q `[B,S,2048]`，
K/V `[B,S,1024]`，输出 `[B,S,2048]`。q/k/v 已由上游 `per_head_rmsnorm_rope` 提供。

全量路径目标是分块 online softmax Kernel：寄存器分块、half8、GQA K/V 复用、
causal 跳过；paged 路径先实现正确性 baseline，再做同样的分块优化。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 |
| --- | --- |
| FR-AT-01 | 支持 q `[B,S,H*D]`、k/v `[B,S,KV*D]`、out `[B,S,H*D]` FP16 |
| FR-AT-02 | `causal=true`、`merge_heads=true`、`has_output_projection=false`；`use_kv_cache=false` 走全量，`true` 走 paged |
| FR-AT-03 | GQA：H % KV == 0；KV head 被 H/KV 个 Q head 复用 |
| FR-AT-04 | softmax_scale 来自 attr；FP32 online softmax |
| FR-AT-05 | 支持 S=1、S=256；D=128（small 可用 D=16） |
| FR-AT-06 | 与 Mock 阈值内一致 |
| FR-AT-07 | Paged：读取 `AttentionMetadataView` 的 7 个数组与 `block_table_stride` | 每个数组按 offset 正确读取 |
| FR-AT-08 | Paged：按 `slots` 把本步 K/V 写入 `KvLayerView` 的 NHD 池 | 跨块、非连续物理块写入正确 |
| FR-AT-09 | Paged：按 `block_table` 与 `seq_lens/context_lens/positions` 计算 causal | chunked prefill 与 decode 同一路径 |
| FR-AT-10 | Paged：混合 batch 逐 token 读取 `request_indices`，不依赖全局 `is_prefill` | 2 prefill + 2 decode 与分开执行一致 |
| FR-AT-11 | Paged：GQA 映射与全量路径一致；KV 池偏移用 64 位 | 大池不溢出 |
| FR-AT-12 | Paged：块表/slot/layer index 越界返回明确错误 | 不越界读写 |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| NFR-AT-01 | Q tile 寄存器化；K/V 流式；避免完整 score 矩阵落 global |
| NFR-AT-02 | GQA ratio=2 时 K/V 复用；causal 跳过上三角 |
| NFR-AT-03 | 每 workgroup 有足够 query 行，减少 K/V 重复读取 |
| NFR-AT-04 | 真实形状（S=256）性能优于 Mock |

### 2.3 约束

- Mali-G925：12 CU、1.6 GHz、half8、32 KB local（CL_GLOBAL）、2 MB L2、WG≤1024；
- 全量路径 `use_kv_cache=false`；paged 路径 `use_kv_cache=true`；
- paged 固定 NHD、`block_size=BackendCapability.attention_block_size`（当前 16）；
- FP16 存储、FP32 softmax/累加；禁止 fast-relaxed-math；
- 不 `clFinish`；paged 两个 kernel 在同一 in-order 队列顺序入队。

### 2.4 验收

- 与 Mock：max_abs ≤ 3e-2、relative_l2 ≤ 1e-2、cosine ≥ 0.999；
- paged 路径在 prefill/decode/chunked/mixed 下与全量 attention 同阈值；
- 真实形状性能数据（全量与 paged 分开记录）；
- 文档/skill/test_plan 齐全。

## 3. 概要设计

### 3.1 OCL 接入

- 全量：`kernel_key="mega_attention"`，`CreateMegaAttentionOclKernel`；
- Paged：`kernel_key="mega_attention_paged"`，`CreateMegaAttentionPagedOclKernel`；
- `Prepare`：校验 H/KV/D/S/causal/scale；paged 额外校验块大小、KV 视图与
  元数据视图；编译；选择 query tile；
- `Execute`：绑定 q/k/v/out 的 `cl_mem+offset`；paged 另绑定 KV 池与 7 个
  元数据数组；传入 B/S/H/KV/D/scale；计算 grid；enqueue；
- `Release`。

### 3.2 分块策略

- 一个 workgroup 处理 `(batch, q_head_pair, query_tile)`；
  - `q_head_pair`：GQA ratio=2 时同时处理 2 个 Q head，共享同一 KV head；
  - `QT`：query tile 行数，例如 8/16；
  - `WG = QT * threads_per_query`；例如 QT=8、每 query 16 线程 → WG=128；
- 每个 work-item 负责一个 query 的 `D/16=8` 维；
- Q tile 载入寄存器/局部；K/V 按 key tile 流式读；
- 对每个 query，维护 FP32 running max/sum 与 D 维累加器；
- causal：key tile 只需到当前 query tile 的最大位置；tile 内按 i 限制。

### 3.3 在线 softmax

对每个 query i：

1. `m = -inf`, `l = 0`, `acc[D] = 0`；
2. 对 key j ≤ i：
   - `s = dot(q_i, k_j) * scale`；
   - `m_new = max(m, s)`；
   - `alpha = exp(m - m_new)`；`p = exp(s - m_new)`；
   - `l = l*alpha + p`；`acc = acc*alpha + p*v_j`；
   - `m = m_new`；
3. `out_i = acc / l`；
- 以 tile 为单位更新，减少 exp 次数；
- FP32 计算，half8 读 q/k/v。

### 3.4 GQA 复用

- Q head h 与 h+1 共享 KV head `h/2`；
- WG 同时处理两个 Q head 的同一 query tile：K/V 读一次，供两个 Q head 使用；
- 寄存器：2 个 Q head × QT queries × D/threads 的 Q 值 + 2 组 acc；需控制 QT/D；
- 若寄存器压力过大，退回一个 Q head/WG（KV 读两次）。

### 3.5 Paged 路径

- 入口 `mega_attention_paged`；`Prepare` 校验 H/KV/D、块大小、KV 视图与
  元数据视图；
- 一次 `Execute` 顺序下发：
  1. `paged_kv_store`：一 work-item 处理 `(token, kv_head)` 的 D 维拷贝，
     地址由 slot 计算；
  2. `mega_attention_paged`：一 work-item 处理 `(token, q_head)`，按块表
     读取历史 K/V，online softmax；
- 混合 batch 不需要分支：每个 token 用 `request_indices/positions/
  context_lens/seq_lens` 自描述；
- causal 用绝对位置比较 `kv_pos <= q_pos`；chunked prefill 与 decode 同一
  公式；
- GQA：`kv_head = q_head / gqa_ratio`，与全量路径一致；
- baseline 不做 query tile/分块归约；后续优化见 4.4。

## 4. 详细设计

### 4.1 接口

```cpp
class MegaAttentionOclKernel final : public OclKernel {
 public:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status Execute(const runtime_graph::KernelContext& context) override;
  Status Release() override;
};
```

### 4.2 参数

```text
__global const half* q, int q_offset,   // [B,S,H*D]
__global const half* k, int k_offset,   // [B,S,KV*D]
__global const half* v, int v_offset,   // [B,S,KV*D]
__global half* out, int out_offset,     // [B,S,H*D]
int batch, int seq, int seq_stride, int heads, int kv_heads, int head_dim,
float softmax_scale, int causal
```

`seq` 是有效长度（causal 上界与 grid 使用），`seq_stride` 是静态图的行跨度
（q/k/v/out 地址计算使用）；二者在 P4.5 有效 token 截断后不再相等。

### 4.3 Kernel 文档

- 入口：`mega_attention_causal_gqa`；
- global：`(ceil(S/QT)*QT, heads/heads_per_wg, batch)` 或 1D 展平；
  - `heads_per_wg=2`（GQA 复用）或 1；
  - local：`(QT * 16)` 或 `(QT * 8)`；
- 每 work-item：`query = local_y`（0..QT-1），`lane = local_x`（0..TPQ-1）；
  - `d0 = lane * (D/TPQ)`；每 lane 处理 `D/TPQ` 维；
- Q tile：每 query 的 `D` 维读入寄存器（或 local）；两个 Q head；
- K/V tile：按 `KT`（如 16/32）加载；每 lane 读 `D/TPQ` 维 half；
- dot：`sum over d in lane slice`；subgroup/局部归约得到 score；
- online softmax 由 lane 0 维护 m/l，广播；acc 每个 lane 维护自己的 D 维片段；
- causal：key tile 上限 `min((q_tile+1)*QT-1, S-1)`；tile 内 `j<=i`；
- 输出：`out[(b*S+i)*H*D + h*D + d]`；half8 写；
- 精度：score/softmax/acc 全 FP32；输出一次 FP16；
- 边界：S=1、非 QT 整倍数、head_dim=16/128；越界保护；
- 性能：Q tile 寄存器化；K/V 流式；GQA 复用；causal 减少约一半 K/V 读。

### 4.4 优化策略

1. baseline：1 Q head/WG，QT=8，TPQ=16，WG=128；
2. GQA pair：2 Q head/WG，K/V 复用；比较寄存器/占用；
3. 调 QT（4/8/16）与 KT（16/32）；QT 越大 K/V 复用越高，但寄存器压力越大；
4. 用 subgroup 归约 dot；避免 local 大数组；
5. causal：按 query tile 限制 key 范围；tile 内 mask；
6. 试 `cl_khr_subgroup_*` 与 `cl_arm_scheduling_controls`；
7. 不物化完整 score；不写中间 score 到 global；
8. 目标：真实 S=256 显著快于 Mock。

### 4.5 错误处理

- Prepare 校验 H/KV/D、causal/merge/projection/KV 标志、scale>0；
- S=0/head_dim=0 拒绝；
- Program 编译失败落 build log；
- 输出 shape/offset 校验。

### 4.6 Paged 接口

```c
__kernel void paged_kv_store(
    __global const half* k, int k_off,
    __global const half* v, int v_off,
    __global half* k_pool, __global half* v_pool,
    __global const long* slots, int slot_off,
    int token_count, int kv_heads, int head_dim);

__kernel void mega_attention_paged(
    __global const half* q, int q_off,
    __global half* out, int out_off,
    __global const half* k_pool, __global half* v_pool,
    __global const int* seq_lens, int seq_off,
    __global const int* query_lens, int query_off,
    __global const int* context_lens, int ctx_off,
    __global const int* request_indices, int req_off,
    __global const int* positions, int pos_off,
    __global const long* slots, int slot_off,
    __global const int* block_table, int bt_off,
    int batch, int token_count, int q_stride, int bt_stride,
    int heads, int kv_heads, int head_dim, int kv_block_size,
    float softmax_scale, int causal);
```

参数说明：

- `q/k/v/out` 的 `*_off` 是缓冲内字节偏移；`q_stride` 是静态行跨度（S=256）；
- 元数据数组的 `*_off` 是元数据缓冲内字节偏移；`bt_stride` 是每请求块表长度；
- `slots` 是扁平 token 下标；KV 池地址 = `slot × kv_heads × head_dim`；
- 池内偏移在 Kernel 中用 `long`；元数据偏移用 `int`；
- 宿主把 `KvLayerView.k_buffer/v_buffer` 包装为 `TensorView`
  （`data=Buffer*`、`offset=0`、`bytes=num_blocks×block_stride`、
  `device=kDevice`）后调用 `SetMemArg`，再单独传入偏移参数。

### 4.7 Paged 算法

```text
t = get_global_id(0) / heads        // packed token 下标
h = get_global_id(0) % heads
req = request_indices[t]
q_pos = positions[t]
seq = seq_lens[req]
kv_h = h / (heads / kv_heads)

m = -inf; l = 0; acc[D] = 0
for b in 0..ceil(seq / kv_block_size)-1:
  block = block_table[req * bt_stride + b]
  for j in 0..kv_block_size-1:
    kv_pos = b * kv_block_size + j
    if kv_pos >= seq: break
    if kv_pos > q_pos: continue
    slot = block * kv_block_size + j
    k_row = k_pool + slot * kv_heads * head_dim + kv_h * head_dim
    v_row = v_pool + slot * kv_heads * head_dim + kv_h * head_dim
    s = dot(q[t,h], k_row) * softmax_scale
    m_new = max(m, s)
    alpha = exp(m - m_new)
    p = exp(s - m_new)
    l = l * alpha + p
    acc = acc * alpha + p * v_row
    m = m_new
out[t,h] = acc / l
```

边界：

- `seq` 可小于 `context_lens + query_lens` 的理论值时必须拒绝（Executor/OpEnv
  已校验）；
- 块表尾部补 0，但 Kernel 只读到 `ceil(seq/kv_block_size)`，不会读无效块；
- `q_pos` 必须落在 `[0, seq-1]`；否则返回错误（由 Prepare/Executor 校验）。

### 4.8 Paged 错误处理

- `Prepare` 校验 `kv_block_size > 0`、`block_table_stride > 0`、各元数据
  视图非空且 dtype 正确；
- KV 视图的 `num_kv_heads/head_dim/dtype` 必须与 attr 一致；
- `layer_index` 越界、`block_stride` 与 `kv_block_size` 不一致时拒绝；
- Kernel 内不做静默截断；越界由执行前校验拦截。

## 5. 实施步骤

1. `mega_attention.cl` baseline（1 head/WG，QT=8，TPQ=16）；
2. 宿主包装；
3. small 与 Mock 对齐（S=1、S=8、H=4/KV=2、D=16）；
4. 真实 S=256、H=16/KV=8、D=128 性能；
5. GQA pair、QT/KT 调优；
6. 记录 kernel/Mock 时间、加速比、有效带宽/FLOPs；
7. **Paged（P4.8）**：
   1. `paged_kv_store` + `mega_attention_paged` baseline；
   2. 与 Mock paged/全量 attention 对齐（prefill/decode/chunked/mixed）；
   3. chunked prefill、4 并发、OOM 抢占端到端；
   4. 记录 paged kernel 时间与后续优化点；
   5. 更新 skill/文档。

## 6. 测试与验证设计

### 6.1 数值

- 用例：S=1/8/256；H/KV=4/2、16/8；D=16/128；causal；
- 与 Mock：max_abs ≤ 3e-2、relative_l2 ≤ 1e-2、cosine ≥ 0.999；
- 额外检查：S=1 时输出=V；causal 上三角不影响输出。
- Paged 用例：
  - 单块 prefill：seq=1/8/16/17/40/256；
  - chunked：context=16+chunk=16；context=15+chunk=17；
  - decode：context=1..256；
  - 混合 batch：2 个 prefill chunk + 2 个 decode；
  - 非连续物理块表、非 block 整数倍 seq；
- Paged 通过标准：与全量 attention 的 max_abs/relative_l2/cosine 同上；
  块表/slot/layer index 越界均被拒绝。

### 6.2 性能

- 预热 3、重复 10、`cl_event` 中位数；
- 记录 kernel/Mock 时间、加速比、有效 FLOPs（2*B*H*S²*D 近似）；
- 真实 S=256。

### 6.3 端到端

所有算子完成后按 `ocl_design.md` 6.6 做同输入 Token 的 OCL vs Mock 一致性。

### 6.4 通过标准

数值、性能、文档/skill/结果齐全。

## 7. 实现状态（P4.3 / P4.5 / P4.8）

- 已实现 `mega_attention` OCL 基线 Kernel：`.cl` + 宿主包装 + `Create<Op>OclKernel`；
- 主机 Intel Arc 上经 `ocl_vs_mock_test` 与 Mock 数值对齐通过；
- P4.5 优化：单遍 online softmax（历史累加按 alpha 缩放，score 只算一次）；
  有效 token 截断把 `seq`（有效长度）与 `seq_stride`（静态行跨度）分离，
  `ocl_contract_test` 覆盖 S=8/有效 S=3 的前缀一致性与 padding 不写；
- 实测（Intel Arc 130T，S=256/H=16/KV=8/D=128，交替 5 组中位数）：
  两遍实现 39.855 ms → 单遍 32.687 ms（1.22x）；Mock 587.705 ms，
  当前实现约为 Mock 的 18x；早期非交替单次曾测得 22.19 ms，不作为结论；
- 当前仍为一 query 一 work-item 的基线；寄存器分块/GQA 复用待下一轮 P4.5。
- P4.8（Paged）已实现：`paged_kv_store` + `mega_attention_paged`；
  4 个 prompt 单请求/4 并发生成 8 token 与 PyTorch 金标准一致，详见
  `Runtime/backend/ocl/tests/test_results.md` 第 12 节；性能 baseline 待补。

## 8. 2026-11-21 优化（query tile / K/V 复用）

- Paged prefill（`token_count>=8`）使用 **64 work-item workgroup**：相邻
  token/head（含 GQA head pair）在 L1 中共享 K/V 行；decode 保持
  `local=1`，把 64 个 work-item 分散到更多 CU。
- 实测（4 并发 × 256 prompt × 8 decode）：
  - prefill 3.24 s → **2.32 s / 256 token**（79.1 → 110.4 token/s）；
  - decode 0.339 s → **0.324 s / 4 token**（11.8 → 12.3 token/s）。
- 显式 GQA head pair（一个 work-item 处理 2 个 Q head、共享 K/V 读取）
  曾实现并实测：因 2×128 FP32 私有累加器导致占用率下降，decode
  退化到 0.73–0.81 s/step，已回退，不作为默认实现。
- 当前 query tile 通过 workgroup 分组复用 L1 K/V，不增加每 work-item
  私有内存；后续可进一步做 workgroup-local K/V tile + barrier 的显式复用。
