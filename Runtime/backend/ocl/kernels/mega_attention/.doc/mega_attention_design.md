# MegaAttention OCL 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 模块 | `Runtime/backend/ocl/kernels/mega_attention/` |
| 权威位置 | 本文件（含 Kernel 文档章节） |
| 上游文档 | [`ocl_design.md`](../../../.doc/ocl_design.md)、[`backend_design.md`](../../../../.doc/backend_design.md) |
| 算子契约 | `Runtime/backend/ops/mega_attention/mega_attention_spec.cc` |
| Mock 参考 | `Runtime/backend/mock/kernels/mega_attention_kernel.cc` |
| Kernel 文档 | 本文件第 4.3 节 |

## 1. 背景与目标

MegaAttention 实现 causal GQA attention（首期 prefill，`use_kv_cache=false`）：
`out = softmax(Q·Kᵀ·scale + causal_mask) · V`。
Qwen3：B=1，S≤256，H=16，KV=8（GQA ratio 2），D=128；Q `[B,S,2048]`，
K/V `[B,S,1024]`，输出 `[B,S,2048]`。q/k/v 已由上游 `per_head_rmsnorm_rope` 提供。

目标是分块 online softmax Kernel：寄存器分块、half8、GQA K/V 复用、causal 跳过，
并在真实形状上显著优于 Mock 单线程。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 |
| --- | --- |
| FR-AT-01 | 支持 q `[B,S,H*D]`、k/v `[B,S,KV*D]`、out `[B,S,H*D]` FP16 |
| FR-AT-02 | `causal=true`、`merge_heads=true`、`has_output_projection=false`、`use_kv_cache=false` |
| FR-AT-03 | GQA：H % KV == 0；KV head 被 H/KV 个 Q head 复用 |
| FR-AT-04 | softmax_scale 来自 attr；FP32 online softmax |
| FR-AT-05 | 支持 S=1、S=256；D=128（small 可用 D=16） |
| FR-AT-06 | 与 Mock 阈值内一致 |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| NFR-AT-01 | Q tile 寄存器化；K/V 流式；避免完整 score 矩阵落 global |
| NFR-AT-02 | GQA ratio=2 时 K/V 复用；causal 跳过上三角 |
| NFR-AT-03 | 每 workgroup 有足够 query 行，减少 K/V 重复读取 |
| NFR-AT-04 | 真实形状（S=256）性能优于 Mock |

### 2.3 约束

- Mali-G925：12 CU、1.6 GHz、half8、32 KB local（CL_GLOBAL）、2 MB L2、WG≤1024；
- 首期无 Paged KV / Decode；`use_kv_cache=false`；
- FP16 存储、FP32 softmax/累加；禁止 fast-relaxed-math；
- 不 `clFinish`。

### 2.4 验收

- 与 Mock：max_abs ≤ 3e-2、relative_l2 ≤ 1e-2、cosine ≥ 0.999；
- 真实形状性能数据；
- 文档/skill/test_plan 齐全。

## 3. 概要设计

### 3.1 OCL 接入

- `kernel_key="mega_attention"`；`CreateMegaAttentionOclKernel`；
- `Prepare`：校验 H/KV/D/S/causal/scale；编译；选择 query tile；
- `Execute`：绑定 q/k/v/out 的 `cl_mem+offset`；传入 B/S/H/KV/D/scale；计算 grid；enqueue；
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
int batch, int seq, int heads, int kv_heads, int head_dim,
float softmax_scale, int causal
```

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

## 5. 实施步骤

1. `mega_attention.cl` baseline（1 head/WG，QT=8，TPQ=16）；
2. 宿主包装；
3. small 与 Mock 对齐（S=1、S=8、H=4/KV=2、D=16）；
4. 真实 S=256、H=16/KV=8、D=128 性能；
5. GQA pair、QT/KT 调优；
6. 记录 kernel/Mock 时间、加速比、有效带宽/FLOPs；
7. 更新 skill/文档。

## 6. 测试与验证设计

### 6.1 数值

- 用例：S=1/8/256；H/KV=4/2、16/8；D=16/128；causal；
- 与 Mock：max_abs ≤ 3e-2、relative_l2 ≤ 1e-2、cosine ≥ 0.999；
- 额外检查：S=1 时输出=V；causal 上三角不影响输出。

### 6.2 性能

- 预热 3、重复 10、`cl_event` 中位数；
- 记录 kernel/Mock 时间、加速比、有效 FLOPs（2*B*H*S²*D 近似）；
- 真实 S=256。

### 6.3 端到端

所有算子完成后按 `ocl_design.md` 6.6 做同输入 Token 的 OCL vs Mock 一致性。

### 6.4 通过标准

数值、性能、文档/skill/结果齐全。

## 7. 实现状态（P4.3）

- 已实现 `mega_attention` OCL 基线 Kernel：`.cl` + 宿主包装 + `Create<Op>OclKernel`；
- 主机 Intel Arc 上经 `ocl_vs_mock_test` 与 Mock 数值对齐通过；
- 当前为基线实现；性能优化与真实形状误差/端到端 Token 待 P4.5/P4.6。
