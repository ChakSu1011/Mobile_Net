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
10. 首期无 KV cache；`use_kv_cache=true` 应被 OpSpec 拒绝。

## 性能数据

待 P4 实测后填写：QT/KT、heads_per_wg、kernel 时间、Mock 时间、加速比、FLOPs。
