# Skill: OCL Linear + Per-Head RMSNorm + RoPE

## 适用场景

- 实现/优化 Qwen3 Q/K path 融合算子；
- GEMM + per-head RMSNorm + NeoX RoPE；
- Mali/Adreno 上两阶段 kernel 与 cos/sin 表；
- 与 Mock 对比数值/性能。

## 输入

- 契约：`ops/linear_per_head_rmsnorm_rope/per_head_rmsnorm_rope_spec.cc`；
- Mock：`mock/kernels/per_head_rmsnorm_rope_kernel.cc`；
- q：W [2048,1024]，16 heads×128；k：W [1024,1024]，8 heads×128；
- 设备：Mali-G925，12 CU，half8，2 MB L2。

## 步骤

1. 阶段 1 复用 `linear` NT GEMM，输出 scratch；
2. 阶段 2：一 (row,head) 一 WG；D=128；subgroup 归约；
3. per-head RMSNorm：FP32 平方和 + rsqrt + norm 权重；
4. RoPE：pair i 与 i+D/2；theta^(-2i/D)；
   - 先 kernel 内 `sincos` 做 baseline；
   - 再试按 Step S 的 cos/sin 小表（H2D 或 device 生成）；
   - 大表（context×D/2）按需评估 L2 影响；
5. 与 Mock 对齐（q/k 两种 heads）；真实形状测性能；
6. 试单 kernel 融合（一 WG 一 head），与两阶段对比；
7. 更新文档/skill/结果。

## 输出

- 两个 `.cl`（GEMM + epilogue）或一个融合 `.cl`；
- 数值阈值内一致；
- 阶段 1/2 与总时间、Mock 时间、加速比。

## 常见陷阱

1. per-head 归约范围是 D=128，不是整个 N；不要把 N 当归一化维。
2. RoPE 是 pair `(i, i+D/2)`，不是相邻元素；NeoX 布局。
3. `norm` 权重每个 head 共享；D 长度。
4. cos/sin 的 position 是 token 的 position_ids，不是行号；必须读 positions。
5. theta=1e6；exponent `-2i/D`；FP32 三角。
6. 单 kernel 融合需要 head 的完整 128 输出；GEMM 按 N 分块时不易融合；先两阶段。
7. scratch 需要额外 M*N*2 B；用 workspace 或后端私有 buffer，确认生命周期。
8. Mali local 是 CL_GLOBAL；不要用大 local。
9. 不要 fast-relaxed-math；FP32 归约/三角。

## 性能数据

待 P4 实测后填写：阶段 1/2 时间、cos/sin 方案、kernel 时间、Mock 时间、加速比。

## P4.3 实现状态

- `linear_per_head_rmsnorm_rope` OCL 基线 Kernel 已实现，并在主机 Intel Arc 上通过 `ocl_vs_mock_test` 与 Mock 数值对齐；
- 真实形状性能见 `ocl/tests/test_results.md` 11.5；多 seed 误差待补。
- 手机 Mali-G925 已断开，历史结果保留；
- P4.5：接入 `packed_tokens` 有效 token 截断；
- P4.5：per-head GEMM 的 K 维改为 `vload_half8` + FP32 累加，K 非 8 倍数
  保留标量尾部；`ocl_vs_mock_test` 增加 K=16 向量路径用例。
