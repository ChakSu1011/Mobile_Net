# Skill: OCL LinearSwiGlu Kernel

## 适用场景

- 实现/优化 `silu(x·gateᵀ)*(x·upᵀ)` 融合算子；
- 双权重 GEMM 共享 x、SiLU epilogue 融合；
- Mali/Adreno half8 与寄存器分块；
- 与 Mock 对比数值/性能。

## 输入

- 契约：`ops/linear_swi_glu/linear_swi_glu_spec.cc`；
- Mock：`mock/kernels/linear_swi_glu_kernel.cc`；
- 形状：M=B*S=256，H=1024，I=3072；
- 设备：Mali-G925，12 CU，half8，2 MB L2。

## 步骤

1. `linear_swi_glu.cl`：TM=4,TN=2，gate/up 各 8 个 FP32 累加器；
2. K 循环 half8 加载 x/gate/up；`mad` 累加；
3. epilogue：`silu(g)=g/(1+exp(-g))`，`out=silu(g)*u`，FP16 写回；
4. 与 Mock 对齐；
5. 真实形状测 kernel 时间/FLOPs；
6. 调 TM/TN/BM/BI；比较 32 累加器是否 spill；
7. 更新文档/skill/结果。

## 输出

- `.cl` + 宿主包装；
- 数值阈值内一致；
- kernel/Mock 时间、加速比、FLOPs。

## 常见陷阱

1. 32 个 FP32 累加器（gate+up 各 16）容易 spill；优先 16 个（TN=2）。
2. x 复用是融合收益来源；不要把 gate/up 分成两个 kernel。
3. SiLU 用 FP32 `exp`；half 精度不够。
4. gate/up 权重行连续，half8 读；不要转置。
5. H 非 8 倍数尾部标量；M/I 边界保护。
6. Mali local 是 CL_GLOBAL；不要用大 local。
7. 不要 fast-relaxed-math；FP32 累加。
8. 性能对比要同时报告“融合 vs 两个独立 Linear”的收益（可选）。

## 性能数据

待 P4 实测后填写：TM/TN/BM/BI、kernel 时间、Mock 时间、加速比、FLOPs。

## P4.3 实现状态

- `linear_swi_glu` OCL 基线 Kernel 已实现，并在主机 Intel Arc 上通过 `ocl_vs_mock_test` 与 Mock 数值对齐；
- 真实形状性能见 `ocl/tests/test_results.md` 11.5；多 seed 误差待补。
- 手机 Mali-G925 已断开，历史结果保留；
- P4.5：接入 `packed_tokens` 有效 token 截断；
- P4.5：K 维 `vload_half8` 向量化 gate/up 双点积，K 非 8 倍数保留标量尾部；
  `ocl_vs_mock_test` 增加 K=16 向量路径用例。
