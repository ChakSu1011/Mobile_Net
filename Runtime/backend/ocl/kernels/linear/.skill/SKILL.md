# Skill: OCL NT GEMM (Linear)

## 适用场景

- 实现/优化 `y = x·W^T`（W [N,K] row-major，transpose_weight=true）；
- Mali/Adreno 上 half8 寄存器分块 GEMM；
- 大 N（lm_head N=151936）的 grid 策略；
- 与 Mock Linear 数值/性能对比。

## 输入

- 契约：`ops/linear/linear_spec.cc`；
- Mock：`mock/kernels/linear_kernel.cc`；
- 形状：M=B*S（256）、K=1024/2048、N=1024/2048/3072/151936；
- 设备：Mali-G925，12 CU，1.6 GHz，half8，2 MB L2，max WG 1024。

## 步骤

1. `linear.cl` baseline：BM=16/32、BN=64、TM=TN=4、half8；
2. 每个 work-item 16 个 FP32 累加器；K 循环 half8 加载 A/B，`mad` 累加；
3. 输出一次 FP16 舍入；
4. M/N 非块倍数时加边界保护；K 非 8 倍数尾部标量；
5. 与 Mock 对齐（small）；真实形状测 kernel 时间/FLOPs；
6. 调 BM/BN/TM/TN、WG；对大 N 用 grid-stride 或 8x4；
7. 记录最优参数、时间、加速比；更新文档/skill。

## 输出

- `linear.cl` + 宿主包装；
- 数值阈值内一致；
- 常规 GEMM 与 lm_head 的性能数据。

## 常见陷阱

1. W 是 `[N,K]` row-major；不要按 `[K,N]` 读；NT GEMM 的 B 行是输出通道。
2. `Storage.data_ptr` 是 `OclBuffer*`；传 `cl_mem+offset`。
3. K 可能不是 8 的倍数；尾部标量；权重可在 ConvertParam 补齐到 8 倍数。
4. 大 N 网格过大：`ceil(151936/64)=2374` 个 N 块；用 grid-stride 或更大 BN。
5. 寄存器 spill：16 个 FP32 累加器 + half8 向量；若 spill，降 TM/TN。
6. Mali local 是 CL_GLOBAL；不要用大 local tile 冒充专用 SRAM。
7. 不要 fast-relaxed-math；FP16 乘、FP32 累加。
8. lm_head 输出 77.8 MB；不要额外复制或转置。
9. 小 M/N 时 launch 开销主导；性能结论以真实形状为准。

## 性能数据（Intel Arc 130T，2026-11-20）

warmup 1 + 3 次 `Execute+Synchronize` 平均，3 次进程运行取中位数（ms）：

| 形状 | 优化前（4×4 标量） | P4.5（half8） | Mock | P4.5 vs Mock |
| --- | --- | --- | --- | --- |
| 256×1024×1024 | 2.237 | 1.211 | 821.015 | 678x |
| 256×1024×3072 | 10.129 | 3.543 | 2510.330 | 708x |
| 256×1024×151936 (lm_head) | 843.809 | 337.469 | 未测（过慢） | — |

Mali-G925 已断开，真机数据待重连后补测。

## P4.3 实现状态

- `linear` OCL 基线 Kernel 已实现，并在主机 Intel Arc 上通过 `ocl_vs_mock_test` 与 Mock 数值对齐；
- 真实形状性能见 `ocl/tests/test_results.md` 11.5；多 seed 误差待补。
- 手机 Mali-G925 已断开，历史结果保留。

## P4.5 优化状态（2026-11-20）

- K 维改为 `vload_half8` + FP32 累加；`vload_half8` 返回 **float8**，
  不要声明成 `half8`，否则编译报 incompatible type；
- `row0 + 1 >= m` 时走单行尾部路径，M=1 decode 不再计算 4 行 padding；
- 越界行/列回落到第 0 行/列读取，写回时仍按 `row<m/col<n` 丢弃；
- K 非 8 倍数保留标量尾部；`ocl_vs_mock_test` 增加 K=16 向量路径用例；
- 有效 token 截断（`packed_tokens`）减少 padding 行计算；基准不设该标志，
  仍按静态 M 测量；
- 真实形状性能数据待补入 `ocl/tests/test_results.md`。
