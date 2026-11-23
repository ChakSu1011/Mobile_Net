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
7. 不要 fast-relaxed-math；当前 Intel 路径为 FP16 乘 + FP16 累加（half FMA）。
8. lm_head 输出 77.8 MB；不要额外复制或转置。
9. 小 M/N 时 launch 开销主导；性能结论以真实形状为准。

## 性能数据（Intel Arc 130T，2026-11-23）

FP16 累加 + half8 FMA，4×4 入口；5 次中位数（ms）：

| 形状 | FP32 累加（4×8） | FP16 累加（4×4） | Mock |
| --- | --- | --- | --- |
| 256×1024×1024 | 约 1.39 | **0.556** | 810.632 |
| 256×1024×3072 | 约 3.65–4.59 | **1.554** | 2464.520 |
| 256×1024×151936（全量 lm_head） | 约 359–382 | 472.258 | 未测（过慢） |

- 全量 lm_head 的 FP16 累加比 FP32 慢约 1.24x；生产 prefill 走
  `linear_nt_rows`（sample_rows），不会计算完整 256 行，需用真实 4 并发
  benchmark 复核；
- 4×8 + FP16 累加会寄存器溢出（约 13.2/36.2 ms），已停用；
- Mali-G925 已断开，真机数据待重连后补测。

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
  仍按静态 M 测量；真实形状性能见上方“性能数据”与
  `ocl/tests/test_results.md` 11.5。

## LM Head 行选择与 4×8 分块（2026-11-21）

- LM Head Linear 带 `sample_rows=true`，使用 `linear_nt_rows` 只算
  `output_slot` 行；`row_count==1` 走单行路径，≤4 行按组复用 B。
- **2026-11-23 修正**：FP16 累加下 4×8 的 32 个 half 累加器 + 12 个
  half8 向量导致寄存器溢出，实测比 4×4 慢 8–10x；常规 Linear 统一使用
  `linear_nt_4x4`，`linear_nt_4x8` 保留源码但不再启用。

## FP16 累加 + FMA（2026-11-23，Intel 优先）

- 累加器改为 `half`，K 维 `fma(half, half, half)`；SiLU/其他 epilogue
  仍在 FP32 中计算；
- half8 直接指针装载需要 16 B 对齐：Buffer 基址由后端保证 128 B，
  `k`/offset 为 8 的倍数时直接 `*(const __global half8*)`，否则
  `vload_half8 + convert_half8` 回退；K 尾部标量 half；
- `vload_half8` 返回 float8；要 half8 运算必须显式 `convert_half8` 或
  对齐指针转换；
- 精度：`ocl_vs_mock_test` max_abs 4.9e-4–2.0e-3、cosine=1；Qwen3-0.6B
  3 条 prompt 的 token 与 golden 一致；长序列/多 seed 仍需补测；
- 4×8 FP16 会溢出；lm_head 全量 M=256 比 FP32 慢约 1.24x，生产
  sample_rows 路径需单独基准。
- 常见陷阱：rows kernel 的输出写回原始行号 `c[row*n+col]`，不要压缩行；
  `sample_rows` 未启用时仍走全量 4×4/4×8 路径。

## 最终结论（2026-11-23 晚，FP16 回退）

- FP16 累加虽快（linear 0.556/1.554 ms），但 KV 长序列 8/20-token 贪心
  出现 token 漂移；单独保留 rope 的 FP16 仍失败。
- 全部回退 FP32 后 `ocl_kv_e2e_test` 的 8/20-token、4 并发、mixed 用例
  全部通过；4×8 主 entry 恢复，保留 half8 装载与 `packed_tokens` 截断。
- 当前默认：FP32 累加；FP16 累加仅作实验记录，不作为生产路径。
- 若继续 FP16，需 partial FP16（每 32/64 K 提升 FP32）或按层混合精度，
  并把 8/20-token KV e2e 加入回归门槛。
