# Linear OCL 测试计划

## 1. 目标

- OCL Linear 与 Mock 数值一致；
- 常规 GEMM 与 lm_head 的真实形状性能与 Mock 对比；
- 参与端到端 OCL vs Mock Token 一致性。

## 2. 用例

| 用例 | M/K/N | 内容 | 通过标准 |
| --- | --- | --- | --- |
| small/default | 8/64/64 | 随机 FP16 | max_abs≤2e-2, cos≥0.9999 |
| small/k1024 | 8/1024/1024 | 常规 | 同上 |
| small/n2048 | 8/1024/2048 | o_proj 形状 | 同上 |
| small/n3072 | 8/1024/3072 | mlp 形状 | 同上 |
| edge/m1 | 1/1024/1024 | 单行 | 同上 |
| edge/ktail | 8/1023/64 | K 非 8 倍数 | 同上 |
| real/n1024 | 256/1024/1024 | 真实 | 同上 + 性能 |
| real/n3072 | 256/1024/3072 | 真实 | 同上 + 性能 |
| real/lm_head | 256/1024/151936 | 大 N | 抽样数值 + 性能 |

## 3. 环境

vivo V2426A / Mali-G925 / OpenCL 3.0；`backend=ocl` vs `backend=mock`；
NDK arm64；adb push/run；记录驱动/温度。

## 4. 指标

- 数值：max_abs、relative_l2、cosine；lm_head 可抽样；
- 性能：kernel 时间、Mock 时间、加速比、有效 FLOPs；
- 端到端：同输入 Token 生成一致率。

## 5. 结果记录

实测后生成 `test_results.md`，含命令、实际输出、指标、结论、未覆盖项与风险。
