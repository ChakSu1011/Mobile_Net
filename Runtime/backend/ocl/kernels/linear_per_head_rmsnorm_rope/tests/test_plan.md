# LinearPerHeadRMSNormRope OCL 测试计划

## 1. 目标

- OCL 与 Mock 数值一致（q/k 两种 heads）；
- 阶段 1/2 与总 kernel 时间、Mock 时间对比；
- 参与端到端 OCL vs Mock Token 一致性。

## 2. 用例

| 用例 | 配置 | 内容 | 通过标准 |
| --- | --- | --- | --- |
| small/q | M=8,K=64,N=64,heads=4,D=16 | pos=0 | max_abs≤2e-2, cos≥0.9999 |
| small/k | M=8,K=64,N=32,heads=2,D=16 | pos 非连续 | 同上 |
| edge/pos0 | M=1,pos=0 | 单行 | 同上 |
| edge/posn | pos={0,5,7} | 非连续 position | 同上 |
| real/q | M=256,K=1024,N=2048,heads=16,D=128 | 真实 | 同上 + 性能 |
| real/k | M=256,K=1024,N=1024,heads=8,D=128 | 真实 | 同上 + 性能 |

## 3. 环境

vivo V2426A / Mali-G925 / OpenCL 3.0；`backend=ocl` vs `backend=mock`；
NDK arm64；adb push/run。

## 4. 指标

- 数值：max_abs、relative_l2、cosine；
- 性能：阶段 1、阶段 2、总时间；Mock 时间；加速比；GEMM FLOPs；
- cos/sin 方案（kernel 内 vs 表）与影响；
- 端到端：同输入 Token 生成一致率。

## 5. 结果记录

实测后生成 `test_results.md`，含命令、实际输出、指标、结论、未覆盖项与风险。
