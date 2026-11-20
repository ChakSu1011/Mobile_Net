# MegaAttention OCL 测试计划

## 1. 目标

- OCL 与 Mock 数值一致（causal GQA）；
- 真实 S=256 性能与 Mock 对比；
- 参与端到端 OCL vs Mock Token 一致性。

## 2. 用例

| 用例 | 形状 | 内容 | 通过标准 |
| --- | --- | --- | --- |
| small/s1 | B=1,S=1,H=4,KV=2,D=16 | 单 key，输出=V | max_abs≤3e-2, cos≥0.999 |
| small/s8 | B=1,S=8,H=4,KV=2,D=16 | causal | 同上 |
| small/noncausal | 不使用（首期 causal=true） | — | — |
| edge/s256d16 | B=1,S=256,H=4,KV=2,D=16 | 长序列 | 同上 |
| real | B=1,S=256,H=16,KV=8,D=128 | 真实 | 同上 + 性能 |
| perf | real | 预热 3、重复 10 | kernel/Mock 时间、FLOPs |

## 3. 环境

vivo V2426A / Mali-G925 / OpenCL 3.0；`backend=ocl` vs `backend=mock`；
NDK arm64；adb push/run。

## 4. 指标

- 数值：max_abs、relative_l2、cosine；S=1 特例；
- 性能：kernel/Mock 时间、加速比、有效 FLOPs（按 2*B*H*S²*D/2 causal）；
- 端到端：同输入 Token 生成一致率。

## 5. 结果记录

实测后生成 `test_results.md`，含命令、实际输出、指标、结论、未覆盖项与风险。
