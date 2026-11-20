# RMSNorm OCL 测试计划

## 1. 目标

- OCL RMSNorm 与 Mock 数值一致；
- 真实形状（256×1024）kernel 时间与 Mock 对比；
- 参与端到端 OCL vs Mock Token 一致性。

## 2. 用例

| 用例 | shapes | 内容 | 通过标准 |
| --- | --- | --- | --- |
| small/default | B=1,S=8,H=64 | eps=1e-6 | max_abs≤2e-2, cos≥0.9999 |
| small/s1 | B=1,S=1,H=1024 | 单行 | 同上 |
| small/eps0 | B=1,S=8,H=1024 | eps=0 | 同上 |
| small/tail | H=65 | 非 8 倍数 | 同上 |
| real | B=1,S=256,H=1024 | 真实形状 | 同上 + 性能 |
| perf | real | 预热 3、重复 10 | 记录中位数 |

## 3. 环境

vivo V2426A / Mali-G925 / OpenCL 3.0；`backend=ocl` vs `backend=mock`；
NDK arm64 构建；adb push/run。

## 4. 指标

- 数值：max_abs、relative_l2、cosine；
- 性能：kernel 时间、Mock 时间、加速比、有效带宽（读 x 两次 + 写 out）；
- 端到端：同输入 Token 生成一致率。

## 5. 结果记录

实测后生成 `test_results.md`，包含环境、命令、实际输出、指标、结论、未覆盖项与风险。
