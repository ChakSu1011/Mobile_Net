# LinearSwiGlu OCL 测试计划

## 1. 目标

- OCL 与 Mock 数值一致；
- 真实形状（M=256,H=1024,I=3072）kernel 时间与 Mock 对比；
- 评估融合相对两个独立 Linear 的收益；
- 参与端到端 OCL vs Mock Token 一致性。

## 2. 用例

| 用例 | M/H/I | 内容 | 通过标准 |
| --- | --- | --- | --- |
| small/default | 8/64/128 | 随机 FP16 | max_abs≤2e-2, cos≥0.9999 |
| small/edge | 1/1024/3072 | 单行 | 同上 |
| edge/htail | 8/1023/128 | H 非 8 倍数 | 同上 |
| real | 256/1024/3072 | 真实 | 同上 + 性能 |
| perf | real | 预热 3、重复 10 | kernel/Mock 时间、FLOPs |

## 3. 环境

vivo V2426A / Mali-G925 / OpenCL 3.0；`backend=ocl` vs `backend=mock`；
NDK arm64；adb push/run。

## 4. 指标

- 数值：max_abs、relative_l2、cosine；
- 性能：kernel 时间、Mock 时间、加速比、有效 FLOPs；
- 端到端：同输入 Token 生成一致率。

## 5. 结果记录

实测后生成 `test_results.md`，含命令、实际输出、指标、结论、未覆盖项与风险。
