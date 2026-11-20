# Sample OCL 测试计划

## 1. 目标

- OCL argmax 与 Mock Token 100% 一致（无并列数据）；
- 并列最大值取最小下标；
- 真实 rows=256,V=151936 的性能与 Mock 对比；
- 端到端 OCL vs Mock Token 一致性（Sample 为最后一个算子）。

## 2. 用例

| 用例 | rows/V | 内容 | 通过标准 |
| --- | --- | --- | --- |
| small/default | 8/128 | 随机无并列 | Token 100% 一致 |
| small/tie | 1/3 | {0.9,0.9,0.1} | 返回最小下标 |
| small/tail | 8/125 | V 非 8 倍数 | Token 一致 |
| real | 256/151936 | 真实 logits | Token 一致 + 性能 |
| perf | real | 预热 3、重复 10 | kernel/Mock 时间、带宽 |

## 3. 环境

vivo V2426A / Mali-G925 / OpenCL 3.0；`backend=ocl` vs `backend=mock`；
NDK arm64；adb push/run。

## 4. 指标

- 数值：Token 一致率（100%）；并列下标；
- 性能：kernel/Mock 时间、加速比、有效带宽（rows*V*2 / time）；
- 端到端：同输入 Token 生成一致率（≥95%，目标 100%）。

## 5. 结果记录

实测后生成 `test_results.md`，含命令、实际输出、指标、结论、未覆盖项与风险。
