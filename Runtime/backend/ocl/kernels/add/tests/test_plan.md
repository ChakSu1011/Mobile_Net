# Add OCL 测试计划

## 1. 目标

- OCL 与 Mock 数值一致；
- 真实形状（262144 元素）kernel 时间与 Mock 对比；
- 明确小形状 launch 开销；
- 参与端到端 OCL vs Mock Token 一致性。

## 2. 用例

| 用例 | 元素数 | 内容 | 通过标准 |
| --- | --- | --- | --- |
| small/tail | 4 | 非 8 倍数 | max_abs≤5e-3 |
| small/1024 | 1024 | 1 WG | 同上 |
| real | 262144 | 真实 | 同上 + 性能 |
| edge/max | 262144 | 大值/次正规 | 无 NaN/Inf |

## 3. 环境

vivo V2426A / Mali-G925 / OpenCL 3.0；`backend=ocl` vs `backend=mock`；
NDK arm64；adb push/run。

## 4. 指标

- 数值：max_abs、relative_l2、cosine；
- 性能：kernel/Mock 时间、加速比、有效带宽（3×元素×2 B）；
- 端到端：同输入 Token 生成一致率。

## 5. 结果记录

实测后生成 `test_results.md`，含命令、实际输出、指标、结论、未覆盖项与风险。
