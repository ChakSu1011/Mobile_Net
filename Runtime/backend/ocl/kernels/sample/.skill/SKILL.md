# Skill: OCL Sample (Argmax)

## 适用场景

- 对 [B,S,V] logits 做 argmax（V=151936）；
- Mali/Adreno 上大向量归约、half8、确定性 tie-break；
- 与 Mock Sample 的 Token 一致性/性能对比。

## 输入

- 契约：`ops/sample/sample_spec.cc`；
- Mock：`mock/kernels/sample_kernel.cc`；
- 形状：rows=B*S=256，V=151936，FP16 logits，int64 输出；
- 设备：Mali-G925，12 CU，half8，2 MB L2。

## 步骤

1. `sample.cl`：一行一 WG（256 threads），half8 扫描；
2. 局部 max/index；严格 `>` 保留最早下标；
3. subgroup reduce + 跨 subgroup 小组归约；
4. 根线程写 int64；
5. 与 Mock 对齐（含并列用例）；
6. 真实 V=151936 测 kernel 时间/带宽；
7. 评估一行多 WG + 第二阶段；记录；
8. 更新文档/skill/结果。

## 输出

- `.cl` + 宿主包装；
- Token 100% 一致（无并列数据）；
- kernel/Mock 时间、加速比、有效带宽。

## 常见陷阱

1. tie-break：严格 `>`，归约同值取较小 index；否则与 Mock 不一致。
2. half 比较要转 FP32；half 的 -0/+0/NaN 行为需注意。
3. half8 尾部 V 非 8 倍数时标量。
4. 一行一 WG 在 V=151936 时每线程约 593 元素；可一行多 WG + 两阶段，但先测 baseline。
5. 输出 int64（`long`），地址 8 字节对齐。
6. 77.8 MB 读；带宽主导；不要物化中间数组。
7. 不要 `clFinish`；每 Step 由 Executor 同步。
8. 若未来融合 lm_head + argmax，需图/优化器支持，不在本 Kernel 内偷偷改图。

## 性能数据（Intel Arc 130T，2026-11-20）

256 行 × V=151936 argmax；交替 5 组中位数：

| 实现 | kernel+同步时间 | Mock |
| --- | --- | --- |
| sample_argmax（一行一 WG） | 1.644 ms | 63.718 ms（约 39x） |

Mali-G925 已断开，真机数据待重连后补测。

## P4.2 实现状态

- `sample` OCL Kernel 已实现并通过主机 Intel Arc 与手机 Mali-G925 基础数值测试；
- 当前主测试环境为本机 Intel Arc；手机已断开，性能/P4.6 待重连后补测；
- 下一步：与 Mock/torch 的完整精度对比（P4.3）、性能调优（P4.5）；
- P4.5：接入 `packed_tokens` 有效 token 截断。
