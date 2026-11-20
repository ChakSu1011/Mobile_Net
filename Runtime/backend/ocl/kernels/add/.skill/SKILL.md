# Skill: OCL Add Kernel

## 适用场景

- 逐元素加法 `out=a+b`（残差连接）；
- Mali/Adreno 上 half8、grid-stride；
- 与 Mock Add 数值/性能对比。

## 输入

- 契约：`ops/add/add_spec.cc`；
- Mock：`mock/kernels/add_kernel.cc`；
- 典型形状：`[1,256,1024]` FP16（262144 元素）；small 可用 1024；
- 设备：Mali-G925，12 CU，half8。

## 步骤

1. `add.cl`：half8 load；FP32 相加；half8 store；grid-stride；
2. 元素数非 8 倍数时尾部标量；
3. 与 Mock 对齐；
4. 真实形状测 kernel 时间/带宽；
5. 调 WG（128/256）与 half8/half4；
6. 小形状对比 Mock，记录 launch 开销；
7. 更新文档/skill/结果。

## 输出

- `.cl` + 宿主包装；
- 数值阈值内一致；
- kernel/Mock 时间、加速比、有效带宽。

## 常见陷阱

1. FP16 相加应转 FP32 后加、再舍入；直接 half 加法会有额外误差。
2. half8 读写的元素数必须是 8 的倍数；尾部标量。
3. 小形状 launch 开销主导；OCL 可能慢于 Mock，报告要区分。
4. 56 个 add/step；不要在每个 add 里 `clFinish`。
5. `Storage.data_ptr` 是 `OclBuffer*`；传 `cl_mem+offset`。
6. 不要 in-place 写输入；输出独立 Buffer。
7. 不要 fast-relaxed-math。

## 性能数据（Intel Arc 130T，2026-11-20）

262144 个 half 元素；交替 5 组中位数：

| 实现 | kernel+同步时间 | Mock |
| --- | --- | --- |
| add half8 | 0.063 ms | 2.160 ms（约 34x） |

该形状已到 launch 下限（微秒级），时间随驱动/系统负载波动明显。
Mali-G925 已断开，真机数据待重连后补测。

## P4.2 实现状态

- `add` OCL Kernel 已实现并通过主机 Intel Arc 与手机 Mali-G925 基础数值测试；
- 当前主测试环境为本机 Intel Arc；手机已断开，性能/P4.6 待重连后补测；
- 下一步：与 Mock/torch 的完整精度对比（P4.3）、性能调优（P4.5）；
- P4.5：接入 `packed_tokens` 有效 token 截断。
