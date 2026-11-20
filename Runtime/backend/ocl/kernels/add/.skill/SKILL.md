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

## 性能数据

待 P4 实测后填写：WG、向量宽度、kernel 时间、Mock 时间、加速比、带宽。
