# Skill: OCL Gather / Embedding Kernel

## 适用场景

- 实现或优化 token embedding 查表（Gather axis=0）；
- 需要在 Mali/Adreno 上做连续行拷贝、half8 向量化、WG 合并；
- 与 Mock Gather 做数值/性能对比。

## 输入

- 算子契约：`Runtime/backend/ops/gather/gather_spec.cc`；
- Mock 参考：`Runtime/backend/mock/kernels/gather_kernel.cc`；
- 设备能力：`ocl_design.md` 1.1 / `ocl/tests/test_results.md`；
- 形状：`ids [B,S] int64`、`weight [V,H] FP16`、`out [B,S,H] FP16`。

## 步骤

1. 写 `gather.cl`：一 token 一 WG（或按 H 合并 token），half8 load/store；
2. 处理 `H % 8 != 0` 的尾部标量；
3. `gather_kernel.cc` 绑定参数：ids/weight/out 的 `cl_mem+offset`，tokens/vocab/hidden；
4. `Prepare` 校验 dtype/shape，编译 Program；`Execute` 计算 global/local 并 enqueue；
5. small 形状与 Mock 对齐（应逐字节一致）；
6. 真机真实形状测 kernel 时间与带宽；
7. 调 WG/token 合并，记录最优参数与结果；
8. 更新本文档与 `tests/test_results.md`。

## 输出

- `gather.cl`、`gather_kernel.h/.cc`；
- 数值一致（`max_abs=0`）；
- 真实形状 kernel 时间、Mock 时间、加速比、有效带宽；
- 更新后的 skill 与设计文档。

## 常见陷阱

1. `ids` 是 int64（`long`），不是 int32；地址按 8 字节对齐。
2. `Storage.data_ptr` 是 `OclBuffer*`；参数传 `cl_mem` 与字节 offset，不能只传 `cl_mem`。
3. `H` 不一定是 8 的倍数；尾部必须补标量循环。
4. 小 `tokens`/`H` 时 launch 开销主导；不要用 OCL 慢于 Mock 就断定 kernel 差，报告要区分。
5. 不要用 local memory 广播 id；Mali local 是 CL_GLOBAL，直接读同址更简单。
6. 不要 `clFinish`；每 Step 由 Executor 同步。
7. 越界 id 不会自动报错；P4 只测合法输入，若需校验用独立 error buffer，不放进热路径。

## 性能数据

待 P4 真机实测后填写：WG 大小、token/WG、kernel 时间、Mock 时间、加速比、有效带宽。
