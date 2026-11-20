# Skill: OCL RMSNorm Kernel

## 适用场景

- 实现/优化最后一维 RMSNorm（row-wise reduction）；
- Mali/Adreno 上行归约、half8、subgroup 调优；
- 与 Mock RMSNorm 数值/性能对比。

## 输入

- 契约：`ops/rms_norm/rms_norm_spec.cc`；
- Mock：`mock/kernels/rms_norm_kernel.cc`；
- 形状：`x [B,S,H]`、`w [H]`、`out [B,S,H]`，H=1024；
- 设备：Mali-G925，12 CU，32 KB local（CL_GLOBAL），subgroups，half8。

## 步骤

1. `rms_norm.cl`：一行一 WG，half8 读，FP32 平方和；
2. 行内归约：优先 `sub_group_reduce_add`；跨 subgroup 用 4/8 个 float 的 local 数组；
3. 第二遍乘 `scale*w` 写回；weight 常驻缓存；
4. 256 threads（每线程 4 halves）与 128 threads（每线程 8 halves）对比；
5. 可选：2/4 行一 WG，比较 work per WG；
6. 与 Mock 对齐；真机测 kernel 时间；
7. 记录最优 WG、归约方式、带宽；更新文档/skill。

## 输出

- `.cl` + 宿主包装；
- 数值阈值内一致；
- kernel/Mock 时间、加速比、有效带宽。

## 常见陷阱

1. Mali `CL_DEVICE_LOCAL_MEM_TYPE=CL_GLOBAL`；local 数组不一定比 global 快，先测。
2. `rsqrt`/平方和必须 FP32；不要在 half 上累加。
3. 第二次读 x 可能命中 L2（2 MB）；但 H=1024、256 行 = 512 KB，命中概率高；
   不要假设一定不命中，实测。
4. 权重 2 KB 常驻；每个 WG 读同一权重，L2 命中。
5. H 非 8 倍数时尾部标量循环；不要越界读写。
6. WG 必须整除 global；max WG 1024；不要用非 uniform 假设。
7. 不要 `clFinish`；每 Step 由 Executor 同步。

## 性能数据

待 P4 真机实测后填写：WG、线程/行、归约方式、kernel 时间、Mock 时间、加速比、带宽。

## P4.3 实现状态

- `rms_norm` OCL 基线 Kernel 已实现，并在主机 Intel Arc 上通过 `ocl_vs_mock_test` 与 Mock 数值对齐；
- 真实形状性能见 `ocl/tests/test_results.md` 11.5；多 seed 误差待补。
- 手机 Mali-G925 已断开，历史结果保留；
- P4.5：接入 `packed_tokens` 有效 token 截断。
