# Linear OCL 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 模块 | `Runtime/backend/ocl/kernels/linear/` |
| 权威位置 | 本文件（含 Kernel 文档章节） |
| 上游文档 | [`ocl_design.md`](../../../.doc/ocl_design.md)、[`backend_design.md`](../../../../.doc/backend_design.md) |
| 算子契约 | `Runtime/backend/ops/linear/linear_spec.cc` |
| Mock 参考 | `Runtime/backend/mock/kernels/linear_kernel.cc` |
| Kernel 文档 | 本文件第 4.3 节 |

## 1. 背景与目标

Linear 计算 `y = x · W^T`，W 为 `[N,K]` row-major（transpose_weight=true），x 为
`[B,S,K]` FP16，y 为 `[B,S,N]` FP16。Qwen3 中：
- q/o/v/down/gate/up 等常规 GEMM：K=1024/2048，N=1024/2048/3072；
- lm_head：K=1024，N=151936，输出 `[1,256,151936]`（77.8 MB），是最大单算子。

目标是实现一个可复用的 NT GEMM Kernel：FP16 存储、FP32 累加、寄存器分块、half8 向量化，
并在真实形状上显著优于 Mock 单线程实现。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 |
| --- | --- |
| FR-LN-01 | 支持 `x [..,K]`、`W [N,K]`、`y [..,N]` FP16 |
| FR-LN-02 | `transpose_weight=true`、`bias=false` |
| FR-LN-03 | 任意 M（行数）、K、N；K 以 8 为对齐最优 |
| FR-LN-04 | 支持 N=151936 等大 N；不能依赖单 WG 覆盖 |
| FR-LN-05 | 与 Mock 在阈值内一致 |
| FR-LN-06 | lm_head 输出 77.8 MB 不额外复制 |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| NFR-LN-01 | 寄存器分块，减少全局内存流量 |
| NFR-LN-02 | half8 向量加载；权重复用（同一 W 行被多个 M 行使用） |
| NFR-LN-03 | 大 N 使用 grid-stride / N 分块，避免网格过大 |
| NFR-LN-04 | 真实形状 kernel 时间优于 Mock；GEMM 目标为显著加速 |
| NFR-LN-05 | 不引入额外中间 buffer（除内部可选打包） |

### 2.3 约束

- Mali-G925：12 CU、1.6 GHz、half8、2 MB L2、32 KB local（CL_GLOBAL）；
- 权重 W 在 ConvertParam 中可做 K 维对齐/打包；A 为 row-major；
- FP16 乘、FP32 累加；禁止 fast-relaxed-math；
- 不 `clFinish`。

### 2.4 验收

- 数值阈值：max_abs ≤ 2e-2、relative_l2 ≤ 5e-3、cosine ≥ 0.9999；
- 真实形状有 kernel/Mock 时间与加速比；
- 文档/skill/test_plan 齐全。

## 3. 概要设计

### 3.1 OCL 接入

- `kernel_key="linear"`；`CreateLinearOclKernel`；
- `Prepare`：校验 K/N/M、attr；选择 kernel 变体（寄存器块大小）；编译；
- `Execute`：绑定 A/B/C 的 `cl_mem+offset`；设置 M/N/K；计算 grid；enqueue；
- `Release`。

### 3.2 GEMM 分块

采用 NT GEMM（A row-major，B row-major，C=A·Bᵀ）：

- 一个 workgroup 计算 `BM × BN` 输出块；
- 每个 work-item 计算 `TM × TN` 输出（寄存器累加）；
- 沿 K 维按 `KB` 步长循环，half8 加载 A 与 B；
- 典型参数（待实测）：
  - `BM=16, BN=64, TM=4, TN=4`，WG=(BN/TN=16, BM/TM=4)=64 threads；
  - 或 `BM=32, BN=64, TM=4, TN=4`，WG=128；
- 每个 work-item：4 个 A 行 × 4 个 B 行，16 个 FP32 累加器；
- K 循环内：加载 4 个 A half8、4 个 B half8，做 16 次 `mad`；
- 输出：FP32→FP16 一次写回。

大 N 处理：

- grid 的 x 维为 `ceil(N/BN)`；N=151936/64=2374 个块，y 维 M/BM=16（M=256），
  共 ~38k WGs；Mali 可调度；若网格过大，使用 grid-stride 让每个 WG 处理多个 N 块；
- 对 M=256，优先 `BM=16/32`，让每个输出块被一个 WG 完成。

### 3.3 权重打包

- ConvertParam 时可把 W 的 K 维补齐到 8 的倍数，行首按 128 B 对齐；
- Kernel 以 half8 读 W 行；若 K 非 8 倍数，尾部标量；
- 不转置 W（保持 [N,K]），Kernel 直接按行读，避免额外转置成本；
- 若后续发现 B 行跨步导致 cache 低效，可在 ConvertParam 中转成 tile-major（P4 后评估）。

## 4. 详细设计

### 4.1 接口

```cpp
class LinearOclKernel final : public OclKernel {
 public:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status Execute(const runtime_graph::KernelContext& context) override;
  Status Release() override;
};
```

### 4.2 参数

```text
__global const half* a, int a_offset,     // [M,K]
__global const half* b, int b_offset,     // [N,K]
__global half* c, int c_offset,           // [M,N]
int m, int n, int k
```

### 4.3 Kernel 文档

- 入口：`linear_nt_4x4`（baseline），可选 `linear_nt_8x4`、`linear_nt_4x8`；
- global：`(ceil(N/BN)*BN, ceil(M/BM)*BM)`；local：`(BN/TN, BM/TM)`；
- 每 work-item：
  1. 计算 `row0 = block_y*BM + local_y*TM`，`col0 = block_x*BN + local_x*TN`；
  2. 16 个 `float acc[TM][TN] = 0`；
  3. K 循环 `kk=0; kk<K; kk+=8`：
     - 加载 `a[row0+ti][kk..kk+7]` 4 个 half8；
     - 加载 `b[col0+tj][kk..kk+7]` 4 个 half8；
     - `acc[ti][tj] += dot(a_vec, b_vec)`（FP32）；
  4. 写回 `c[row][col] = half(acc)`；
- 边界：M/N 不是 BM/BN 倍数时，用 `if` 保护，不越界；
- K 非 8 倍数：尾部标量 `for (; kk<K; ++kk)`；
- 精度：乘积 FP16→FP32（`vload_half8` 转 float 或 `mad`），累加 FP32，写回一次舍入；
- 权重：B 行连续，half8 读；A 行按 K 连续；L2 复用 A/B 块。

### 4.4 优化策略

1. baseline：4x4 寄存器块、64/128 threads、half8；
2. 调 BM/BN/TM/TN：在 12 CU、1024 WG 限制下比较；
3. 大 N（lm_head）：N 方向 grid-stride 或 8x4 块，降低网格与调度开销；
4. A/B 的 half8 加载是否对齐；K=1024 时 8 对齐；
5. 若寄存器 spill，降 TM/TN 或 BM/BN；
6. 评估 `cl_khr_subgroup`/`cl_arm_matrix_multiply`（Mali 专有矩阵乘）作为后续优化；
7. 不采用 local 大块（local 为 CL_GLOBAL）；如需 A tile，限制 2–8 KB 并实测；
8. 目标：真实形状 GEMM 达到设备 FP16 吞吐的可用比例，并显著快于 Mock。

### 4.5 错误处理

- Prepare 校验 `K==W[1]`、`N==W[0]`、shape/attr；失败返回错误；
- Program 编译失败落 build log；
- offset/对齐由 OclBuffer/CreateBuffer 保证；
- 未知变体回退 baseline.

## 5. 实施步骤

1. `linear.cl` baseline（4x4）；
2. 宿主包装 + 变体选择；
3. small 与 Mock 对齐；
4. 真实形状（N=1024/2048/3072）性能；
5. lm_head（N=151936）单独测；调 grid-stride/块大小；
6. 记录 kernel/Mock 时间、加速比、FLOPs；更新 skill/文档。

## 6. 测试与验证设计

### 6.1 数值

- 用例：M=8/256，K=64/1024，N=64/1024/2048/3072；随机 FP16；
- 与 Mock 阈值：max_abs ≤ 2e-2、relative_l2 ≤ 5e-3、cosine ≥ 0.9999；
- lm_head 抽样：N=151936，检查数值与性能，不要求全量 CPU 参考（可用 Mock 抽样或 PyTorch 参考）。

### 6.2 性能

- 预热 3、重复 10、`cl_event` 中位数；
- 记录 kernel 时间、Mock 时间、加速比、有效 FLOPs（2*M*N*K/time）；
- 分别报告常规 GEMM 与 lm_head。

### 6.3 端到端

所有算子完成后按 `ocl_design.md` 6.6 做同输入 Token 的 OCL vs Mock 生成一致性。

### 6.4 通过标准

数值阈值、真实形状性能数据、文档/skill/结果齐全。

## 7. 实现状态（P4.3）

- 已实现 `linear` OCL 基线 Kernel：`.cl` + 宿主包装 + `Create<Op>OclKernel`；
- 主机 Intel Arc 上经 `ocl_vs_mock_test` 与 Mock 数值对齐通过；
- 当前为基线实现；性能优化与真实形状误差/端到端 Token 待 P4.5/P4.6。
