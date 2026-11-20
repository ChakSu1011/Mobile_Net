# LinearSwiGlu OCL 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 模块 | `Runtime/backend/ocl/kernels/linear_swi_glu/` |
| 权威位置 | 本文件（含 Kernel 文档章节） |
| 上游文档 | [`ocl_design.md`](../../../.doc/ocl_design.md)、[`backend_design.md`](../../../../.doc/backend_design.md) |
| 算子契约 | `Runtime/backend/ops/linear_swi_glu/linear_swi_glu_spec.cc` |
| Mock 参考 | `Runtime/backend/mock/kernels/linear_swi_glu_kernel.cc` |
| Kernel 文档 | 本文件第 4.3 节 |

## 1. 背景与目标

LinearSwiGlu 计算 `y = silu(x·gateᵀ) * (x·upᵀ)`。Qwen3 每层 MLP 使用：
x `[B,S,H]`（H=1024），gate/up 均为 `[I,H]`（I=3072），输出 `[B,S,I]`。
两个 GEMM 共享 x，适合在一个 Kernel 中融合，并 epilogue 融合 SiLU 与乘法。

目标：与 Mock 数值一致；真实形状 kernel 时间优于 Mock；沉淀双权重 GEMM/SwiGLU 融合方法。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 |
| --- | --- |
| FR-SG-01 | 支持 `x [..,H]`、gate/up `[I,H]`、out `[..,I]` FP16 |
| FR-SG-02 | `activation=silu`、`transpose_weight=true`、`bias=false` |
| FR-SG-03 | 两个 GEMM 共享 x；输出一次写回 |
| FR-SG-04 | 与 Mock 在阈值内一致 |
| FR-SG-05 | 支持 I=3072、H=1024 与 small 形状 |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| NFR-SG-01 | x 复用；gate/up 权重按行 half8 读 |
| NFR-SG-02 | 寄存器分块控制累加器数量；避免 32 个累加器 spill |
| NFR-SG-03 | SiLU 在 FP32 计算，FP16 写回 |
| NFR-SG-04 | 真实形状性能优于 Mock |

### 2.3 约束

- Mali-G925：12 CU、half8、2 MB L2、32 KB local（CL_GLOBAL）；
- 仅 FP16；FP32 累加；禁止 fast-relaxed-math；
- 不 `clFinish`。

### 2.4 验收

- 与 Mock `max_abs ≤ 2e-2`、relative_l2 ≤ 5e-3、cosine ≥ 0.9999；
- 真实形状有 kernel/Mock 时间与加速比；
- 文档/skill/test_plan 齐全。

## 3. 概要设计

### 3.1 OCL 接入

- `kernel_key="linear_swi_glu"`；`CreateLinearSwiGluOclKernel`；
- `Prepare`：校验 I/H/attr；选择变体；编译；
- `Execute`：绑定 x/gate/up/out 的 `cl_mem+offset`；设置 M/H/I；enqueue；
- `Release`。

### 3.2 Kernel 分解

- 一个 workgroup 计算 `BM × BI` 输出块（I 方向）；
- 每个 work-item 计算 `TM × TN` 个输出，同时对 gate 与 up 各维护一组累加器；
- 为控制寄存器：`TM=2/4, TN=2/4`；gate/up 共 `2*TM*TN` 个 FP32 累加器；
  - `TM=4,TN=2`：16 个累加器；
  - `TM=4,TN=4`：32 个，可能 spill；仅在小块/高占用时尝试；
- K 循环：加载 x 的 `TM` 个 half8，gate/up 各 `TN` 个 half8；分别 `mad`；
- epilogue：`silu(g)=g/(1+exp(-g))`，`out=silu(g)*u`，FP32→FP16；
- gate/up 权重不共享；x 复用是主要收益。

### 3.3 技术选型

| 选择 | 理由 |
| --- | --- |
| 双 GEMM 融合 | 减少 x 读取与 kernel launch；x 在 L2/寄存器复用 |
| TN=2 控制累加器 | 16 个 FP32 累加器，降低 spill 风险 |
| 不用 local tile | Mali local 为 CL_GLOBAL；寄存器/L2 优先 |
| SiLU 融合 | 避免中间 gate/up 写回（2×I×M×2 B） |

## 4. 详细设计

### 4.1 接口

```cpp
class LinearSwiGluOclKernel final : public OclKernel {
 public:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status Execute(const runtime_graph::KernelContext& context) override;
  Status Release() override;
};
```

### 4.2 参数

```text
__global const half* x, int x_offset,           // [M,H]
__global const half* gate, int gate_offset,     // [I,H]
__global const half* up, int up_offset,         // [I,H]
__global half* out, int out_offset,             // [M,I]
int m, int hidden, int intermediate
```

### 4.3 Kernel 文档

- 入口：`linear_swi_glu_nt`；
- global：`(ceil(I/BI)*BI, ceil(M/BM)*BM)`；local：`(BI/TN, BM/TM)`；
- 每 work-item：
  1. 16 个 FP32 累加器（gate 8 + up 8，TM=4,TN=2）；
  2. K 循环 half8：x 的 4 个 half8，gate/up 各 2 个 half8；
  3. `mad` 累加；
  4. `silu(g)*u`，写回 half；
- 边界：M/I 非块倍数保护；H 非 8 倍数尾部标量；
- 精度：FP32 累加、FP32 SiLU；一次 FP16 舍入；
- 性能：x 读取量 `M*H*2` 一次，gate/up 读取 `2*I*H*2`；输出 `M*I*2`；
  比两个独立 GEMM 少一次 x 读取与一次中间写回。

### 4.4 优化策略

1. baseline：TM=4,TN=2，WG=(BI/TN=16, BM/TM=4)=64；
2. 试 TM=4,TN=4（32 累加器）与 TM=2,TN=4；比较 spill/性能；
3. 调 BI/BM；真实 I=3072，BI=64 -> 48 个 N 块，M/BM=16 -> 768 WGs；
4. gate/up 权重连续行；half8；L2 复用；
5. SiLU 用 `exp`，FP32；避免 half exp；
6. 目标：真实形状显著快于 Mock；与两个独立 Linear 的 OCL 版本对比融合收益（可选）。

### 4.5 错误处理

- Prepare 校验 I/H/shape/attr；失败返回错误；
- Program 编译失败落 build log；
- 非 8 倍数 H 尾部标量；M/I 边界保护。

## 5. 实施步骤

1. `linear_swi_glu.cl` baseline；
2. 宿主包装与变体；
3. small 与 Mock 对齐；
4. 真实形状（M=256,H=1024,I=3072）性能；
5. 调 TM/TN/BM/BI；记录；
6. 更新 skill/文档。

## 6. 测试与验证设计

### 6.1 数值

- 用例：M=8/256，H=64/1024，I=128/3072；随机 FP16；
- 与 Mock：max_abs ≤ 2e-2、relative_l2 ≤ 5e-3、cosine ≥ 0.9999。

### 6.2 性能

- 预热 3、重复 10、`cl_event` 中位数；
- 记录 kernel/Mock 时间、加速比、有效 FLOPs（2*M*I*H*2）；
- 真实形状为主。

### 6.3 端到端

所有算子完成后按 `ocl_design.md` 6.6 做同输入 Token 的 OCL vs Mock 一致性。

### 6.4 通过标准

数值、性能、文档/skill/结果齐全。

## 7. 实现状态（P4.3）

- 已实现 `linear_swi_glu` OCL 基线 Kernel：`.cl` + 宿主包装 + `Create<Op>OclKernel`；
- 主机 Intel Arc 上经 `ocl_vs_mock_test` 与 Mock 数值对齐通过；
- 真实形状性能与端到端 Token 结果见 `ocl/tests/test_results.md` 11.3–11.5；
- P4.5：接入 `packed_tokens` 有效 token 截断，只计算有效行的 gate/up/SiLU；
- P4.5：K 维 `vload_half8` 向量化 gate/up 双点积，K 非 8 倍数保留标量尾部；
  实测 M=256/K=1024/I=3072：104.773→41.856 ms（2.50x），Mock 3795.8 ms。
