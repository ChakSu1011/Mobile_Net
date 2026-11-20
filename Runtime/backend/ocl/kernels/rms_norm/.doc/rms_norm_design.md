# RMSNorm OCL 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 模块 | `Runtime/backend/ocl/kernels/rms_norm/` |
| 权威位置 | 本文件（含 Kernel 文档章节） |
| 上游文档 | [`ocl_design.md`](../../../.doc/ocl_design.md)、[`backend_design.md`](../../../../.doc/backend_design.md) |
| 算子契约 | `Runtime/backend/ops/rms_norm/rms_norm_spec.cc` |
| Mock 参考 | `Runtime/backend/mock/kernels/rms_norm_kernel.cc` |
| Kernel 文档 | 本文件第 4.3 节 |

## 1. 背景与目标

RMSNorm 对最后一维做归一化：`y = x * rsqrt(mean(x²)+eps) * w`。Qwen3 中每层有
attention norm、MLP norm 与 final norm，x 为 `[B,S,H]` FP16（H=1024），w 为 `[H]` FP16。
瓶颈是读 x/写 y 的带宽与行内归约。

目标：与 Mock 数值一致；真实形状 kernel 时间优于 Mock；沉淀行归约/subgroup 优化方法。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 |
| --- | --- |
| FR-RN-01 | 支持 `x [B,S,H]`、`w [H]`、`out [B,S,H]` FP16 |
| FR-RN-02 | `axis=-1`、`normalized_shape==H`、`eps` 来自 attr |
| FR-RN-03 | FP32 平方和/均值/rsqrt，FP16 写回 |
| FR-RN-04 | 支持 H=1024 与 64；H 不要求 8 的倍数时尾部循环 |
| FR-RN-05 | 与 Mock 输出在阈值内一致 |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| NFR-RN-01 | 一行一 workgroup；行内并行归约 |
| NFR-RN-02 | 优先 subgroup reduce；避免依赖专用 local memory |
| NFR-RN-03 | half8 加载/写回；权重缓存复用 |
| NFR-RN-04 | 真实形状 256 行 × 1024 的带宽效率 |

### 2.3 约束

- 仅 FP16；D=1024；Mali local memory 为 CL_GLOBAL；max WG 1024；
- 不使用 fast-relaxed-math；
- 不 `clFinish`。

### 2.4 验收

- 与 Mock `max_abs ≤ 2e-2`、relative_l2 ≤ 5e-3、cosine ≥ 0.9999；
- 真实形状 kernel 时间 < Mock；
- 文档/skill/test_plan 齐全。

## 3. 概要设计

### 3.1 OCL 接入

- `kernel_key="rms_norm"`；`CreateRmsNormOclKernel`；
- `Prepare`：校验 attr/shape；Program；`cl_kernel`；
- `Execute`：绑定 x/w/out 的 `cl_mem+offset`，传入 rows/last_dim/eps；
- `Release`。

### 3.2 Kernel 分解

- 一行一 workgroup；`WG = min(256, H/4)`，H=1024 时 256 线程（每线程 4 halves），
  或 128 线程（每线程 8 halves）；
- 每线程计算局部平方和（FP32），用 subgroup reduce 或 local 数组（8 个 subgroup 的部分和）；
- 根线程计算 `scale = rsqrt(sum/H + eps)`，写回共享；
- 所有线程按 half8 读 x、乘 scale 与 weight、写 out；
- 权重 `w` 常驻 L2（2 KB），每个 WG 重复读但命中缓存。

### 3.3 技术选型

| 选择 | 理由 |
| --- | --- |
| 一行一 WG | 归约范围固定、负载均衡、实现简单 |
| subgroup reduce | Mali 支持 `cl_khr_subgroups`；避免 local 争用 |
| 两次遍历 x | 第一次求平方和，第二次乘 scale；x 可能命中 L2；若寄存器足够可缓存一行 |
| H=1024 时 128 threads | 每线程 8 halves，half8 一次加载；占用率与寄存器平衡 |

## 4. 详细设计

### 4.1 接口

```cpp
class RmsNormOclKernel final : public OclKernel {
 public:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status Execute(const runtime_graph::KernelContext& context) override;
  Status Release() override;
};
```

### 4.2 参数

```text
__global const half* x, int x_offset,
__global const half* w, int w_offset,
__global half* out, int out_offset,
int rows, int hidden, float eps
```

### 4.3 Kernel 文档

- 入口：`rms_norm_row`；
- global size：`rows * WG`；local size：`WG`；
- 每 WG：
  1. `row = get_global_id(0) / WG`，`tid = get_local_id(0)`；
  2. `base = row*hidden`；每线程按 `i = tid*8; i < hidden; i += WG*8` 读 half8，累加 FP32 平方和；
  3. subgroup reduce + 跨 subgroup 汇总（`__local float partial[8]` 或 `sub_group_reduce_add`）；
  4. `scale = rsqrt(partial_sum/hidden + eps)`；
  5. 第二遍：`out[base+i] = half(x[base+i] * scale * w[i])`（half8 读、half8 写；weight 可用 half8 读）；
- 边界：`hidden % 8 != 0` 时尾部标量；
- 精度：平方和/均值/rsqrt 全 FP32；输出一次 RN-even；
- 性能：读 x 两次 + 写 out 一次；若 L2 命中，第二次读不增加 DRAM；权重 2 KB 常驻。

### 4.4 优化策略

1. baseline：一行一 WG，256 threads，half8；
2. 尝试 128 threads（每线程 8 halves）与 256 threads（每线程 4 halves），比较寄存器/占用；
3. 用 subgroup reduce 替代 local 树归约；Mali local 为 CL_GLOBAL；
4. 若第二次读 x 成为瓶颈，评估寄存器缓存整行（1024 halves = 2 KB/线程组，不现实）
   或使用 local 缓存（32 KB 可容一行 2 KB）——需实测 local 收益；
5. 多行一 WG（2/4 行）提高 work per WG，减少调度；比较；
6. 目标：真实形状带宽接近设备可持续读+写带宽的 60%+（实测锁定）。

### 4.5 错误处理

- attr/shape/dtype 在 Prepare 校验；Program 编译失败落 build log；
- `rows/hidden` 为 0 拒绝；eps 非负。

## 5. 实施步骤

1. `rms_norm.cl` baseline；
2. 宿主包装与创建；
3. small 与 Mock 对齐；
4. 真机真实形状性能；
5. 调 WG/归约/多行 WG；记录结果；
6. 更新 skill 与文档。

## 6. 测试与验证设计

### 6.1 数值

- 用例：H=64/1024，S=1/8/256，eps=0/1e-6，随机 FP16；
- 与 Mock 阈值：max_abs ≤ 2e-2、relative_l2 ≤ 5e-3、cosine ≥ 0.9999。

### 6.2 性能

- 预热 3、重复 10、`cl_event` 中位数；
- 记录 kernel 时间、Mock 时间、加速比、有效带宽；
- 真实形状：256 行 × 1024。

### 6.3 端到端

所有算子完成后按 `ocl_design.md` 6.6 做同输入 Token 的 OCL vs Mock 生成一致性。

### 6.4 通过标准

数值阈值、性能数据、文档/skill/结果齐全。

## 7. 实现状态（P4.3）

- 已实现 `rms_norm` OCL 基线 Kernel：`.cl` + 宿主包装 + `Create<Op>OclKernel`；
- 主机 Intel Arc 上经 `ocl_vs_mock_test` 与 Mock 数值对齐通过；
- 真实形状性能与端到端 Token 结果见 `ocl/tests/test_results.md` 11.3–11.5；
- P4.5：接入 `packed_tokens` 有效 token 截断，只归一化有效行。
