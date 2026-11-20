# LinearPerHeadRMSNormRope OCL 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 模块 | `Runtime/backend/ocl/kernels/linear_per_head_rmsnorm_rope/` |
| 权威位置 | 本文件（含 Kernel 文档章节） |
| 上游文档 | [`ocl_design.md`](../../../.doc/ocl_design.md)、[`backend_design.md`](../../../../.doc/backend_design.md) |
| 算子契约 | `Runtime/backend/ops/linear_per_head_rmsnorm_rope/per_head_rmsnorm_rope_spec.cc` |
| Mock 参考 | `Runtime/backend/mock/kernels/per_head_rmsnorm_rope_kernel.cc` |
| Kernel 文档 | 本文件第 4.3 节 |

## 1. 背景与目标

Q/K path 融合算子：`y = rope(per_head_rms_norm(x·Wᵀ))`。
Qwen3：
- q_path：x `[B,S,1024]`，W `[2048,1024]`（16 heads × 128），norm `[128]`；
- k_path：x `[B,S,1024]`，W `[1024,1024]`（8 heads × 128），norm `[128]`；
- positions `[B,S]` int64；rope_layout=neox，theta=1e6。

GEMM 是主计算；per-head RMSNorm 与 RoPE 是逐 head 的 epilogue。由于 per-head 归约需要
一个 head 的完整 128 维输出，而 GEMM 通常按 N 分块，直接融合较复杂。首期采用
**两阶段 Kernel**（GEMM → per-head norm+rope），中间结果由后端内部 scratch 承载；
后续再评估单 kernel 融合。

目标：与 Mock 数值一致；真实形状性能优于 Mock；沉淀“GEMM + per-head epilogue”方法。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 |
| --- | --- |
| FR-PR-01 | 支持 `x [..,K]`、W `[N,K]`、norm `[D]`、positions `[..]`、out `[..,N]` FP16/int64 |
| FR-PR-02 | `N = num_heads*D`；D=128；rope_layout=neox；theta 来自 attr |
| FR-PR-03 | 先 linear，再 per-head RMSNorm（D 维），再 NeoX RoPE（pair i, i+D/2） |
| FR-PR-04 | FP32 归约/三角/累加；FP16 写回 |
| FR-PR-05 | 与 Mock 在阈值内一致 |
| FR-PR-06 | q/k 两种 heads 数共用同一 Kernel 配置 |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| NFR-PR-01 | GEMM 复用 `linear` 策略；epilogue 带宽友好 |
| NFR-PR-02 | 避免每元素 `sin/cos` 的重复计算（可用 cos/sin 表或按位置缓存） |
| NFR-PR-03 | 中间结果不进入 graph arena 的额外复制；使用后端 scratch/workspace |
| NFR-PR-04 | 真实形状性能优于 Mock |

### 2.3 约束

- Mali-G925：12 CU、half8、2 MB L2、local CL_GLOBAL；
- position 值域 [0, context_length)；D=128；
- 仅 FP16 存储；FP32 计算；
- 不 `clFinish`。

### 2.4 验收

- 与 Mock `max_abs ≤ 2e-2`、relative_l2 ≤ 5e-3、cosine ≥ 0.9999；
- q/k 两种配置通过；
- 真实形状性能数据；
- 文档/skill/test_plan 齐全。

## 3. 概要设计

### 3.1 OCL 接入

- `kernel_key="per_head_rmsnorm_rope"`；`CreatePerHeadRmsNormRopeOclKernel`；
- `Prepare`：
  - 校验 N==heads*D、norm 长度 D、positions 形状；
  - 编译两个 Program/Kernel：`linear_nt` 与 `head_norm_rope`；
  - 计算/分配后端内部 scratch（M*N*2 B），或使用 `KernelContext.workspace`（P4 更新 OpSpec
    `need_workspace=true`）；
  - 预计算/缓存 cos/sin 表（按 position 与 D/2；若用表）。
- `Execute`：
  1. 调用 GEMM kernel 写 scratch；
  2. 调用 head_norm_rope kernel 读 scratch、norm、positions，写 out；
- `Release`。

### 3.2 两阶段 Kernel

**阶段 1：GEMM**：复用 `linear` 的 NT GEMM 策略，A=x，B=W，C=scratch。
**阶段 2：per-head norm + rope**：

- 一个 workgroup 处理 `HB` 个 head（例如 2/4 个 head），或一个 head；
- 每个 head 的 D=128；WG=128 threads，每线程 1 维；或 WG=64，每线程 2 维；
- 步骤：
  1. 读 scratch 的 head 向量（half8）；
  2. FP32 平方和 + subgroup/local 归约；`scale=rsqrt(mean+eps)`；
  3. `y_i = y_i*scale*norm_i`；
  4. RoPE：`pair i, i+D/2`；`angle = pos * theta^(-2i/D)`；`cos/sin`；`out1=y1*c-y2*s`，`out2=y2*c+y1*s`；
  5. FP16 写回 out。

### 3.3 cos/sin 方案

- 方案 A：kernel 内每个 (position, pair) 计算 `sincos`；简单，但 256*H*64 次三角；
- 方案 B：host/device 预计算 `cos/sin[context_length][D/2]`，FP32/FP16；
  - context=40960，D/2=64：40960*64*4*2 = 20.9 MB（FP32）或 10.5 MB（FP16）；
  - 放 device global，按 position 索引；L2 2 MB 只能缓存窗口；
- 方案 C：按 Step 的 S=256 位置预计算 `[S,D/2]`（小表，几十 KB），每步更新；
  - 需要每步 H2D 小表或 device 计算；适合 P4；
- 首期优先 C（每步小表）或 A（实现简单），实测后选择；文档记录。

## 4. 详细设计

### 4.1 接口

```cpp
class PerHeadRmsNormRopeOclKernel final : public OclKernel {
 public:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status Execute(const runtime_graph::KernelContext& context) override;
  Status Release() override;
};
```

### 4.2 参数

阶段 1（复用 linear）：A/B/C、M/N/K。
阶段 2：

```text
__global const half* y, int y_offset,        // [M,N]
__global const half* norm, int norm_offset,  // [D]
__global const long* positions, int pos_offset, // [M]
__global half* out, int out_offset,          // [M,N]
__global const half* cos_table, int cos_offset, // [S,D/2] 或 [context,D/2]
__global const half* sin_table, int sin_offset,
int rows, int heads, int head_dim, float eps
```

### 4.3 Kernel 文档

- 入口：`linear_nt`（阶段 1）+ `per_head_norm_rope`（阶段 2）；
- 阶段 2 global：`rows * heads * WG`；local：`WG`；
  - `WG = 128`（每线程 1 维）或 64（每线程 2 维）；
  - 一个 workgroup 处理一个 (row, head)；
- 每 WG：
  1. `base = (row*heads+head)*D`；
  2. 每线程读 `y[base+tid]`（half8 可选），FP32 平方和；
  3. subgroup/local 归约；`scale`；
  4. `v = y*scale*norm[tid]`；
  5. 对 `i < D/2`：`c=cos_table[pos][i]`，`s=sin_table[pos][i]`；
     `a=v[i]`，`b=v[i+D/2]`；写 `out[i]=a*c-b*s`，`out[i+D/2]=b*c+a*s`；
  6. 若表中没有 pos，回退 kernel 内 `sincos`（或 Prepare 拒绝）；
- 边界：D 偶数；rows/heads 正确；position 越界拒绝；
- 精度：FP32 归约/乘法；FP16 写回；
- 性能：阶段 1 GEMM 主导；阶段 2 读 scratch + 写 out，额外 ~2*M*N*2 B。

### 4.4 优化策略

1. 阶段 1 复用 linear 的 4x4/8x4 分块；
2. 阶段 2 一 (row,head) 一 WG；D=128，128 threads；subgroup 归约；
3. cos/sin 表：按 Step S 预计算并上传（小表）；或 kernel 内 `sincos`，比较；
4. 若阶段 2 带宽主导，合并多个 head/WG 或 half8 读写；
5. 评估单 kernel 融合（一个 WG 计算一个 head 的 128 输出，K 循环 1024）：
   - 计算量 2*128*1024 per head；权重 256 KB/head；L2 复用有限；
   - 可能不优于 GEMM + epilogue；实测决定；
6. 目标：真实形状（q: 2048, k: 1024）性能优于 Mock。

### 4.5 错误处理

- Prepare 校验 heads*D=N、norm=D、positions 形状、rope_layout=neox；
- 表越界/position 越界返回错误；
- Program 编译失败落 build log。

## 5. 实施步骤

1. 阶段 1 复用/链接 linear kernel；
2. `per_head_norm_rope.cl` baseline（kernel 内 sincos）；
3. 宿主包装与两阶段 Execute；
4. small 与 Mock 对齐（q/k 两种 heads）；
5. 真实形状性能；
6. 试 cos/sin 表与单 kernel 融合；记录；
7. 更新 skill/文档。

## 6. 测试与验证设计

### 6.1 数值

- 用例：q 路径 heads=4,D=16（small）与 heads=16,D=128（real）；k 路径 heads=2/8；
- position 0、连续、非连续（如 {0,5,7}）；
- 与 Mock：max_abs ≤ 2e-2、relative_l2 ≤ 5e-3、cosine ≥ 0.9999。

### 6.2 性能

- 阶段 1/阶段 2 分别计时；总 kernel 时间；
- Mock 时间；加速比；GEMM FLOPs；
- 真实 q/k 形状。

### 6.3 端到端

所有算子完成后按 `ocl_design.md` 6.6 做同输入 Token 的 OCL vs Mock 一致性。

### 6.4 通过标准

数值、性能、文档/skill/结果齐全。

## 7. 实现状态（P4.3）

- 已实现 `linear_per_head_rmsnorm_rope` OCL 基线 Kernel：`.cl` + 宿主包装 + `Create<Op>OclKernel`；
- 主机 Intel Arc 上经 `ocl_vs_mock_test` 与 Mock 数值对齐通过；
- 当前为基线实现；性能优化与真实形状误差/端到端 Token 待 P4.5/P4.6。
