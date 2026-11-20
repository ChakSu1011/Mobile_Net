# Gather OCL 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 模块 | `Runtime/backend/ocl/kernels/gather/` |
| 权威位置 | 本文件（含 Kernel 文档章节） |
| 上游文档 | [`ocl_design.md`](../../../.doc/ocl_design.md)、[`backend_design.md`](../../../../.doc/backend_design.md) |
| 算子契约 | `Runtime/backend/ops/gather/gather_spec.cc` |
| Mock 参考 | `Runtime/backend/mock/kernels/gather_kernel.cc` |
| Kernel 文档 | 本文件第 4.3 节 |

## 1. 背景与目标

Gather 实现 token embedding 查表：`out[b,s,:] = weight[ids[b,s],:]`。Qwen3 中
`ids` 为 `[B,S]` int64，`weight` 为 `[V,H]` FP16（V=151936, H=1024），输出
`[B,S,H]` FP16。该算子没有算术，瓶颈是内存带宽与 launch 开销。

目标：

- OCL 输出与 Mock CPU 参考逐元素一致（不引入数值误差）；
- 真实形状下单次 kernel 时间优于 Mock 单线程实现；
- 提供可复用的 embedding/gather 调优方法并沉淀 skill。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 |
| --- | --- |
| FR-GA-01 | 支持 `ids [B,S] int64`、`weight [V,H] FP16`、`out [B,S,H] FP16` |
| FR-GA-02 | `axis=0`；不接受其他 axis |
| FR-GA-03 | 支持任意合法 H（至少 64/1024）；H 不要求是 8 的倍数时用尾部循环 |
| FR-GA-04 | 与 Mock 输出逐字节一致（同一 FP16 拷贝） |
| FR-GA-05 | 无效 id 的行为由上层保证；P4 测试只使用合法 id |
| FR-GA-06 | 支持 batch/seq 维度合并为 token 数；不修改 weight |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| NFR-GA-01 | 内存带宽优先；避免逐元素 scalar load/store |
| NFR-GA-02 | 单次 launch；不引入额外同步或中间 buffer |
| NFR-GA-03 | 真实形状（256 tokens × 1024 halves）达到可用带宽的较大部分 |
| NFR-GA-04 | 与 Mock 对比性能；小形状标注 launch 开销 |

### 2.3 约束

- 仅 FP16 计算；D 为 row-major；`ids` 与 `weight/out` 是独立 Buffer/offset；
- 设备 Mali-G925：12 CU、1.6 GHz、half8、cacheline 64 B、max WG 1024；
- 不假设专用 local memory（`CL_GLOBAL`）；
- 不在 kernel 内 `clFinish`。

### 2.4 验收

- 与 Mock 输出 `max_abs=0`；
- 真实形状 kernel 时间 < Mock 时间，或给出 launch-bound 说明；
- 文档/skill/test_plan 齐全，实测数据回填。

## 3. 概要设计

### 3.1 OCL 接入框架

- `kernel_key="gather"`；
- `CreateGatherOclKernel` 在 `gather_kernel.cc` 中实现；
- `Prepare`：校验 `ids`/`weight`/`out` 的形状与 dtype；获取 Program；创建 `cl_kernel`；
- `Execute`：绑定 `ids`、`weight`、`out` 的 `cl_mem` 与 offset，设置 `tokens/vocab/hidden`，
  计算 global/local，enqueue；
- `Release`：释放 `cl_kernel`。

### 3.2 Kernel 分解

- 一个 workgroup 处理一个 token（或多 token，按 H 调整）；
- 线程数 = `min(256, max(32, H/8))`；每个线程拷贝 8 个 half（half8）；
- `H=1024` 时 WG=128，每线程 8 halves，恰好 1024；
- `ids[token]` 由各线程直接读取（同址，L1/L2 缓存），避免 local 广播；
- 输出按 `token*H` 连续写，合并访存。

### 3.3 技术选型

| 选择 | 理由 |
| --- | --- |
| 一 token 一 WG | 访存连续、实现简单、负载均衡 |
| half8 拷贝 | 设备偏好 half8，128-bit 事务 |
| 不使用 local | Mali local 为 CL_GLOBAL，无收益 |
| 不排序/不去重 | ids 随机，重复读由缓存吸收 |

## 4. 详细设计

### 4.1 接口

```cpp
class GatherOclKernel final : public OclKernel {
 public:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status Execute(const runtime_graph::KernelContext& context) override;
  Status Release() override;
};
Status CreateGatherOclKernel(std::unique_ptr<Kernel>* kernel);
```

### 4.2 参数

Kernel 参数顺序（固定）：

```text
__global const long* ids,          // [tokens]
int ids_offset,
__global const half* weight,       // [vocab, hidden]
int weight_offset,
__global half* out,                // [tokens, hidden]
int out_offset,
int tokens,
int vocab,
int hidden
```

### 4.3 Kernel 文档

- 入口：`gather_embedding`；
- global size：`(tokens * threads_per_token)` 或 2D `(threads, tokens)`；
- local size：`threads_per_token`；
- 每线程：
  1. `token = get_global_id(1)`；
  2. `id = ids[token]`；越界按 P4 约定不处理（测试用合法 id）；
  3. `base = id*hidden + thread*8`；`out_base = token*hidden + thread*8`；
  4. 对 `i = thread*8; i < hidden; i += wg_size*8` 做 `half8` load/store；
- 边界：`H % 8 != 0` 时尾元素用 half 标量循环；
- 不对齐处理：Buffer 由 ConvertParam/CreateBuffer 按 256 B 对齐，`hidden*2` 连续；
- 误差：纯拷贝，`max_abs=0`；
- 性能：读 `tokens*H*2` + 写 `tokens*H*2`；权重随机访问，L2 2 MB 只能缓存部分行。

### 4.4 优化策略

1. baseline：1 token/WG，128 threads，half8；
2. 若 WG 太小（H 小），合并多个 token/WG，减少 launch 与调度；
3. 若 `tokens` 很大，使用 2D NDRange 避免 global size 过大；
4. 评估 `cl_arm_scheduling_controls` 与 workgroup 大小；
5. 目标：真实形状带宽接近设备可持续读带宽的 60%+（以实测为准）。

### 4.5 错误处理

- `Prepare` 校验 dtype/shape/offset；失败返回 `kInvalidArgument`/`kUnsupportedDtype`；
- Program 编译失败返回 `kKernelError`，build log 落盘；
- id 越界：P4 由输入合法性保证；若需要强校验，增加 4 B error buffer 并在调试模式读回，
  不进入生产热路径。

## 5. 实施步骤

1. 写 `gather.cl` baseline（half8）；
2. `gather_kernel.cc` 宿主包装与 `CreateGatherOclKernel`；
3. small 形状与 Mock 对齐（`max_abs=0`）；
4. 真机真实形状性能；
5. 调整 WG/token 合并；记录 `test_results.md`；
6. 更新 `.skill/SKILL.md` 与本文档实测参数。

## 6. 测试与验证设计

### 6.1 数值

- 用例：`B=1,S=8,H=64,V=128`；`B=1,S=256,H=1024,V=151936` 抽样；
- 输入 ids 合法；权重随机 FP16；
- 与 Mock 输出逐字节一致（`max_abs=0`）。

### 6.2 性能

- 预热 3 次，重复 10 次，`cl_event` 中位数；
- 记录 kernel 时间、Mock 时间、加速比、有效带宽；
- 真实形状为主；小形状标注 launch 开销；
- 结果写入 `tests/test_results.md`。

### 6.3 端到端

所有算子完成后，按 `ocl_design.md` 6.6：同一模型包、同一输入 Token、贪心生成，
OCL vs Mock 逐 Step Token 一致率 ≥95%（目标 100%）。

### 6.4 通过标准

- 数值 `max_abs=0`；
- 性能有实测数据与结论；
- 文档/skill/测试结果齐全。
