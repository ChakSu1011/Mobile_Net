# Sample OCL 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 模块 | `Runtime/backend/ocl/kernels/sample/` |
| 权威位置 | 本文件（含 Kernel 文档章节） |
| 上游文档 | [`ocl_design.md`](../../../.doc/ocl_design.md)、[`backend_design.md`](../../../../.doc/backend_design.md) |
| 算子契约 | `Runtime/backend/ops/sample/sample_spec.cc` |
| Mock 参考 | `Runtime/backend/mock/kernels/sample_kernel.cc` |
| Kernel 文档 | 本文件第 4.3 节 |

## 1. 背景与目标

Sample 对 logits 最后一维做 argmax：输入 `[B,S,V]` FP16，输出 `[B,S]` int64。
Qwen3 中 V=151936，S=256；logits 77.8 MB，是单步中最大的只读缓冲之一。
瓶颈是内存带宽；每个 token 需要扫描 151936 个 half。

目标：与 Mock Token 100% 一致（无并列数据）；真实形状 kernel 时间优于 Mock；
沉淀大向量 argmax/两阶段归约方法。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 |
| --- | --- |
| FR-SA-01 | 输入 logits rank≥2、FP16；输出秩=输入秩-1、int64 |
| FR-SA-02 | `method=argmax`、`axis=-1` |
| FR-SA-03 | 并列最大值取最小下标（与 Mock 一致） |
| FR-SA-04 | 支持 V=151936 与 small V |
| FR-SA-05 | 与 Mock Token 100% 一致（无并列测试数据） |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| NFR-SA-01 | half8 扫描；每 row 大并行 |
| NFR-SA-02 | 两阶段归约（WG 内 → 跨 WG 或单 WG 足够） |
| NFR-SA-03 | 真实形状带宽效率 |
| NFR-SA-04 | 确定性 tie-break |

### 2.3 约束

- Mali-G925：12 CU、half8、WG≤1024、2 MB L2；
- logits 只读、输出 int64；
- FP16 比较用 FP32 转换；不 `clFinish`。

### 2.4 验收

- Token 100% 一致（无并列数据）；
- 并列用例返回最小下标；
- 真实形状性能数据；
- 文档/skill/test_plan 齐全。

## 3. 概要设计

- `kernel_key="sample_argmax"`；`CreateSampleOclKernel`；
- Prepare 校验 rank/dtype；Execute 绑定 logits/out，传入 rows/vocab；
- 一行一 WG（或一行多 WG + 第二阶段），WG=256；
- 每线程按 half8 扫描 `V/WG` 个元素，维护局部 max/index；
- WG 内归约（subgroup + local 小组）得到 row argmax；
- 写 int64 输出。

## 4. 详细设计

### 4.1 接口

```cpp
class SampleOclKernel final : public OclKernel {
 public:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status Execute(const runtime_graph::KernelContext& context) override;
  Status Release() override;
};
```

### 4.2 参数

```text
__global const half* logits, int logits_offset,  // [rows, vocab]
__global long* out, int out_offset,              // [rows]
int rows, int vocab
```

### 4.3 Kernel 文档

- 入口：`sample_argmax_row`；
- global：`rows * WG`；local：`WG`（256）；
- 每 work-item：
  1. `row = gid / WG`，`tid = lid`；
  2. 局部 `best=-inf`、`best_index=0`；
  3. `for (i=tid*8; i<vocab; i+=WG*8)`：half8 load；逐元素转 FP32 比较；
     - 严格 `>` 更新，保留最早下标；
  4. WG 内归约：subgroup reduce（max + index），跨 subgroup 用 local 小数组；
  5. 根线程写 `out[row]=best_index`；
- 边界：`vocab % 8 != 0` 尾部标量；rows 任意；
- 精度：half→FP32 比较；int64 输出；
- 性能：读 V*rows*2 B；V=151936、rows=256 -> 77.8 MB；带宽主导。

### 4.4 优化策略

1. baseline：一行一 WG、256 threads、half8；
2. 若 V 很大，单 WG 每线程 ~593 元素；可一行多 WG + 第二阶段，但需临时 buffer；
   先测一行一 WG（256 WGs，足够 12 CU）；
3. half8 比较；避免 half 比较（转 FP32）；
4. tie-break 用严格 `>`，归约时同值取较小 index；
5. 评估与 lm_head 融合（不在本算子范围；需图/优化器支持）；
6. 目标：真实形状接近读带宽；显著快于 Mock。

### 4.5 错误处理

- Prepare 校验 rank/dtype/axis/method；
- vocab=0 拒绝；Program 编译失败落 build log；
- rows=0 允许（无输出）。

## 5. 实施步骤

1. `sample.cl` baseline；
2. 宿主包装；
3. small 与 Mock 对齐（含并列用例）；
4. 真实 V=151936 性能；
5. 调 WG/两阶段；记录；
6. 更新 skill/文档。

## 6. 测试与验证设计

### 6.1 数值

- 用例：rows=1/8/256，V=3/128/151936；
- 随机无并列；构造并列（如 {0.9,0.9,0.1}）验证最小下标；
- 与 Mock Token 100% 一致（无并列数据）。

### 6.2 性能

- 预热 3、重复 10、`cl_event` 中位数；
- 记录 kernel/Mock 时间、加速比、有效带宽（读 77.8 MB / 时间）；
- 真实 rows=256,V=151936。

### 6.3 端到端

所有算子完成后按 `ocl_design.md` 6.6 做同输入 Token 的 OCL vs Mock 一致性；
Sample 是最后一个算子，端到端 Token 直接受其影响。

### 6.4 通过标准

Token 100%、性能数据、文档/skill/结果齐全。
