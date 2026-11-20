# Add OCL 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 模块 | `Runtime/backend/ocl/kernels/add/` |
| 权威位置 | 本文件（含 Kernel 文档章节） |
| 上游文档 | [`ocl_design.md`](../../../.doc/ocl_design.md)、[`backend_design.md`](../../../../.doc/backend_design.md) |
| 算子契约 | `Runtime/backend/ops/add/add_spec.cc` |
| Mock 参考 | `Runtime/backend/mock/kernels/add_kernel.cc` |
| Kernel 文档 | 本文件第 4.3 节 |

## 1. 背景与目标

Add 为逐元素加法：`out = a + b`。Qwen3 每层有两个残差 add：
`[B,S,1024]` FP16，共 56 个/图。典型形状元素数 256×1024 = 262144，
数据量 512 KB/输入，读 1 MB + 写 512 KB。瓶颈是内存带宽与 kernel launch 开销。

目标：与 Mock 一致；真实形状 kernel 时间与 Mock 对比；明确小算子的 launch 开销边界。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 |
| --- | --- |
| FR-AD-01 | 两个 FP16 输入同形状、同 dtype；FP16 输出 |
| FR-AD-02 | 支持任意元素数；half8 向量化；尾部标量 |
| FR-AD-03 | 与 Mock 输出阈值内一致 |
| FR-AD-04 | 不广播、不 in-place（输出独立 Buffer） |

### 2.2 非功能需求

| 编号 | 要求 |
| --- | --- |
| NFR-AD-01 | 合并读写；half8 |
| NFR-AD-02 | 大形状用 grid-stride/多 WG；小形状控制 launch 开销 |
| NFR-AD-03 | 真实形状性能优于 Mock |

### 2.3 约束

- Mali-G925：12 CU、half8、WG≤1024；FP16 存储、FP32 相加；
- 不 `clFinish`。

### 2.4 验收

- max_abs ≤ 5e-3、relative_l2 ≤ 1e-3、cosine ≥ 0.99999；
- 真实形状性能数据；
- 文档/skill/test_plan 齐全。

## 3. 概要设计

- `kernel_key="add"`；`CreateAddOclKernel`；
- Prepare 校验 shape/dtype；
- Execute 绑定 a/b/out 的 `cl_mem+offset`，传入元素数；
- 一维 NDRange；WG=128/256；每线程处理 half8，grid-stride 覆盖总元素；
- 小形状（<WG*8）使用 1 个 WG，剩余线程不工作。

## 4. 详细设计

### 4.1 接口

```cpp
class AddOclKernel final : public OclKernel {
 public:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status Execute(const runtime_graph::KernelContext& context) override;
  Status Release() override;
};
```

### 4.2 参数

```text
__global const half* a, int a_offset,
__global const half* b, int b_offset,
__global half* c, int c_offset,
int elements
```

### 4.3 Kernel 文档

- 入口：`add_half8`；
- global：`ceil(elements/8)` 对齐到 WG 倍数；local：128/256；
- 每线程：
  1. `i = (get_global_id(0))*8`；grid-stride `i += global_size*8`；
  2. `half8 va = vload8(0, a+i)`；`vb`；`vc = va+vb`（FP16 加法？）
     - 精度要求：FP32 相加后舍入；可 `convert_float8` 相加再 `convert_half8`；
  3. `vstore8(vc, 0, c+i)`；
- 边界：`i+8 > elements` 时标量循环；
- 精度：FP32 相加，一次 FP16 舍入；
- 性能：读 2×、写 1×；带宽主导。

### 4.4 优化策略

1. baseline：half8、WG=256、grid-stride；
2. 真实 262144 元素：global = 262144/8=32768 work-items；WG=256 -> 128 WGs；
3. 小形状 launch 开销：1 WG；对比 Mock；
4. 评估 half8 vs half4；Mali half8 偏好；
5. 注意 56 个 add/step 的 launch 总量；后续可考虑图级融合/命令缓冲（非本算子范围）。

### 4.5 错误处理

- Prepare 校验形状/dtype；失败返回错误；
- Program 编译失败落 build log；
- 元素数为 0 拒绝。

## 5. 实施步骤

1. `add.cl` half8 baseline；
2. 宿主包装；
3. small 与 Mock 对齐；
4. 真实形状性能；
5. 调 WG/half8；记录；
6. 更新 skill/文档。

## 6. 测试与验证设计

### 6.1 数值

- 用例：元素数 4/1024/262144；随机 FP16；边界（最大值、次正规）；
- 与 Mock：max_abs ≤ 5e-3、relative_l2 ≤ 1e-3、cosine ≥ 0.99999。

### 6.2 性能

- 预热 3、重复 10、`cl_event` 中位数；
- 记录 kernel/Mock 时间、加速比、有效带宽；
- 小形状标注 launch 开销。

### 6.3 端到端

所有算子完成后按 `ocl_design.md` 6.6 做同输入 Token 的 OCL vs Mock 一致性。

### 6.4 通过标准

数值、性能、文档/skill/结果齐全。
