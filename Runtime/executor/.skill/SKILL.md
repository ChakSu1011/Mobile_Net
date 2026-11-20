# Skill: Runtime Executor 实现与验证

## 适用场景

- 新增或修改 `Runtime/executor/` 的 Executor 实现、主机 I/O 打包与
  `StepResult` 回读逻辑；
- 为新的图输入约定（token/position 顺序、额外输入）扩展 Executor；
- 排查 `Executor::Execute` 报出的 `OpEnv is not bound`、
  `host storage is not bound`、`packed tokens exceed host buffer` 等问题；
- 对照 `Runtime/.doc/Runtime_design.md` 4.2.4/4.3.4/3.5 校验执行链路；
- 为其他后端（OCL/CUDA/Vulkan）接入 Executor 前的主机侧契约回归。

不适用：Scheduler/KVCache 策略、Loader 模型包解析、Kernel 数学实现、
Engine 线程模型；这些模块只作为借用方出现。

## 输入

1. 接口与所有权权威：`Runtime/.doc/Runtime_design.md`（4.2.4、4.3.4、3.5）；
2. 本模块设计：`Runtime/executor/.doc/executor_design.md`；
3. 公共头：
   - `Runtime/executor/executor.h`；
   - `Runtime/backend/backend.h`、`exec_op_env.h`、`kernel_context.h`；
   - `Runtime/common/{plan.h,kv_view.h,io_binder.h,sequence.h}`；
   - `Graph/include/{graph.h,op.h,tensor.h,storage.h,op_type.h}`；
4. 已构建的静态库：`Graph/build/libgraph.a`、
   `Runtime/backend/build/libruntime_backend.a`（含 MockBackend）；
5. 编译器：`python -m ziglang c++`（Clang 22.x，C++17）。

## 步骤

### 1. 冻结执行约定

- 输入顺序：`Graph::GetAllInputOp()` 第 1 个是 token_ids，第 2 个是
  position_ids；只有 1 个时仅填 token；
- token/position 主机缓冲必须 `int64`，`byte_size % 8 == 0`；
- plan 的 `packed_offset` 是 packed 下标；未覆盖位置必须补 0；
- sample 输出从 `GetSampleOp()->DstTensors()[0]->GetStorage()` 读取，
  `data_ptr` 是 `Buffer*`，`offset` 单独传给 `DownloadAt`，不要对
  `DevicePtr()` 做指针加法；
- 全部尺寸来自 Graph Storage，禁止写死模型/词表/序列长度。

### 2. 实现 Init

- 接管 `std::unique_ptr<Backend>`，保存 Graph/IoBinder 借用指针；
- 校验 Input/Output/Sample op 存在（缺失返回 `kUnsupportedOp`）；
- 每个边界 op 恰好 1 个 tensor；从 Storage 复制
  shape/precision/layout/byte_size，分配并清零主机缓冲；
- 全部成功后再提交成员；失败时让参数 `backend` 自动析构。

### 3. 实现 Execute

1. 清空 `StepResult`，校验 result/未初始化状态；
2. 校验 plan：entries 非空、token_count>0、token_ids/positions 长度一致、
   `plan.total_tokens == sum(token_count)`、packed 范围合法；
3. 校验前两个输入为 int64；总 token 不超过缓冲元素数；
4. 清零缓冲后按 packed 偏移用 `memcpy` 写入 I64；
5. 构造 host Storage（`kHost`、`offset=0`、`data_ptr=buffer.data()`、
   `reusable=false`）并调用 `SetInput/SetOutput`；
6. 构造 `StepContext`：backend/plan/kv_view/seq_lens/batch/
   packed_tokens/is_prefill；
7. 遍历 `GetOrderlyOPList()`：`SetStepContext(env,&ctx)` → `Forward()` →
   `SetStepContext(env,nullptr)`，Forward 失败立即返回；
8. `Backend::Synchronize`；
9. 从 sample Storage 用 `DownloadAt` 回读到 `vector<I64>`（`byte_size/8`），
   `offset` 取 Storage；
10. 按 `output_slot` 填 `RequestStepResult`；`output_slot<0` 跳过。

### 4. 实现 Deinit

- 幂等；未 Init 返回 `kOk`；
- 调用 `Backend::Deinit`，失败仍释放 Backend 并返回带前缀错误；
- 清空图/绑定指针与全部主机缓冲；析构函数调用 Deinit。

### 5. 测试

- 用 `MockBackend` + C++ Graph 构造
  `Input → Gather → Linear → Sample → Output`；
- 设计可解析权重使 argmax 等于 token id，例如
  `embedding[i]=[i,1]`、`W[v]=[2v,-v²]`；
- `TestIoBinder` 必须模拟 Engine：校验后调用 `op->SetAttr("storage", host)`，
  否则 `IoOpEnv::Forward` 报 `host storage is not bound`；
- 覆盖：单/双输入、prefill/decode、补 0、output_slot<0、越界、缺失 OpEnv、
  重复 Init、Deinit 幂等。

### 6. 编译与运行

```bash
cd d:/Mobile_Net
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  Runtime/executor/executor.cc Runtime/executor/tests/executor_test.cc \
  Runtime/backend/build/libruntime_backend.a Graph/build/libgraph.a \
  -o Runtime/executor/build/executor_test.exe
./Runtime/executor/build/executor_test.exe
```

## 输出

- `Runtime/executor/executor.cc`：`IExecutor` 默认实现与
  `CreateDefaultExecutor()`；
- `Runtime/executor/tests/executor_test.cc`：主机/O 绑定与执行链测试；
- `Runtime/executor/build/executor_test.exe`：测试可执行文件；
- 预期输出：逐条 `PASS ...`，最后 `ALL TESTS PASSED`，退出码 0；
- 交付文档：`Runtime/executor/.doc/executor_design.md`、
  `Runtime/executor/tests/test_results.md`、本 SKILL。

## 常见陷阱

1. **忘记写 `attr["storage"]`**：Executor 只调用 `IoBinder::SetInput`；
   真实 Engine 会写 attr，测试 Binder 必须同样写入，否则 IoOpEnv 失败。
2. **把 `data_ptr` 当裸设备地址**：设备侧是 `Buffer*`，必须
   `buffer->DevicePtr() + storage.offset`。
3. **忽略 host `offset`**：主机 Storage 的 `offset` 必须为 0。
4. **用 `reinterpret_cast<I64*>` 访问 `vector<U8>`**：可能违反对齐假设；
   使用 `memcpy` 读写单个 I64。
5. **每步忘记清零**：只覆盖 total 个位置会导致上一步残留 token 留在
   padding 区，必须先 `memset` 整个缓冲。
6. **把 `packed_offset` 当成 entry 下标**：它是 packed token 下标，必须
   加上 entry 内部偏移 `i`。
7. **sample 输出按 Output op 读取**：Output op 的源张量不一定是
   sample.out，必须从 `GetSampleOp()` 读取。
8. **只填 token 不填 position**：双输入图会因 position 全 0 而数值错误；
   按输入顺序同时填充。
9. **Deinit 前未清理 OpEnv**：Engine 的 Destroyer 负责先 ClearOpEnv；
   测试中也要先清 OpEnv 再 Deinit，避免 Kernel 持有已释放的 Backend。
10. **`output_slot<0` 仍生成请求结果**：会让 Scheduler 收到假 token；
    本实现按约定跳过，Engine 侧需支持“无 token 步”（已知限制）。
11. **用 `malloc`/`new` 手工管理主机缓冲**：使用 `vector<U8>`，Deinit 与
    析构自动释放，避免泄漏。
12. **在公共头里新增执行细节**：`executor.h` 已冻结；实现细节放 `.cc`
    内部类型，不改公共头。

## 检查清单

- [ ] `-Wall -Wextra` 零告警；
- [ ] Init 失败后 `backend()` 为 nullptr；
- [ ] 重复 Init 返回 `kStateError`；
- [ ] 单/双输入的 token/position 缓冲内容与补 0 被断言；
- [ ] 绑定的 Storage 满足 `kHost/指针/精度/布局/字节数/形状`；
- [ ] StepResult 的 `request_id/token/select/ok/error` 被断言；
- [ ] 缺失 OpEnv、尺寸越界、total 不一致返回明确错误码；
- [ ] Deinit 两次均成功且释放 Backend；
- [ ] 设计文档、测试结果、skill 三份文档与代码同步。
