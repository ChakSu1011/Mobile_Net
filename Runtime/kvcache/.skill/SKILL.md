# Skill: Runtime KVCache 默认实现与回归

## 适用场景

- 修改 `Runtime/kvcache/kv_cache_manager.cc` 的容量计算、缓冲分配、块池、
  Reserve/Release 或 slot 计算；
- 为 KVCache 增加前缀共享/COW、滑窗、MLA 或分层存储；
- 排查“KV 预算不足”“块表/slot 不一致”“释放后仍被访问”；
- 需要在主机 Fake Backend 上回归 KVCache 分配语义。

不适用：Scheduler 组批/停止判定（见 `Runtime/scheduler/.skill/SKILL.md`）、
Attention Kernel 数值、Engine 销毁编排。

## 输入

1. `Runtime/.doc/Runtime_design.md` 4.2.3 与 `MobileNet_design.md` 4.3.3；
2. `Runtime/kvcache/.doc/kvcache_design.md`（本模块权威设计）；
3. `Runtime/common/{kv_allocator,kv_view,resource_stack,data_types,status}.h`；
4. `Runtime/backend/backend.h`（Buffer/BackendCapability/CreateBuffer）；
5. `Runtime/kvcache/kv_cache_manager.h`（公共接口）。

## 步骤

### 1. 计算容量

```text
block_bytes = block_size × num_kv_heads × head_dim × DataTypeSize(dtype)
per_block_pair = block_bytes × 2 × num_layers
blocks = kv_budget_bytes / per_block_pair
blocks = min(blocks, max_buffer_bytes / block_bytes)   # max_buffer_bytes > 0
```

- 所有乘法使用溢出检查（`MultiplyU64`）；
- `blocks==0` 返回 `kDeviceOom`；
- 块数不得超过 `INT32_MAX`（块 ID 为 I32）。

### 2. 分配与登记

- 每层分配 K、V 两个缓冲（`BufferUsage::kKv`，`alignment=min_alignment`）；
- 只有在全部创建成功后才向 `ResourceStack` Push；
- 失败时对已创建缓冲调用 `Backend::DestroyBuffer`，不留下栈条目；
- 释放闭包用 `shared_ptr<unique_ptr<Buffer>>` 包装，保证 `std::function`
  可拷贝且只销毁一次。

### 3. 维护块池

- `free_blocks_` 为 `vector<I32>`；
- `Reserve`：先校验容量与空闲块数，再创建表项/弹块，失败零副作用；
- `Release`：逆序归还块并删除表项，未知请求静默成功；
- `GetSlots`：`slot = blocks[token/block_size] × block_size + token%block_size`；
- `GetBlockTable`：未知请求 `kRequestNotFound`。

### 4. 编译与运行单测

```bash
cd d:/Mobile_Net
mkdir -p Runtime/build
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  Runtime/kvcache/tests/kvcache_test.cc \
  Runtime/kvcache/kv_cache_manager.cc \
  Runtime/common/resource_stack.cc \
  Runtime/common/status.cc \
  Runtime/common/data_types.cc \
  -o Runtime/build/kvcache_test.exe
./Runtime/build/kvcache_test.exe
```

退出码必须为 0，最后一行 `failures=0`。

### 5. 更新文档与结果

- 公式/接口语义变化先改 `Runtime/kvcache/.doc/kvcache_design.md`；
- 实际命令/输出写入 `Runtime/kvcache/tests/test_results.md`；
- 新增测试覆盖容量、回滚、幂等与越界。

## 输出

- 可编译的 `kv_cache_manager.cc`；
- `kvcache_test` 可执行文件与全部 PASS 输出；
- 每层 K/V 缓冲、块表、slot 与视图的回归结论。

## 常见陷阱

1. **std::function 可拷贝**：不能直接捕获 `unique_ptr<Buffer>`；用
   `shared_ptr<unique_ptr<Buffer>>`，首次执行后句柄为空，重复 Clear 安全。
2. **失败的零副作用**：`Reserve` 必须先完成全部校验再修改块表/空闲池；
   否则 Scheduler 重试会重复计费。
3. **块号空间**：所有层共用块号；不要按层分配独立块 ID，否则
   `KvCacheView`/Executor 的 slot 语义不一致。
4. **max_buffer_bytes**：限制的是单缓冲字节数；每层单边缓冲为
   `blocks × block_bytes`，必须按此换算块数上限。
5. **视图生命周期**：`ResourceStack::Clear` 后 `k_base/v_base` 失效；
   Destroy 顺序必须保证 Clear 后不再 BuildView。
6. **Release 逆序**：按块表逆序 push 回空闲池，便于后续复用与调试。
7. **未知 dtype**：`DataTypeSize` 返回 0 时返回 `kUnsupportedDtype`，
   不要继续除法。
8. **非全注意力**：当前拒绝 `kSlidingWindow/kMla`；扩展时需同步更新
   `KvCacheView` 契约与 Executor。
9. **重复 Init**：返回 `kStateError`，不得覆盖既有缓冲或 ResourceStack 条目。
10. **Backend 生命周期**：释放闭包捕获 `Backend*`；必须保证
    `ResourceStack::Clear` 在 Backend 释放之前执行。正常 Destroy 满足；
    `Executor::Init` 失败回滚会先释放 Backend，存在契约缺口（见 Destroy
    模块文档），集成时需先修复 Engine/Executor 所有权时序。
