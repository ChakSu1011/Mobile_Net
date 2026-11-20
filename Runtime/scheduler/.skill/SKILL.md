# Skill: Runtime Scheduler 默认实现与回归

## 适用场景

- 修改 `Runtime/scheduler/scheduler.cc` 的请求合并、chunked prefill、
  decode、停止判定或统计；
- 为 Scheduler 增加新请求状态、抢占或阶段三投机解码；
- 排查“请求不结束”“计划字段不一致”“KV Reserve/Release 次数异常”；
- 需要在无设备、无 Graph 的主机环境回归 Scheduler 行为。

不适用：KV 物理布局/块分配（见 `Runtime/kvcache/.skill/SKILL.md`）、
Executor 执行、Engine 生命周期编排。

## 输入

1. `Runtime/.doc/Runtime_design.md` 4.2.2 与 `MobileNet_design.md` 4.3.2；
2. `Runtime/scheduler/.doc/scheduler_design.md`（本模块权威设计）；
3. `Runtime/common/{sequence,plan,kv_allocator,model_config,status}.h`；
4. `Runtime/scheduler/scheduler.h`（公共接口，不得随意修改）；
5. 编译器：`python -m ziglang c++`（C++17）。

## 步骤

### 1. 明确执行形态

- `recompute_attention=true`：每步计划必须包含完整 history；positions 从 0
  连续；`output_slot=len-1`；`context_length=len-1`；slots/block_table 为空；
  不调用 `KvAllocator::Reserve`。
- `recompute_attention=false`：Waiting 走 chunked prefill，Running 每步
  decode 1 token；必须调用 `Reserve/GetBlockTable/GetSlots`。

### 2. 修改状态机

- `Merge`：校验 prompt/采样参数/token id/并发上限；分配 id；
  `status=kWaiting`、`computed_tokens=0`、`kv_reserved=false`。
- `Schedule`：只改状态与计划，不生成 Response；推进 `computed_tokens`。
- `Update`：只有 `kRunning` 追加 token；`kChunkedPrefill` 忽略输出；
  终态幂等返回原状态。
- `Stop/Fail`：终态设置后调用 `ReleaseIfReserved`，再 `FillResponse`。

### 3. 校验预算与停止条件

- chunk = `min(prompt 剩余, max_prefill_chunk, 剩余预算)`；
- decode 需要 1 个预算；总 token 必须 ≤ `max_num_tokens`；
- 停止顺序：max_tokens/context_length → 重算预算 → min_tokens →
  sample stop ids →（ignore_eos=false 时）默认 stop ids；
- 多 token Update 命中停止后不再追加后续 token。

### 4. 编译与运行单测

```bash
cd d:/Mobile_Net
mkdir -p Runtime/build
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  Runtime/scheduler/tests/scheduler_test.cc \
  Runtime/scheduler/scheduler.cc \
  Runtime/common/status.cc \
  -o Runtime/build/scheduler_test.exe
./Runtime/build/scheduler_test.exe
```

退出码必须为 0，最后一行 `failures=0`。

### 5. 更新文档与结果

- 行为变化先改 `Runtime/scheduler/.doc/scheduler_design.md`；
- 实际命令/输出追加到 `Runtime/scheduler/tests/test_results.md`；
- 新增需求需在文档中编号并映射到测试用例。

## 输出

- 可编译的 `scheduler.cc`；
- `scheduler_test` 可执行文件与全部 PASS 输出；
- 与实现一致的设计文档和测试结果。

## 常见陷阱

1. **partial chunk 输出 token**：Executor 对未完成 Prefill 可能返回
   `token_id=-1`；`Update` 必须在 `kChunkedPrefill` 状态忽略它，否则会把
   占位 token 追加进 history。
2. **Update 的 computed 语义**：`computed_tokens` 只在 Schedule 推进；
   Update 不得修改，否则 decode 会重复/跳过 token。
3. **失败计划的状态提交**：Reserve 失败时不得先推进状态再返回错误；本实现
   采用“已成功条目返回 kOk、计划为空才返回错误”。
4. **重算预算**：重算模式 history 必须能在单步容纳；Merge/Update 双重截停，
   否则下一次 Schedule 无法生成计划。
5. **终态不删除**：为了 Stop/Update/Fail 幂等，终态请求保留在表中；
   长时间服务需上层重建 Engine 或后续增加回收接口。
6. **Stop 与 Fail 的终态优先级**：已 `kIllegal` 的请求再 Stop 仍返回
   `kIllegal`；已 `kStopped` 的请求再 Fail 仍返回 `kStopped`。
7. **锁范围**：所有公共方法在锁内调用 KvAllocator；不要在锁内调用
   Engine/回调，避免反向加锁。
8. **max_concurrent_requests**：在 Merge 处限制未结束请求数，不是单步
   plan 条目上限；不要误删该检查。
9. **Status 消息**：错误消息保持方法名前缀，便于日志定位；不要只返回
   `kInternal` 而丢失上下文。
10. **Engine 对 output_slot<0 的处理**：本模块对未完成 chunked prefill 设置
    `output_slot=-1`，Executor 会跳过该请求；当前 Engine 把“缺少
    RequestStepResult”当成 `kInternal` 并 Fail。集成长 prompt 前需确认
    Engine 已跳过 `output_slot<0` 的条目，或 Executor 返回 `token_id=-1`
    的空结果（Update 在 `kChunkedPrefill` 下会忽略它）。
