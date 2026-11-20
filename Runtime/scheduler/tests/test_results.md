# Scheduler 模块测试结果

## 1. 测试范围

本文件记录 `Runtime/scheduler/scheduler.cc`（默认 `IScheduler` 实现）的
主机单元测试结果，覆盖：

- 完整历史重算模式的 Prefill/Decode 计划字段与预算；
- KV 模式 chunked prefill、decode、slots/block_table 与 Release；
- `Update` 追加 token、停止判定（min_tokens/stop_token_ids/ignore_eos/
  max_tokens/context_length）；
- `Stop`/`Fail`、未知请求、空指针、重复调用幂等；
- `Snapshot`/`Stats`/`HasRequests`。

不覆盖：抢占、投机解码、Beam、真实设备执行、多线程压力。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-20 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | `python -m ziglang c++`（zig 0.17.1，Clang 兼容） |
| 语言标准 | C++17 |
| 编译选项 | `-Wall -Wextra` |
| 包含路径 | `-I "d:/Mobile_Net"` |
| 测试源 | `Runtime/scheduler/tests/scheduler_test.cc` |
| 被测源 | `Runtime/scheduler/scheduler.cc`、`Runtime/common/status.cc` |

## 3. 测试命令

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

构建日志（stdout+stderr）：

```text
（无输出）
```

构建退出码：`0`。

## 4. 实际运行输出

```text
Runtime Scheduler unit test
[PASS] CreateDefaultScheduler returns non-null
[PASS] recompute Init succeeds
[PASS] recompute Merge assigns request_id=1
[PASS] recompute HasRequests after Merge
[PASS] recompute snapshot is Waiting with prompt=3
[PASS] recompute first Schedule succeeds
[PASS] recompute first plan has 1 entry and 3 tokens
[PASS] recompute Prefill entry carries full prompt metadata
[PASS] recompute Prefill token ids/positions cover the full prompt
[PASS] recompute Prefill has empty slots/block_table
[PASS] recompute Schedule never calls KvAllocator::Reserve
[PASS] recompute Update succeeds
[PASS] recompute Update appends token and returns Running
[PASS] recompute second Schedule succeeds
[PASS] recompute Decode replans the whole history
[PASS] recompute Update stops on default stop token
[PASS] recompute terminal does not release KV (never reserved)
[PASS] recompute HasRequests false after stop
[PASS] recompute Update after terminal is idempotent
[PASS] recompute Stop after terminal is idempotent
[PASS] KV Init succeeds
[PASS] KV Merge succeeds
[PASS] KV first chunk Schedule succeeds
[PASS] KV first chunk is 4 tokens, non-final, with slots/block_table
[PASS] KV first chunk reserves KV once
[PASS] KV Stats reports waiting request and block usage
[PASS] KV partial prefill snapshot is ChunkedPrefill/computed=4
[PASS] KV partial chunk ignores the placeholder output token
[PASS] KV second chunk Schedule succeeds
[PASS] KV final chunk is 2 tokens and produces output slot
[PASS] KV final chunk reserves KV again
[PASS] KV completed prefill snapshot is Running/computed=6
[PASS] KV prefill completion Update returns Running
[PASS] KV decode Schedule succeeds
[PASS] KV decode entry feeds the last generated token
[PASS] KV decode reserves one token
[PASS] KV Update stops on stop token
[PASS] KV terminal releases allocated blocks
[PASS] KV HasRequests false after stop
[PASS] KV Schedule with no active request returns empty plan
[PASS] Stop/Fail Init succeeds
[PASS] Stop/Fail first Merge
[PASS] Stop returns Stopped response
[PASS] Stop before Reserve does not call Release
[PASS] Stop after terminal is idempotent
[PASS] Fail after Stop preserves the terminal state
[PASS] Merge honors caller-provided request_id
[PASS] Fail returns Illegal response
[PASS] Fail after terminal is idempotent
[PASS] Stop after Fail preserves the terminal state
[PASS] Update unknown request (got REQUEST_NOT_FOUND)
[PASS] Stop unknown request (got REQUEST_NOT_FOUND)
[PASS] Fail unknown request (got REQUEST_NOT_FOUND)
[PASS] Update null response (got INVALID_ARGUMENT)
[PASS] Stop null response (got INVALID_ARGUMENT)
[PASS] Fail null response (got INVALID_ARGUMENT)
[PASS] Fail rejects kOk code (got INVALID_ARGUMENT)
[PASS] Schedule null plan (got INVALID_ARGUMENT)
[PASS] Merge rejects empty prompt (got INVALID_ARGUMENT)
[PASS] Merge rejects duplicate request_id (got INVALID_ARGUMENT)
[PASS] Merge null request_id output (got INVALID_ARGUMENT)
[PASS] limited concurrency Init succeeds
[PASS] limited concurrency first Merge succeeds
[PASS] limited concurrency rejects second active request (got DEVICE_OOM)
[PASS] limited concurrency Stop succeeds
[PASS] limited concurrency admits request after Stop
[PASS] budget Init succeeds
[PASS] budget Merge accepts two short prompts
[PASS] budget first Schedule succeeds
[PASS] budget first schedule stops at max_num_tokens
[PASS] budget Update first request
[PASS] budget second Schedule succeeds
[PASS] budget second schedule still respects token budget
[PASS] budget Stop both requests
[PASS] min_tokens Init succeeds
[PASS] min_tokens Merge succeeds
[PASS] min_tokens Schedule succeeds
[PASS] stop token is ignored before min_tokens
[PASS] default stop token stops after min_tokens
[PASS] ignore_eos Merge succeeds
[PASS] ignore_eos Schedule succeeds
[PASS] ignore_eos suppresses default stop token
[PASS] max_tokens stops ignore_eos request
checks=83 failures=0
```

运行退出码：`0`。

## 5. 用例与需求覆盖

| 用例 | 覆盖需求 | 结论 |
| --- | --- | --- |
| 重算单请求 | FR-SCH-01/02/03/04/08 | 通过 |
| KV chunked prefill + decode | FR-SCH-05/06/07/09/11 | 通过 |
| Stop/Fail/错误码/幂等 | FR-SCH-10/13 | 通过 |
| 重算预算 | FR-SCH-07 | 通过 |
| min_tokens/ignore_eos | FR-SCH-08 | 通过 |
| 并发上限 | 约束条件 | 通过 |

## 6. 结论

- 编译：退出码 0，`-Wall -Wextra` 无警告。
- 运行：退出码 0，`checks=83 failures=0`。
- 实现满足 `Runtime/scheduler/.doc/scheduler_design.md` 中 FR-SCH-01~13
  的可测试部分。

## 7. 未覆盖项与风险

1. 未做多线程并发 Merge/Stop 压力测试；实现使用互斥锁，但未验证高竞争。
2. 未实现/未测试抢占、阶段三投机解码、Beam、采样参数（temperature 等）。
3. 终态请求保留在请求表中；长时间服务的内存回收不在本模块实现。
4. KV 不足时 Running 请求保留 `kRunning` 等待下一步，可能出现长时间饥饿。
5. 未测试超长 prompt（接近 INT32 上限）与极端预算的整型边界。
6. 单测不经过 Engine；当前 `Engine::ExecuteStep` 对 `output_slot<0` 的
   PlanEntry 若缺少 `RequestStepResult` 会按 `kInternal` Fail。长 prompt
   chunked prefill 的端到端路径需 Engine/Executor 修复（Scheduler 侧的
   `Update` 已兼容 `token_id=-1`）。
