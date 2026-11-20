# KVCache 模块测试结果

## 1. 测试范围

本文件记录 `Runtime/kvcache/kv_cache_manager.cc`（默认 `IKVCache` 实现）
的主机单元测试结果，覆盖：

- 容量公式：`block_bytes`、物理块数与 `max_total_tokens`；
- 每层 K/V 缓冲创建、资源栈登记与 `BuildView`；
- `Reserve`/`Release`/`GetBlockTable`/`GetSlots` 与 slot 公式；
- 预算不足、`max_buffer_bytes` 上限、非法规格；
- `CreateBuffer` 中途失败回滚与 `ResourceStack::Clear` 幂等。

不覆盖：真实设备缓冲、共享/COW、滑窗/MLA、多线程与长时间压力。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-20 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | `python -m ziglang c++`（zig 0.17.1，Clang 兼容） |
| 语言标准 | C++17 |
| 编译选项 | `-Wall -Wextra` |
| 包含路径 | `-I "d:/Mobile_Net"` |
| 测试源 | `Runtime/kvcache/tests/kvcache_test.cc` |
| 被测源 | `Runtime/kvcache/kv_cache_manager.cc`、`Runtime/common/resource_stack.cc`、`Runtime/common/status.cc`、`Runtime/common/data_types.cc` |
| Fake Backend | 主机 `std::vector<U8>`，统计 CreateBuffer/DestroyBuffer |

## 3. 测试命令

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

构建日志（stdout+stderr）：

```text
（无输出）
```

构建退出码：`0`。

## 4. 实际运行输出

```text
Runtime KVCache unit test
[PASS] CreateDefaultKVCache returns non-null
[PASS] Stats before Init is empty
[PASS] Reserve before Init fails (got STATE_ERROR)
[PASS] BuildView before Init fails (got STATE_ERROR)
[PASS] Init with budget=1024 succeeds (got OK)
[PASS] Stats reports 2 blocks x 4 tokens
[PASS] BlockSize/FreeBlocks/MaxTotalTokens match stats
[PASS] two layers register four buffers
[PASS] backend created four buffers
[PASS] per-layer buffer holds both blocks (256 bytes)
[PASS] BuildView after Init succeeds (got OK)
[PASS] view reports two layers and capacity
[PASS] every layer has K/V base pointers
[PASS] block_stride equals per-block single-side bytes
[PASS] repeated Init is rejected (got STATE_ERROR)
[PASS] BuildView rejects null view (got INVALID_ARGUMENT)
[PASS] Reserve/Release Init succeeds
[PASS] Reserve 5 tokens succeeds (got OK)
[PASS] Reserve consumes all blocks
[PASS] GetBlockTable succeeds (got OK)
[PASS] 5 tokens use 2 blocks
[PASS] GetSlots for 5 reserved tokens succeeds (got OK)
[PASS] slots follow block_id*block_size+offset
[PASS] GetSlots beyond capacity is rejected (got INVALID_ARGUMENT)
[PASS] GetSlots with negative first_token is rejected (got INVALID_ARGUMENT)
[PASS] GetSlots for unknown request is rejected (got REQUEST_NOT_FOUND)
[PASS] Reserve one more token within existing block succeeds (got OK)
[PASS] no extra block is consumed for capacity 6
[PASS] Reserve beyond capacity returns DeviceOom (got DEVICE_OOM)
[PASS] block table survives failed Reserve (got OK)
[PASS] failed Reserve has zero side effects
[PASS] Release returns both blocks
[PASS] duplicate Release is idempotent
[PASS] GetBlockTable after Release returns NotFound (got REQUEST_NOT_FOUND)
[PASS] Reserve exact full capacity succeeds (got OK)
[PASS] GetSlots covers full capacity (got OK)
[PASS] full capacity yields 8 slots
[PASS] Reserve zero tokens is rejected (got INVALID_ARGUMENT)
[PASS] Reserve more than total capacity is rejected (got DEVICE_OOM)
[PASS] failed oversized Reserve leaves the pool unchanged
[PASS] budget below one block returns OOM (got DEVICE_OOM)
[PASS] budget 512 yields one block (got OK)
[PASS] 512-byte budget produces exactly one block
[PASS] max_buffer_bytes caps block count (got OK)
[PASS] backend max_buffer_bytes limits pool to one block
[PASS] max_buffer_bytes below block size returns OOM (got DEVICE_OOM)
[PASS] Init rejects null backend (got INVALID_ARGUMENT)
[PASS] Init rejects null ResourceStack (got INVALID_ARGUMENT)
[PASS] Init rejects zero block_size (got INVALID_ARGUMENT)
[PASS] Init rejects zero layers (got INVALID_ARGUMENT)
[PASS] Init rejects unknown dtype (got UNSUPPORTED_DTYPE)
[PASS] Init rejects non-full-attention cache type (got UNSUPPORTED_OP)
[PASS] CreateBuffer failure propagates BufferError (got BUFFER_ERROR)
[PASS] backend saw four CreateBuffer calls
[PASS] three successfully created buffers are rolled back
[PASS] failed Init leaves no ResourceStack entries
[PASS] failed Init keeps cache uninitialized
[PASS] Init succeeds after rollback (got OK)
[PASS] retry registers four buffers
[PASS] ResourceStack releases KV buffers (got OK)
[PASS] Clear destroys all buffers including retried ones
[PASS] second ResourceStack Clear is idempotent (got OK)
[PASS] idempotent Clear does not destroy buffers twice
checks=63 failures=0
```

运行退出码：`0`。

## 5. 用例与需求覆盖

| 用例 | 覆盖需求 | 结论 |
| --- | --- | --- |
| 容量/视图（2 层、4 token/块、1024 B） | FR-KV-01/02/03/04 | 通过 |
| Reserve/Release/slot | FR-KV-05/06/07/08 | 通过 |
| 预算/后端上限/非法规格 | FR-KV-02/09/10 | 通过 |
| CreateBuffer 失败回滚 | FR-KV-03 | 通过 |

## 6. 结论

- 编译：退出码 0，`-Wall -Wextra` 无警告。
- 运行：退出码 0，`checks=63 failures=0`。
- 容量公式、视图字段、块池语义与回滚路径符合
  `Runtime/kvcache/.doc/kvcache_design.md` 的 FR-KV-01~10。

## 7. 未覆盖项与风险

1. Fake Backend 不模拟设备对齐、真实 OOM 与异步释放。
2. 未实现/未测试前缀共享、COW、滑窗与 MLA。
3. 未测试多线程或并发 Reserve/Release（接口约定单推理线程）。
4. 未做长时间运行后的碎片与容量压力测试。
5. `ResourceStack::Clear` 后 `KvCacheView` 立即失效；测试未验证调用方
   违反该约束时的行为（属于未定义使用）。
6. 释放闭包捕获 `Backend*`；若上游在 `Executor::Init` 失败时先释放
   Backend 再 Clear 资源栈，会产生 UAF。该所有权时序问题不在 KVCache
   内部解决（详见 Destroy 模块测试结果与设计文档）。
