# KVCache 模块设计文档

| 项目 | 内容 |
| --- | --- |
| 模块路径 | `Runtime/kvcache/` |
| 权威接口 | `Runtime/kvcache/kv_cache_manager.h`（`IKVCache`） |
| 上游设计 | `Runtime/.doc/Runtime_design.md` 4.2.3；`MobileNet_design.md` 4.3.3 |
| 实现文件 | `Runtime/kvcache/kv_cache_manager.cc` |
| 测试 | `Runtime/kvcache/tests/kvcache_test.cc`、`Runtime/kvcache/tests/test_results.md` |
| 状态 | 阶段一默认实现；v0.2 增加 Paged Attention 所需视图字段与 NHD 校验；共享/COW 仅预留 |

## 1. 背景与目标

### 1.1 背景

Paged KV Cache 把序列的 KV 按固定 token 块管理：请求拥有逻辑块表，物理块
来自统一空闲池；Attention Kernel 通过 `slot = block_id * block_size + offset`
定位 KV。块大小不是模型常量，而由后端 Attention tile
（`BackendCapability.attention_block_size` 或 `RuntimeConfig.kv_block_size`）
决定；容量由显存预算、层数、KV heads、head_dim 与 dtype 共同计算。

Scheduler 只通过 `KvAllocator` 子接口申请/释放块；Executor 只通过
`KvCacheView` 读取每层 K/V 基址与块步长。KVCache 负责在两者之间维护物理
资源与块表。

### 1.2 目标

1. 通用 block pool：块大小、层数、KV head、head_dim、dtype 全部来自
   `KvCacheSpec` 与 `BackendCapability`，不写死任何模型常量。
2. 用 `Backend::CreateBuffer` 分配每层 K/V 缓冲，登记到 `ResourceStack`，
   由 Destroy 统一释放。
3. 实现 `Reserve/Release/FreeBlocks/BlockSize/MaxTotalTokens/GetBlockTable/
   GetSlots`，`Reserve` 失败零副作用，`Release` 幂等。
4. `BuildView` 输出每层 `k_buffer/v_buffer`（`Buffer*`）、
   `block_stride/num_blocks/num_kv_heads/head_dim/dtype`、`block_size` 与
   `max_total_tokens`；阶段一固定 NHD。
5. 预算不足、规格非法、后端分配失败有明确错误码。

### 1.3 非目标

- 不决定抢占对象、不实现 Attention/采样。
- 不直接调用设备 API；不解析模型包。
- 不实现前缀共享/COW（阶段二扩展点，接口已预留）。
- 不保证多线程安全；只允许单推理线程使用。
- 不负责 `GraphEnv`、OpEnv 或 Backend 生命周期。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| FR-KV-01 | 由 spec/backend 计算 `block_bytes = block_size × num_kv_heads × head_dim × dtype_size` | TC-KV-01 |
| FR-KV-02 | 由预算计算 `blocks = budget / (num_layers × 2 × block_bytes)`；预算不足返回 `kDeviceOom` | TC-KV-01/03 |
| FR-KV-03 | 每层分配 K/V 两个缓冲，登记到 `ResourceStack`；Init 失败回滚已创建缓冲 | TC-KV-01/04 |
| FR-KV-04 | `BuildView` 提供每层 `k_buffer/v_buffer`（`Buffer*`）、`block_stride=block_bytes`、`num_blocks`、`num_kv_heads`、`head_dim`、`dtype`、`block_size`、`max_total_tokens` | TC-KV-01、TC-KV-05 |
| FR-KV-05 | 维护请求块表、空闲块池；`Reserve` 按累计容量追加块，失败零副作用 | TC-KV-02 |
| FR-KV-06 | `Release` 逆序归还全部块并删除块表；重复 Release 幂等 | TC-KV-02 |
| FR-KV-07 | `GetSlots` 计算 `block_id×block_size+offset`，越界返回 `kInvalidArgument` | TC-KV-02 |
| FR-KV-08 | `GetBlockTable` 对未知请求返回 `kRequestNotFound` | TC-KV-02 |
| FR-KV-09 | 非 `kFullAttention`、未知 dtype、零块大小等非法规格返回对应错误 | TC-KV-03 |
| FR-KV-10 | `BackendCapability.max_buffer_bytes` 限制单缓冲块数，避免超限分配 | TC-KV-03 |
| FR-KV-11 | `KvLayerView` 保存 `Buffer*` 而非 `DevicePtr()`，并附 `num_blocks/num_kv_heads/head_dim/dtype` | TC-KV-05 |
| FR-KV-12 | `Init` 校验 `BackendCapability.kv_layout == KvLayout::kNhd`，非 NHD 返回 `kUnsupportedOp` | TC-KV-05 |
| FR-KV-13 | `Init` 校验 KV dtype 在 `supported_dtypes` 中，否则 `kUnsupportedDtype` | TC-KV-05 |
| FR-KV-14 | slot 仍为扁平 token 下标；Kernel 可直接用 `slot × num_kv_heads × head_dim` 定位 NHD 池 | TC-KV-02、TC-KV-05 |

### 2.2 非功能需求

- **NFR-KV-01 通用性**：不出现 Qwen3 或其他模型常量。
- **NFR-KV-02 内存**：总 KV 字节数不超过预算；每层单边缓冲不超过后端
  `max_buffer_bytes`。
- **NFR-KV-03 复杂度**：Reserve O(新增块数)，Release O(块数)，GetSlots O(count)。
- **NFR-KV-04 可回滚**：Init 任一步失败后不留下 `ResourceStack` 条目与半
  初始化状态，可重试 Init。
- **NFR-KV-05 所有权清晰**：KVCache 不拥有 Backend；缓冲所有权通过
  ResourceStack 闭包转移给释放阶段。

### 2.3 约束条件

- `block_size/num_layers/num_kv_heads/head_dim > 0`，`dtype_size > 0`。
- `Init` 只允许成功一次；重复 Init 返回 `kStateError`。
- 块 ID 使用 `I32`，物理块数不得超过 `INT32_MAX`。
- `KvCacheView` 的 `Buffer*` 在 `ResourceStack::Clear` 之前有效；清理后视图失效。
- 阶段一布局固定 NHD：块主序、`[block_size, num_kv_heads, head_dim]`；
  `block_stride = block_size × num_kv_heads × head_dim × dtype_size`。
- `KVCache` 内部保留非拥有 `Buffer*`（`k_buffers_/v_buffers_`）供
  `BuildView` 使用；实际释放由 `ResourceStack` 闭包完成。
- 阶段二共享/COW 不实现；`RequestEntry.ref_cnt` 仅为未来扩展保留。

### 2.4 验收标准

1. `kvcache_test` 全部断言通过，退出码 0（当前 63 条断言）。
2. 容量公式、视图字段、slot 公式逐项验证。
3. Reserve/Release 的零副作用、幂等与越界路径均有覆盖。
4. Init 失败回滚后缓冲销毁次数与创建次数一致。
5. 编译在 `-Wall -Wextra` 下无警告。

## 3. 概要设计

### 3.1 总体架构

```text
Engine
  └── IKVCache (DefaultKVCache)
        ├── Backend*            （借用，CreateBuffer/DestroyBuffer）
        ├── ResourceStack*      （借用，登记释放闭包）
        ├── 物理层：
        │     ├── k_buffers_/v_buffers_：每层 K/V 的 Buffer*
        │     ├── block_bytes_：单块单边字节数
        │     ├── total_blocks_/max_total_tokens_
        │     └── free_blocks_：空闲块池
        └── 逻辑层：
              └── tables_: map<request_id, RequestEntry{blocks, capacity, ref_cnt}>
```

### 3.2 模块划分

| 组成 | 职责 |
| --- | --- |
| `DefaultKVCache` | `IKVCache` 全部接口实现 |
| `RequestEntry` | 请求逻辑块表、已预留 token 容量、引用计数扩展位 |
| `DestroyBuffers` | Init 失败时回滚已创建缓冲 |
| `MultiplyU64` | 容量计算溢出保护 |

### 3.3 关键流程

#### 3.3.1 Init

```text
校验 backend/resource_stack/spec/dtype/budget
  → Capability() 读取 max_buffer_bytes/min_alignment
  → block_bytes = block_size*heads*head_dim*element_bytes
  → blocks = budget/(layers*2*block_bytes)
  → blocks = min(blocks, max_buffer_bytes/block_bytes)
  → 若 blocks==0：kDeviceOom
  → 逐层 CreateBuffer(K)、CreateBuffer(V)
  → 记录每层基址
  → Push 2*layers 个释放闭包到 ResourceStack
  → 初始化 free_blocks_、tables_、initialized_=true
```

#### 3.3.2 Reserve

```text
capacity = 旧容量（无表项为 0）
needed = capacity + new_tokens
needed_blocks = ceil(needed / block_size)
extra = needed_blocks - 当前块数
若 needed_blocks > total_blocks 或 extra > 空闲块数：kDeviceOom（不改状态）
创建表项（若不存在）
从 free_blocks_ 弹出 extra 个块追加到块表
capacity_tokens = needed
```

#### 3.3.3 GetSlots

```text
校验 first_token/count 位于 [0, capacity]
slot[i] = blocks[(first+i)/block_size] * block_size + (first+i)%block_size
```

### 3.4 技术选型与取舍

- **每层独立 K/V 缓冲**：便于遵守 `max_buffer_bytes`，避免单缓冲过大；
  代价是 `2×num_layers` 个缓冲句柄与释放闭包。
- **统一物理块号空间**：所有层共享同一块号，块表只有一份，符合设计文档
  4.3.3；每层通过基址 + 块步长定位。
- **空闲池用 vector**：Reserve pop、Release push，O(1)；不追求按块号排序。
- **释放闭包用 shared_ptr 包装 unique_ptr**：`std::function` 要求可拷贝，
  而 Buffer 独占；首个闭包副本执行后句柄变为空，重复 Clear 安全。

## 4. 详细设计

### 4.1 数据结构

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `initialized_` | `Bool` | 是否已完成 Init |
| `block_size_` | `I32` | 每块 token 数 |
| `block_bytes_` | `I64` | 单块单边（K 或 V）字节数 |
| `total_blocks_` | `I64` | 物理块总数 |
| `max_total_tokens_` | `I64` | `total_blocks × block_size` |
| `free_blocks_` | `vector<I32>` | 空闲物理块 |
| `k_buffers_/v_buffers_` | `vector<void*>` | 每层 K/V 的 `Buffer*`（非拥有）；原 `k_bases_/v_bases_` 由该字段替代 |
| `tables_` | `map<U64, RequestEntry>` | 请求块表 |

```cpp
struct RequestEntry {
  std::vector<I32> blocks;  // 逻辑块 → 物理块
  I64 capacity_tokens;      // 已预留 token 容量
  I32 ref_cnt = 1;          // 阶段二共享预留
};
```

### 4.2 接口语义

| 接口 | 成功 | 失败 |
| --- | --- | --- |
| `Init` | 分配完成、视图可构建 | `kInvalidArgument`/`kUnsupportedOp`/`kUnsupportedDtype`/`kDeviceOom`/`kBufferError`/`kInternal` |
| `BuildView` | 写入 layers（Buffer*/stride/num_blocks/heads/dtype）、block_size、max_total_tokens | `kInvalidArgument`（空视图）/`kStateError` |
| `Stats` | 返回统计；未初始化返回全 0 | 无 |
| `Reserve` | 容量满足、块表可追加 | `kInvalidArgument`（new_tokens≤0 或溢出）/`kDeviceOom`/`kStateError` |
| `Release` | 归还块并删除表项 | 无（幂等） |
| `GetBlockTable` | 拷贝块表 | `kInvalidArgument`/`kStateError`/`kRequestNotFound` |
| `GetSlots` | 写入 count 个 slot | `kInvalidArgument`/`kStateError`/`kRequestNotFound`/`kInternal`（块表损坏） |

### 4.3 容量与布局

- 单块单边字节数：
  `block_bytes = block_size × num_kv_heads × head_dim × DataTypeSize(dtype)`。
- 每块物理成本（所有层 K+V）：
  `per_block_pair = block_bytes × 2 × num_layers`。
- 物理块数：
  `blocks = kv_budget_bytes / per_block_pair`。
- 后端单缓冲限制：
  `blocks = min(blocks, max_buffer_bytes / block_bytes)`（当
  `max_buffer_bytes > 0`）。
- `KvCacheView.layers[i].block_stride = block_bytes`；Attention 通过
  `slot` 与块步长计算地址。阶段一固定 NHD 块主序：
  `offset = block_id×block_stride + token_in_block×num_kv_heads×head_dim×dtype_size + kv_head×head_dim×dtype_size`。
- `Init` 读取 `BackendCapability.kv_layout`；非 `kNhd` 直接拒绝，避免
  KVCache 与 Kernel 对布局的理解不一致。

### 4.4 异常与错误处理

| 场景 | 处理 |
| --- | --- |
| 重复 Init | `kStateError`，不触碰既有缓冲 |
| CreateBuffer 第 N 次失败 | 销毁前 N-1 个成功缓冲，返回 `kBufferError`（前缀保留原码），不登记 ResourceStack |
| 返回空 Buffer | 销毁已创建缓冲，`kInternal` |
| Reserve 超容量 | 校验先于提交，块表/空闲池不变 |
| Release 未知请求 | 静默成功（幂等） |
| GetSlots 越界 | `kInvalidArgument`，不修改输出 |
| 块表与容量不一致 | `kInternal`（视为内部损坏） |

### 4.5 性能与内存约束

- Reserve/Release 均为 O(新增/归还块数)，无全局排序。
- `GetBlockTable` 每次拷贝整个块表；Scheduler 每步对每个请求调用一次，
  块数通常为个位数到数百。
- `GetSlots` 每次分配 count 个 I64；chunk 大小受单步预算限制。
- 每层 K/V 各一个缓冲，总字节数 ≤ 预算；单缓冲 ≤ `max_buffer_bytes`。
- `k_buffers_/v_buffers_` 只为 BuildView 缓存 `Buffer*`，不持有所有权。

### 4.6 扩展点与限制

1. **共享/COW**：`RequestEntry.ref_cnt` 已预留；实现时需增加块级引用计数与
   复制路径，`Release` 改为递减引用。
2. **滑窗/MLA**：当前拒绝非 `kFullAttention`；阶段三在 `Init` 分支中扩展。
3. **视图生命周期**：`ResourceStack::Clear` 后 `Buffer*` 失效；Engine 必须在
   Destroy 阶段清理栈后不再调用 `BuildView`。
4. **线程安全**：无锁；`Merge/Stop` 不会直接调用 KVCache，Scheduler 在锁内
   串行化 Reserve/Release。
5. **块号空间**：所有层共享块号；如需分层独立块表，必须同步修改
   `KvCacheView` 与 Executor 契约。

## 5. 实施步骤

### 5.1 阶段划分

| 阶段 | 任务 | 产出 |
| --- | --- | --- |
| P1 | 阅读 `KvAllocator`/`KvCacheView`/Backend Buffer 契约 | 设计 4.1/4.2 |
| P2 | 实现 Init 容量计算、缓冲分配、ResourceStack 登记 | 可构建空池 |
| P3 | 实现 Reserve/Release/GetBlockTable/GetSlots | 块分配可用 |
| P4 | 实现 BuildView/Stats 与失败回滚 | 视图与错误路径完整 |
| P5 | 编写 Fake Backend 单测并记录结果 | `tests/` |

### 5.2 依赖与里程碑

- 依赖 `Runtime/backend/backend.h`、`Runtime/common/{kv_allocator,kv_view,
  resource_stack,data_types,status}.h`。
- 里程碑 1：容量/视图公式测试通过。
- 里程碑 2：Reserve/Release/slot 测试通过。
- 里程碑 3：预算/上限/回滚/错误码测试通过。

### 5.3 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| `max_buffer_bytes` 小于单块 | 无法分配 | 返回 `kDeviceOom`，由 Engine 报错 |
| 块表损坏 | slot 越界 | `kInternal`，不写越界内存 |
| 共享块提前释放 | 数据损坏 | 当前不实现共享；ref_cnt 预留 |
| ResourceStack 提前清理 | 视图悬空 | 文档约束 + Engine 固定销毁顺序 |

## 6. 测试与验证设计

### 6.1 测试策略

- 主机 Fake Backend：CreateBuffer 分配 `std::vector<U8>`，统计创建/销毁次数，
  支持注入第 N 次失败与 `max_buffer_bytes`。
- 以值断言验证容量、视图、块表、slot 与错误码。
- 通过 `ResourceStack::Clear` 验证缓冲释放与幂等。

### 6.2 用例设计

| 用例 | 覆盖需求 | 内容 |
| --- | --- | --- |
| TC-KV-01 | FR-KV-01/02/03/04 | 2 层、块大小 4、预算 1024 → 2 块；视图/统计/4 个缓冲正确 |
| TC-KV-02 | FR-KV-05/06/07/08 | Reserve 5/1、越界 OOM、slot 公式、Release 幂等、满容量 |
| TC-KV-03 | FR-KV-02/09/10 | 预算 511/512、max_buffer_bytes=200/100、零块大小/零层/未知 dtype/滑窗 |
| TC-KV-04 | FR-KV-03 | 第 4 次 CreateBuffer 失败 → 3 个缓冲回滚；重试成功并全部释放 |
| TC-KV-05 | FR-KV-04/11/12/13 | 视图字段逐项断言；非 NHD 拒绝；不支持 dtype 拒绝；`Buffer*` 与分配句柄一致 |

### 6.3 测试环境与命令

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

通过标准：退出码 0，`failures=0`。实际输出见
`Runtime/kvcache/tests/test_results.md`。

### 6.4 未覆盖项与残余风险

- 未测试真实后端（OCL/CUDA）的缓冲对齐与布局。
- 未测试共享/COW、滑窗、MLA。
- 未测试多线程或并发 Reserve/Release。
- 未做长时间压力/碎片测试。
- **Backend 生命周期依赖**：释放闭包捕获 `Backend*` 并在
  `ResourceStack::Clear` 时调用 `DestroyBuffer`，因此 Backend 必须比资源栈
  活得更久。正常 Destroy 顺序满足该约束；但 `Executor::Init` 失败时
  Executor 会释放 Backend，而 `ResourceStack` 仍可能持有 KV/加载器闭包，
  Engine 回滚将 `backend` 置空后 Destroyer 仍会 Clear 资源栈。该契约缺口
  需要 Engine/Executor 或资源栈所有权模型修复（见 Destroy 文档 6.4），
  不属于 KVCache 内部可实现范围。

## 附录 A：需求-设计-测试追溯

| 需求 | 设计章节 | 测试用例 |
| --- | --- | --- |
| FR-KV-01 | 4.3 | TC-KV-01 |
| FR-KV-02 | 3.3.1 | TC-KV-01/03 |
| FR-KV-03 | 3.3.1/4.4 | TC-KV-04 |
| FR-KV-04 | 4.3 | TC-KV-01 |
| FR-KV-05 | 3.3.2 | TC-KV-02 |
| FR-KV-06 | 4.2 | TC-KV-02 |
| FR-KV-07 | 3.3.3 | TC-KV-02 |
| FR-KV-08 | 4.2 | TC-KV-02 |
| FR-KV-09 | 4.4 | TC-KV-03 |
| FR-KV-10 | 4.3 | TC-KV-03 |
| FR-KV-11 | 4.1/4.2 | TC-KV-05 |
| FR-KV-12 | 4.3 | TC-KV-05 |
| FR-KV-13 | 3.3.1 | TC-KV-05 |
| FR-KV-14 | 3.3.3/4.3 | TC-KV-02/05 |
