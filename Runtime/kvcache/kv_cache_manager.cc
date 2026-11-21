// 模块：Runtime KVCache 默认实现
//
// 职责：
//   实现 IKVCache：按后端 attention block size 与预算计算物理块容量，
//   通过 Backend::CreateBuffer 分配每层 K/V 缓冲并登记到 ResourceStack，
//   维护请求块表、空闲块池与 slot 计算，向 Executor 提供只读视图。
//
// 边界：
//   - 只允许单推理线程使用，不加锁；
//   - 不决定抢占对象、不实现 Attention 计算；
//   - 不绑定任何模型常量；块大小、层数、头数、头维度、精度全部来自
//     KvCacheSpec/BackendCapability；
//   - 阶段二前缀共享与 COW 仅保留内部结构扩展点，本实现不做共享。
//
// 依赖：
//   Runtime/backend/backend.h、Runtime/common/{data_types,kv_allocator,
//   kv_view,resource_stack,status}.h。

#include "Runtime/kvcache/kv_cache_manager.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Runtime/backend/backend.h"
#include "Runtime/common/data_types.h"
#include "Runtime/common/kv_allocator.h"
#include "Runtime/common/kv_view.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// 无符号 64 位乘法溢出检查。
//
// 参数：
//   left、right：乘数；
//   result：输出参数，成功时写入乘积。
// 返回：
//   不溢出返回 true；溢出返回 false 且不修改 result。
Bool MultiplyU64(U64 left, U64 right, U64* result) {
  if (left != 0 && right > std::numeric_limits<U64>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

// 销毁一组已创建的缓冲（Init 失败回滚）。
//
// 参数：
//   backend：创建缓冲的后端，非空；
//   buffers：待销毁缓冲列表，销毁后清空。
// 说明：
//   单个 DestroyBuffer 失败不中断后续销毁；错误由调用方决定是否上报。
void DestroyBuffers(Backend* backend,
                    std::vector<std::unique_ptr<Buffer>>* buffers) {
  for (std::unique_ptr<Buffer>& buffer : *buffers) {
    if (buffer != nullptr) {
      (void)backend->DestroyBuffer(std::move(buffer));
    }
  }
  buffers->clear();
}

// 默认 KVCache 管理器。
//
// 功能：
//   维护物理块池与每个请求的逻辑块表，按 token 增长追加分配；BuildView
//   输出每层 K/V 基址与块步长。
//
// 边界：
//   - 所有块在所有层使用同一物理块号空间，每层独立 K/V 存储区；
//   - Reserve 失败零副作用；Release 幂等且逆序归还块；
//   - 不实现共享/COW；RequestEntry 保留引用计数字段扩展位。
class DefaultKVCache final : public IKVCache {
 public:
  DefaultKVCache() = default;

  ~DefaultKVCache() override = default;

  // 初始化 KV 池。
  //
  // 参数与返回见 IKVCache::Init；另校验 spec 类型必须为 kFullAttention，
  // 预算/块大小不足时返回 kDeviceOom/kInvalidArgument。
  Status Init(Backend* backend, const KvCacheSpec& spec,
              U64 kv_budget_bytes,
              ResourceStack* resource_stack) override;

  // 构建只读视图。
  Status BuildView(KvCacheView* view) const override;

  // 返回 KV 统计。
  KvCacheStats Stats() const override;

  // 为请求追加 new_tokens 的 KV 容量。
  //
  // 返回：
  //   kOk；new_tokens<=0 或溢出返回 kInvalidArgument；块不足返回
  //   kDeviceOom（不修改空闲池与块表）。
  Status Reserve(U64 request_id, I64 new_tokens) override;

  // 释放请求占用的全部块；幂等。
  void Release(U64 request_id) override;

  // 返回当前空闲块数。
  I64 FreeBlocks() const override;

  // 返回每块 token 数。
  I32 BlockSize() const override;

  // 返回最大 token 容量（总块数 × 块大小）。
  I64 MaxTotalTokens() const override;

  // 读取请求块表快照。
  Status GetBlockTable(U64 request_id,
                       std::vector<I32>* table) const override;

  // 计算连续 token 的 KV 写入 slot。
  Status GetSlots(U64 request_id, I64 first_token, I64 count,
                  std::vector<I64>* slots) const override;

 private:
  // 请求级块表。
  //
  // 说明：
  //   capacity_tokens 为已预留的 token 容量；ref_cnt 为阶段二共享预留位，
  //   当前实现恒为 1。
  struct RequestEntry {
    // 逻辑块下标 → 物理块 ID。
    std::vector<I32> blocks;

    // 已预留 token 容量。
    I64 capacity_tokens = 0;

    // 引用计数；阶段二前缀共享预留。
    I32 ref_cnt = 1;
  };

  // 是否已完成 Init。
  Bool initialized_ = false;

  // 每块 token 数。
  I32 block_size_ = 0;

  // KV 头数与每头维度。
  I32 num_kv_heads_ = 0;
  I32 head_dim_ = 0;

  // KV 精度。
  DataType dtype_ = DataType::kUnknown;

  // 单个物理块单边（K 或 V）的字节数。
  I64 block_bytes_ = 0;

  // 物理块总数。
  I64 total_blocks_ = 0;

  // 最大 token 容量。
  I64 max_total_tokens_ = 0;

  // 空闲物理块池。
  std::vector<I32> free_blocks_;

  // 每层 K 缓冲句柄（Buffer*）；指针在 ResourceStack 清理前有效。
  std::vector<void*> k_buffers_;

  // 每层 V 缓冲句柄（Buffer*）；指针在 ResourceStack 清理前有效。
  std::vector<void*> v_buffers_;

  // 请求块表。
  std::map<U64, RequestEntry> tables_;
};

// 初始化 KV 池：校验规格/布局/精度，计算块数，分配每层 K/V 缓冲并登记 ResourceStack。
Status DefaultKVCache::Init(Backend* backend, const KvCacheSpec& spec,
                            U64 kv_budget_bytes,
                            ResourceStack* resource_stack) {
  if (initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "KVCache::Init: already initialized");
  }
  if (backend == nullptr || resource_stack == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "KVCache::Init: backend and resource_stack must not be null");
  }
  if (spec.type != KvCacheType::kFullAttention) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "KVCache::Init: only full attention is supported");
  }
  if (spec.block_size <= 0 || spec.num_layers <= 0 ||
      spec.num_kv_heads <= 0 || spec.head_dim <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KVCache::Init: invalid KV shape fields");
  }
  const U32 element_bytes = DataTypeSize(spec.dtype);
  if (element_bytes == 0) {
    return MakeStatus(ErrorCode::kUnsupportedDtype,
                      "KVCache::Init: unsupported KV dtype");
  }
  if (kv_budget_bytes == 0) {
    return MakeStatus(ErrorCode::kDeviceOom,
                      "KVCache::Init: KV budget is zero");
  }

  BackendCapability capability;
  Status status = backend->Capability(&capability);
  if (!status.ok()) {
    return PrefixStatus(status, "KVCache::Init");
  }
  if (capability.kv_layout != KvLayout::kNhd) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "KVCache::Init: only NHD KV layout is supported");
  }
  if (!capability.supported_dtypes.empty() &&
      std::find(capability.supported_dtypes.begin(),
                capability.supported_dtypes.end(),
                spec.dtype) == capability.supported_dtypes.end()) {
    return MakeStatus(
        ErrorCode::kUnsupportedDtype,
        "KVCache::Init: KV dtype is not supported by backend");
  }

  U64 elements = 0;
  if (!MultiplyU64(static_cast<U64>(spec.block_size),
                   static_cast<U64>(spec.num_kv_heads), &elements) ||
      !MultiplyU64(elements, static_cast<U64>(spec.head_dim), &elements) ||
      !MultiplyU64(elements, static_cast<U64>(element_bytes), &elements)) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KVCache::Init: block byte size overflows");
  }
  const U64 block_bytes = elements;

  U64 per_block_pair = 0;
  if (!MultiplyU64(block_bytes, 2, &per_block_pair) ||
      !MultiplyU64(per_block_pair, static_cast<U64>(spec.num_layers),
                   &per_block_pair)) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KVCache::Init: KV footprint overflows");
  }
  U64 blocks = kv_budget_bytes / per_block_pair;
  if (capability.max_buffer_bytes > 0) {
    const U64 max_blocks_per_buffer = capability.max_buffer_bytes / block_bytes;
    blocks = std::min(blocks, max_blocks_per_buffer);
  }
  if (blocks == 0) {
    return MakeStatus(ErrorCode::kDeviceOom,
                      "KVCache::Init: budget cannot hold one KV block");
  }
  if (blocks >
      static_cast<U64>(std::numeric_limits<I32>::max())) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KVCache::Init: too many KV blocks");
  }
  U64 layer_bytes = 0;
  if (!MultiplyU64(block_bytes, blocks, &layer_bytes) ||
      layer_bytes > static_cast<U64>(std::numeric_limits<I64>::max())) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KVCache::Init: layer buffer size overflows");
  }

  std::vector<std::unique_ptr<Buffer>> k_buffers;
  std::vector<std::unique_ptr<Buffer>> v_buffers;
  k_buffers.reserve(static_cast<Usize>(spec.num_layers));
  v_buffers.reserve(static_cast<Usize>(spec.num_layers));
  BufferDesc desc;
  desc.bytes = layer_bytes;
  desc.alignment = capability.min_alignment;
  desc.usage = BufferUsage::kKv;
  for (I32 layer = 0; layer < spec.num_layers; ++layer) {
    std::unique_ptr<Buffer> k_buffer;
    status = backend->CreateBuffer(desc, &k_buffer);
    if (!status.ok()) {
      DestroyBuffers(backend, &k_buffers);
      DestroyBuffers(backend, &v_buffers);
      return PrefixStatus(status, "KVCache::Init: create K buffer");
    }
    if (k_buffer == nullptr) {
      (void)backend->DestroyBuffer(std::move(k_buffer));
      DestroyBuffers(backend, &k_buffers);
      DestroyBuffers(backend, &v_buffers);
      return MakeStatus(ErrorCode::kInternal,
                        "KVCache::Init: backend returned null K buffer");
    }
    k_buffers.push_back(std::move(k_buffer));

    std::unique_ptr<Buffer> v_buffer;
    status = backend->CreateBuffer(desc, &v_buffer);
    if (!status.ok()) {
      DestroyBuffers(backend, &k_buffers);
      DestroyBuffers(backend, &v_buffers);
      return PrefixStatus(status, "KVCache::Init: create V buffer");
    }
    if (v_buffer == nullptr) {
      (void)backend->DestroyBuffer(std::move(v_buffer));
      DestroyBuffers(backend, &k_buffers);
      DestroyBuffers(backend, &v_buffers);
      return MakeStatus(ErrorCode::kInternal,
                        "KVCache::Init: backend returned null V buffer");
    }
    v_buffers.push_back(std::move(v_buffer));
  }

  k_buffers_.clear();
  v_buffers_.clear();
  k_buffers_.reserve(k_buffers.size());
  v_buffers_.reserve(v_buffers.size());
  for (const std::unique_ptr<Buffer>& buffer : k_buffers) {
    k_buffers_.push_back(buffer.get());
  }
  for (const std::unique_ptr<Buffer>& buffer : v_buffers) {
    v_buffers_.push_back(buffer.get());
  }

  // ResourceStack 的闭包要求可拷贝，而 Buffer 是独占所有权；用 shared_ptr
  // 包一层 unique_ptr，使多个闭包副本共享同一个待释放句柄，首次执行即移出。
  for (Usize layer = 0; layer < k_buffers.size(); ++layer) {
    auto holder =
        std::make_shared<std::unique_ptr<Buffer>>(std::move(k_buffers[layer]));
    resource_stack->Push(
        "kv_cache.k." + std::to_string(layer),
        [backend, holder]() -> Status {
          if (*holder == nullptr) {
            return Status::Ok();
          }
          return backend->DestroyBuffer(std::move(*holder));
        });
  }
  for (Usize layer = 0; layer < v_buffers.size(); ++layer) {
    auto holder =
        std::make_shared<std::unique_ptr<Buffer>>(std::move(v_buffers[layer]));
    resource_stack->Push(
        "kv_cache.v." + std::to_string(layer),
        [backend, holder]() -> Status {
          if (*holder == nullptr) {
            return Status::Ok();
          }
          return backend->DestroyBuffer(std::move(*holder));
        });
  }

  block_size_ = spec.block_size;
  num_kv_heads_ = spec.num_kv_heads;
  head_dim_ = spec.head_dim;
  dtype_ = spec.dtype;
  block_bytes_ = static_cast<I64>(block_bytes);
  total_blocks_ = static_cast<I64>(blocks);
  max_total_tokens_ = total_blocks_ * static_cast<I64>(block_size_);
  free_blocks_.clear();
  free_blocks_.reserve(static_cast<Usize>(total_blocks_));
  for (I64 block_id = 0; block_id < total_blocks_; ++block_id) {
    free_blocks_.push_back(static_cast<I32>(block_id));
  }
  tables_.clear();
  initialized_ = true;
  return Status::Ok();
}

// 构建只读视图：输出每层 Buffer*、块步长、块数、头数/维度/精度与容量。
Status DefaultKVCache::BuildView(KvCacheView* view) const {
  if (view == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KVCache::BuildView: view must not be null");
  }
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "KVCache::BuildView: cache is not initialized");
  }
  view->layers.clear();
  view->layers.resize(k_buffers_.size());
  for (Usize layer = 0; layer < k_buffers_.size(); ++layer) {
    view->layers[layer].k_buffer = k_buffers_[layer];
    view->layers[layer].v_buffer = v_buffers_[layer];
    view->layers[layer].block_stride = block_bytes_;
    view->layers[layer].num_blocks = total_blocks_;
    view->layers[layer].num_kv_heads = num_kv_heads_;
    view->layers[layer].head_dim = head_dim_;
    view->layers[layer].dtype = dtype_;
  }
  view->block_size = block_size_;
  view->max_total_tokens = max_total_tokens_;
  return Status::Ok();
}

// 返回 KV 统计快照。
KvCacheStats DefaultKVCache::Stats() const {
  KvCacheStats stats;
  if (!initialized_) {
    return stats;
  }
  stats.block_size = block_size_;
  stats.total_blocks = total_blocks_;
  stats.free_blocks = static_cast<I64>(free_blocks_.size());
  stats.max_total_tokens = max_total_tokens_;
  return stats;
}

// 为请求追加 token 容量；不足时返回 OOM 且不改动空闲池/块表。
Status DefaultKVCache::Reserve(U64 request_id, I64 new_tokens) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "KVCache::Reserve: cache is not initialized");
  }
  if (new_tokens <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KVCache::Reserve: new_tokens must be positive");
  }
  auto iterator = tables_.find(request_id);
  const I64 capacity = iterator == tables_.end()
                           ? 0
                           : iterator->second.capacity_tokens;
  if (capacity > max_total_tokens_ ||
      new_tokens > max_total_tokens_ - capacity) {
    return MakeStatus(ErrorCode::kDeviceOom,
                      "KVCache::Reserve: request exceeds KV capacity");
  }
  const I64 needed_tokens = capacity + new_tokens;
  const I64 needed_blocks =
      needed_tokens / static_cast<I64>(block_size_) +
      (needed_tokens % static_cast<I64>(block_size_) != 0 ? 1 : 0);
  const I64 current_blocks =
      iterator == tables_.end()
          ? 0
          : static_cast<I64>(iterator->second.blocks.size());
  const I64 extra_blocks = needed_blocks - current_blocks;
  if (needed_blocks > total_blocks_ ||
      extra_blocks > static_cast<I64>(free_blocks_.size())) {
    return MakeStatus(ErrorCode::kDeviceOom,
                      "KVCache::Reserve: no free KV block");
  }

  if (iterator == tables_.end()) {
    iterator = tables_.emplace(request_id, RequestEntry{}).first;
  }
  RequestEntry& entry = iterator->second;
  for (I64 index = 0; index < extra_blocks; ++index) {
    entry.blocks.push_back(free_blocks_.back());
    free_blocks_.pop_back();
  }
  entry.capacity_tokens = needed_tokens;
  return Status::Ok();
}

// 逆序归还请求全部物理块并删除块表；幂等。
void DefaultKVCache::Release(U64 request_id) {
  auto iterator = tables_.find(request_id);
  if (iterator == tables_.end()) {
    return;
  }
  // 逆序归还，便于后续按创建顺序复用块编号。
  for (auto block = iterator->second.blocks.rbegin();
       block != iterator->second.blocks.rend(); ++block) {
    free_blocks_.push_back(*block);
  }
  tables_.erase(iterator);
}

// 返回空闲块数。
I64 DefaultKVCache::FreeBlocks() const {
  return static_cast<I64>(free_blocks_.size());
}

// 返回每块 token 数。
I32 DefaultKVCache::BlockSize() const { return block_size_; }

// 返回 KV 池最大 token 容量。
I64 DefaultKVCache::MaxTotalTokens() const { return max_total_tokens_; }

// 拷贝请求的逻辑块→物理块表。
Status DefaultKVCache::GetBlockTable(U64 request_id,
                                     std::vector<I32>* table) const {
  if (table == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KVCache::GetBlockTable: table must not be null");
  }
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "KVCache::GetBlockTable: cache is not initialized");
  }
  const auto iterator = tables_.find(request_id);
  if (iterator == tables_.end()) {
    return MakeStatus(ErrorCode::kRequestNotFound,
                      "KVCache::GetBlockTable: request not found");
  }
  *table = iterator->second.blocks;
  return Status::Ok();
}

// 计算 [first_token, first_token+count) 的扁平 slot：block*block_size+offset。
Status DefaultKVCache::GetSlots(U64 request_id, I64 first_token, I64 count,
                                std::vector<I64>* slots) const {
  if (slots == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KVCache::GetSlots: slots must not be null");
  }
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "KVCache::GetSlots: cache is not initialized");
  }
  const auto iterator = tables_.find(request_id);
  if (iterator == tables_.end()) {
    return MakeStatus(ErrorCode::kRequestNotFound,
                      "KVCache::GetSlots: request not found");
  }
  const RequestEntry& entry = iterator->second;
  if (first_token < 0 || count < 0 || first_token > entry.capacity_tokens ||
      count > entry.capacity_tokens - first_token) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KVCache::GetSlots: token range is out of capacity");
  }
  slots->assign(static_cast<Usize>(count), -1);
  for (I64 index = 0; index < count; ++index) {
    const I64 token = first_token + index;
    const I64 block_index = token / static_cast<I64>(block_size_);
    const I64 offset = token % static_cast<I64>(block_size_);
    if (block_index >= static_cast<I64>(entry.blocks.size())) {
      return MakeStatus(ErrorCode::kInternal,
                        "KVCache::GetSlots: block table is corrupted");
    }
    (*slots)[static_cast<Usize>(index)] =
        static_cast<I64>(entry.blocks[static_cast<Usize>(block_index)]) *
            static_cast<I64>(block_size_) +
        offset;
  }
  return Status::Ok();
}

}  // namespace

// 创建默认 KVCache 管理器。
std::unique_ptr<IKVCache> CreateDefaultKVCache() {
  return std::make_unique<DefaultKVCache>();
}

}  // namespace runtime
}  // namespace mobilenet
