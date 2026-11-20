// 模块：Runtime KVCache 单元测试
//
// 职责：
//   用 Fake Backend 验证默认 KVCache 的容量计算、BuildView、块池
//   Reserve/Release、slot 计算、预算上限、失败回滚与 ResourceStack 释放。
//
// 边界：
//   - Fake Backend 使用主机内存模拟设备缓冲，不触及真实后端；
//   - 不验证 Attention 数值、共享/COW 与多线程（当前实现不支持）；
//   - 只检查分配语义与错误码。
//
// 依赖：
//   Runtime/kvcache/kv_cache_manager.cc、Runtime/common/{status,
//   resource_stack,data_types}.cc。

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Runtime/backend/backend.h"
#include "Runtime/common/data_types.h"
#include "Runtime/common/kv_allocator.h"
#include "Runtime/common/kv_view.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/status.h"
#include "Runtime/kvcache/kv_cache_manager.h"
#include "Utils/cpp/types.h"

namespace {

using mobilenet::Backend;
using mobilenet::BackendCapability;
using mobilenet::BackendConfig;
using mobilenet::Buffer;
using mobilenet::BufferDesc;
using mobilenet::BufferUsage;
using mobilenet::DataType;
using mobilenet::ErrorCode;
using mobilenet::ErrorCodeName;
using mobilenet::KernelDesc;
using mobilenet::KvLayout;
using mobilenet::MakeStatus;
using mobilenet::ParamDesc;
using mobilenet::Status;
using mobilenet::Stream;
using mobilenet::runtime::CreateDefaultKVCache;
using mobilenet::runtime::IKVCache;
using mobilenet::runtime::KvCacheSpec;
using mobilenet::runtime::KvCacheStats;
using mobilenet::runtime::KvCacheType;
using mobilenet::runtime::KvCacheView;
using mobilenet::runtime::ResourceStack;

// 全局断言计数器。
I32 g_checks = 0;
I32 g_failures = 0;

// 记录一条断言结果。
//
// 参数：
//   condition：断言条件；
//   message：断言描述。
// 返回：
//   无；失败时累加 g_failures 并打印。
void Check(Bool condition, const std::string& message) {
  ++g_checks;
  if (condition) {
    std::cout << "[PASS] " << message << "\n";
    return;
  }
  ++g_failures;
  std::cout << "[FAIL] " << message << "\n";
}

// 记录状态码断言。
//
// 参数：
//   status：实际状态；
//   expected：期望错误码；
//   message：断言描述。
void CheckStatus(const Status& status, ErrorCode expected,
                 const std::string& message) {
  Check(status.code() == expected,
        message + " (got " + ErrorCodeName(status.code()) + ")");
}

// Fake 设备缓冲，底层为主机字节数组。
class FakeBuffer final : public Buffer {
 public:
  // 构造指定字节数的缓冲。
  //
  // 参数：
  //   bytes：缓冲字节数。
  explicit FakeBuffer(U64 bytes)
      : bytes_(bytes), data_(static_cast<Usize>(bytes)) {}

  // 返回主机地址。
  void* DevicePtr() const override { return data_.data(); }

  // 返回字节数。
  U64 Bytes() const override { return bytes_; }

 private:
  // 字节数。
  U64 bytes_ = 0;

  // 主机存储。
  mutable std::vector<U8> data_;
};

// Fake 后端。
//
// 功能：
//   实现 Backend 纯虚接口，CreateBuffer 返回主机缓冲并统计创建/销毁次数；
//   可配置能力上限与指定次数失败，用于容量与回滚测试。
//
// 边界：
//   - 不做真实设备操作与 Kernel；
//   - 仅 KVCache 需要的 Capability/CreateBuffer/DestroyBuffer 有实际行为。
class FakeBackend final : public Backend {
 public:
  // 初始化后端。
  Status Init(const BackendConfig& config) override {
    (void)config;
    initialized_ = true;
    return Status::Ok();
  }

  // 反初始化后端。
  Status Deinit() override {
    initialized_ = false;
    return Status::Ok();
  }

  // 返回能力声明。
  Status Capability(BackendCapability* capability) const override {
    if (capability == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "FakeBackend: capability must not be null");
    }
    capability->supported_dtypes = {DataType::kFloat16};
    capability->supported_ops = {};
    capability->h2d = true;
    capability->d2h = true;
    capability->max_buffer_bytes = max_buffer_bytes;
    capability->min_alignment = min_alignment;
    capability->attention_block_size = attention_block_size;
    capability->kv_layout = KvLayout::kNhd;
    return Status::Ok();
  }

  // 创建主机缓冲。
  Status CreateBuffer(const BufferDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override {
    if (buffer == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "FakeBackend: buffer output must not be null");
    }
    if (desc.bytes == 0) {
      return MakeStatus(ErrorCode::kBufferError,
                        "FakeBackend: zero-byte buffer");
    }
    if (fail_create_at >= 0 && create_calls == fail_create_at) {
      ++create_calls;
      return MakeStatus(ErrorCode::kBufferError,
                        "FakeBackend: forced CreateBuffer failure");
    }
    ++create_calls;
    *buffer = std::make_unique<FakeBuffer>(desc.bytes);
    if (desc.bytes > max_requested_bytes) {
      max_requested_bytes = desc.bytes;
    }
    return Status::Ok();
  }

  // 销毁缓冲。
  Status DestroyBuffer(std::unique_ptr<Buffer> buffer) override {
    if (buffer == nullptr) {
      return Status::Ok();
    }
    ++destroy_calls;
    buffer.reset();
    return Status::Ok();
  }

  // 权重转换；测试不使用。
  Status ConvertParam(const ParamDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override {
    (void)desc;
    (void)buffer;
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "FakeBackend: ConvertParam is not supported");
  }

  // 主机到设备拷贝；测试不使用。
  Status Upload(void* device_ptr, const void* host_ptr, U64 bytes,
                Stream* stream) override {
    (void)device_ptr;
    (void)host_ptr;
    (void)bytes;
    (void)stream;
    return Status::Ok();
  }

  // 设备到主机拷贝；测试不使用。
  Status Download(void* host_ptr, const void* device_ptr, U64 bytes,
                  Stream* stream) override {
    (void)host_ptr;
    (void)device_ptr;
    (void)bytes;
    (void)stream;
    return Status::Ok();
  }

  // 创建 Kernel；测试不使用。
  Status CreateKernel(const KernelDesc& desc,
                      std::unique_ptr<mobilenet::Kernel>* kernel) override {
    (void)desc;
    (void)kernel;
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "FakeBackend: CreateKernel is not supported");
  }

  // 销毁 Kernel；测试不使用。
  Status DestroyKernel(std::unique_ptr<mobilenet::Kernel> kernel) override {
    (void)kernel;
    return Status::Ok();
  }

  // 同步；测试无在途任务。
  Status Synchronize(Stream* stream) override {
    (void)stream;
    return Status::Ok();
  }

  // 后端声明的 KV 块大小。
  I32 attention_block_size = 16;

  // 单缓冲最大字节数；0 表示无上限。
  U64 max_buffer_bytes = 0;

  // 最小对齐。
  U64 min_alignment = 1;

  // CreateBuffer 调用次数。
  I32 create_calls = 0;

  // DestroyBuffer 调用次数。
  I32 destroy_calls = 0;

  // 在第 N 次 CreateBuffer（从 0 计）强制失败；-1 表示不失败。
  I32 fail_create_at = -1;

  // 观察到的最大单次申请字节数。
  U64 max_requested_bytes = 0;

 private:
  // 是否已初始化。
  Bool initialized_ = false;
};

// 构造 KV 规格。
//
// 参数：
//   block_size：每块 token 数；
//   num_layers：层数。
// 返回：
//   block_size、2 个 KV head、head_dim=8、FP16 的规格。
KvCacheSpec MakeSpec(I32 block_size, I32 num_layers) {
  KvCacheSpec spec;
  spec.type = KvCacheType::kFullAttention;
  spec.block_size = block_size;
  spec.num_layers = num_layers;
  spec.num_kv_heads = 2;
  spec.head_dim = 8;
  spec.dtype = DataType::kFloat16;
  return spec;
}

// 计算期望 slot。
//
// 参数：
//   table：块表；
//   block_size：每块 token 数；
//   first_token：起始 token；
//   count：token 数。
// 返回：
//   期望 slot 列表。
std::vector<I64> ExpectedSlots(const std::vector<I32>& table,
                               I32 block_size, I64 first_token, I64 count) {
  std::vector<I64> slots;
  for (I64 index = 0; index < count; ++index) {
    const I64 token = first_token + index;
    const I64 block_index = token / static_cast<I64>(block_size);
    const I64 offset = token % static_cast<I64>(block_size);
    slots.push_back(
        static_cast<I64>(table[static_cast<Usize>(block_index)]) *
            static_cast<I64>(block_size) +
        offset);
  }
  return slots;
}

// 测试容量计算与视图。
void TestCapacityAndView() {
  FakeBackend backend;
  ResourceStack stack;
  std::unique_ptr<IKVCache> cache = CreateDefaultKVCache();
  Check(cache != nullptr, "CreateDefaultKVCache returns non-null");

  // block_bytes = 4 * 2 * 8 * 2 = 128；每块 K+V 双缓冲需 128*2*2 = 512。
  KvCacheStats empty_stats = cache->Stats();
  Check(empty_stats.total_blocks == 0 && empty_stats.block_size == 0,
        "Stats before Init is empty");
  CheckStatus(cache->Reserve(1, 1), ErrorCode::kStateError,
              "Reserve before Init fails");
  KvCacheView early_view;
  CheckStatus(cache->BuildView(&early_view), ErrorCode::kStateError,
              "BuildView before Init fails");

  CheckStatus(cache->Init(&backend, MakeSpec(4, 2), 1024, &stack),
              ErrorCode::kOk, "Init with budget=1024 succeeds");
  const KvCacheStats stats = cache->Stats();
  Check(stats.block_size == 4 && stats.total_blocks == 2 &&
            stats.free_blocks == 2 && stats.max_total_tokens == 8,
        "Stats reports 2 blocks x 4 tokens");
  Check(cache->BlockSize() == 4 && cache->FreeBlocks() == 2 &&
            cache->MaxTotalTokens() == 8,
        "BlockSize/FreeBlocks/MaxTotalTokens match stats");
  Check(stack.size() == 4, "two layers register four buffers");
  Check(backend.create_calls == 4, "backend created four buffers");
  Check(backend.max_requested_bytes == 2 * 128,
        "per-layer buffer holds both blocks (256 bytes)");

  KvCacheView view;
  CheckStatus(cache->BuildView(&view), ErrorCode::kOk,
              "BuildView after Init succeeds");
  Check(view.layers.size() == 2 && view.block_size == 4 &&
            view.max_total_tokens == 8,
        "view reports two layers and capacity");
  Check(view.layers[0].k_base != nullptr &&
            view.layers[0].v_base != nullptr &&
            view.layers[1].k_base != nullptr &&
            view.layers[1].v_base != nullptr,
        "every layer has K/V base pointers");
  Check(view.layers[0].block_stride == 128 &&
            view.layers[1].block_stride == 128,
        "block_stride equals per-block single-side bytes");
  CheckStatus(cache->Init(&backend, MakeSpec(4, 2), 1024, &stack),
              ErrorCode::kStateError, "repeated Init is rejected");
  CheckStatus(cache->BuildView(nullptr), ErrorCode::kInvalidArgument,
              "BuildView rejects null view");
}

// 测试 Reserve/Release 与 slot 计算。
void TestReserveRelease() {
  FakeBackend backend;
  ResourceStack stack;
  std::unique_ptr<IKVCache> cache = CreateDefaultKVCache();
  Check(cache->Init(&backend, MakeSpec(4, 2), 1024, &stack).ok(),
        "Reserve/Release Init succeeds");

  CheckStatus(cache->Reserve(1, 5), ErrorCode::kOk,
              "Reserve 5 tokens succeeds");
  Check(cache->FreeBlocks() == 0, "Reserve consumes all blocks");
  std::vector<I32> table;
  CheckStatus(cache->GetBlockTable(1, &table), ErrorCode::kOk,
              "GetBlockTable succeeds");
  Check(table.size() == 2, "5 tokens use 2 blocks");

  std::vector<I64> slots;
  CheckStatus(cache->GetSlots(1, 0, 5, &slots), ErrorCode::kOk,
              "GetSlots for 5 reserved tokens succeeds");
  Check(slots == ExpectedSlots(table, 4, 0, 5),
        "slots follow block_id*block_size+offset");
  CheckStatus(cache->GetSlots(1, 4, 2, &slots), ErrorCode::kInvalidArgument,
              "GetSlots beyond capacity is rejected");
  CheckStatus(cache->GetSlots(1, -1, 1, &slots), ErrorCode::kInvalidArgument,
              "GetSlots with negative first_token is rejected");
  CheckStatus(cache->GetSlots(2, 0, 1, &slots), ErrorCode::kRequestNotFound,
              "GetSlots for unknown request is rejected");

  CheckStatus(cache->Reserve(1, 1), ErrorCode::kOk,
              "Reserve one more token within existing block succeeds");
  Check(cache->FreeBlocks() == 0,
        "no extra block is consumed for capacity 6");
  CheckStatus(cache->Reserve(1, 3), ErrorCode::kDeviceOom,
              "Reserve beyond capacity returns DeviceOom");
  std::vector<I32> unchanged;
  CheckStatus(cache->GetBlockTable(1, &unchanged), ErrorCode::kOk,
              "block table survives failed Reserve");
  Check(unchanged.size() == 2, "failed Reserve has zero side effects");

  cache->Release(1);
  Check(cache->FreeBlocks() == 2, "Release returns both blocks");
  cache->Release(1);
  Check(cache->FreeBlocks() == 2, "duplicate Release is idempotent");
  CheckStatus(cache->GetBlockTable(1, &table), ErrorCode::kRequestNotFound,
              "GetBlockTable after Release returns NotFound");

  CheckStatus(cache->Reserve(2, 8), ErrorCode::kOk,
              "Reserve exact full capacity succeeds");
  CheckStatus(cache->GetSlots(2, 0, 8, &slots), ErrorCode::kOk,
              "GetSlots covers full capacity");
  Check(slots.size() == 8, "full capacity yields 8 slots");
  cache->Release(2);

  CheckStatus(cache->Reserve(3, 0), ErrorCode::kInvalidArgument,
              "Reserve zero tokens is rejected");
  CheckStatus(cache->Reserve(4, 9), ErrorCode::kDeviceOom,
              "Reserve more than total capacity is rejected");
  Check(cache->FreeBlocks() == 2,
        "failed oversized Reserve leaves the pool unchanged");
}

// 测试预算、后端缓冲上限与非法规格。
void TestBudgetAndLimits() {
  FakeBackend backend;
  ResourceStack stack;

  std::unique_ptr<IKVCache> cache = CreateDefaultKVCache();
  CheckStatus(cache->Init(&backend, MakeSpec(4, 2), 511, &stack),
              ErrorCode::kDeviceOom, "budget below one block returns OOM");

  cache = CreateDefaultKVCache();
  CheckStatus(cache->Init(&backend, MakeSpec(4, 2), 512, &stack),
              ErrorCode::kOk, "budget 512 yields one block");
  Check(cache->Stats().total_blocks == 1,
        "512-byte budget produces exactly one block");
  (void)stack.Clear();

  backend.max_buffer_bytes = 200;
  cache = CreateDefaultKVCache();
  CheckStatus(cache->Init(&backend, MakeSpec(4, 2), 4096, &stack),
              ErrorCode::kOk, "max_buffer_bytes caps block count");
  Check(cache->Stats().total_blocks == 1 &&
            cache->Stats().free_blocks == 1,
        "backend max_buffer_bytes limits pool to one block");
  (void)stack.Clear();

  backend.max_buffer_bytes = 100;
  cache = CreateDefaultKVCache();
  CheckStatus(cache->Init(&backend, MakeSpec(4, 2), 4096, &stack),
              ErrorCode::kDeviceOom,
              "max_buffer_bytes below block size returns OOM");

  backend.max_buffer_bytes = 0;
  cache = CreateDefaultKVCache();
  CheckStatus(cache->Init(nullptr, MakeSpec(4, 2), 1024, &stack),
              ErrorCode::kInvalidArgument, "Init rejects null backend");
  CheckStatus(cache->Init(&backend, MakeSpec(4, 2), 1024, nullptr),
              ErrorCode::kInvalidArgument, "Init rejects null ResourceStack");

  KvCacheSpec invalid = MakeSpec(0, 2);
  CheckStatus(cache->Init(&backend, invalid, 1024, &stack),
              ErrorCode::kInvalidArgument, "Init rejects zero block_size");
  invalid = MakeSpec(4, 0);
  CheckStatus(cache->Init(&backend, invalid, 1024, &stack),
              ErrorCode::kInvalidArgument, "Init rejects zero layers");
  invalid = MakeSpec(4, 2);
  invalid.dtype = DataType::kUnknown;
  CheckStatus(cache->Init(&backend, invalid, 1024, &stack),
              ErrorCode::kUnsupportedDtype, "Init rejects unknown dtype");
  invalid = MakeSpec(4, 2);
  invalid.type = KvCacheType::kSlidingWindow;
  CheckStatus(cache->Init(&backend, invalid, 1024, &stack),
              ErrorCode::kUnsupportedOp,
              "Init rejects non-full-attention cache type");
}

// 测试 CreateBuffer 失败回滚。
void TestCreateBufferRollback() {
  FakeBackend backend;
  ResourceStack stack;
  std::unique_ptr<IKVCache> cache = CreateDefaultKVCache();
  backend.fail_create_at = 3;
  CheckStatus(cache->Init(&backend, MakeSpec(4, 2), 1024, &stack),
              ErrorCode::kBufferError,
              "CreateBuffer failure propagates BufferError");
  Check(backend.create_calls == 4, "backend saw four CreateBuffer calls");
  Check(backend.destroy_calls == 3,
        "three successfully created buffers are rolled back");
  Check(stack.size() == 0, "failed Init leaves no ResourceStack entries");
  Check(cache->Stats().total_blocks == 0,
        "failed Init keeps cache uninitialized");

  backend.fail_create_at = -1;
  CheckStatus(cache->Init(&backend, MakeSpec(4, 2), 1024, &stack),
              ErrorCode::kOk, "Init succeeds after rollback");
  Check(stack.size() == 4, "retry registers four buffers");
  CheckStatus(stack.Clear(), ErrorCode::kOk,
              "ResourceStack releases KV buffers");
  Check(backend.destroy_calls == 7,
        "Clear destroys all buffers including retried ones");
  CheckStatus(stack.Clear(), ErrorCode::kOk,
              "second ResourceStack Clear is idempotent");
  Check(backend.destroy_calls == 7,
        "idempotent Clear does not destroy buffers twice");
}

}  // namespace

// 测试入口。
//
// 返回：
//   全部用例通过返回 0；存在失败断言返回 1。
int main() {
  std::cout << "Runtime KVCache unit test\n";
  TestCapacityAndView();
  TestReserveRelease();
  TestBudgetAndLimits();
  TestCreateBufferRollback();
  std::cout << "checks=" << g_checks << " failures=" << g_failures << "\n";
  return g_failures == 0 ? 0 : 1;
}
