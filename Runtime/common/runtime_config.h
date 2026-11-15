// 模块：Runtime 运行配置
//
// 职责：
//   定义 Engine/Loader/KVCache 共用的运行参数，包括后端选择、显存预算、
//   单步 token 预算、KV 块大小与阶段三预留开关。
//
// 边界：
//   - 不包含模型结构参数（见 model_config.h）；
//   - 不解析 JSON；配置加载由上层完成；
//   - 所有字段都有安全默认值，0 表示“由清单或后端自动推导”。
//
// 依赖：
//   C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_RUNTIME_CONFIG_H_
#define MOBILENET_RUNTIME_COMMON_RUNTIME_CONFIG_H_

#include <cstdint>
#include <string>

namespace mobilenet {
namespace runtime {

// Runtime 运行配置。
//
// 功能：
//   由调用方填写后交给 Engine.Load；Loader/KVCache/Scheduler 按字段决定
//   后端、预算与容量。
//
// 边界：
//   - max_num_tokens 不得大于模型图的 S 维规划值（Loader 校验）；
//   - kv_block_size 为 0 时必须由后端 Attention tile 决定；
//   - 阶段三字段（max_concurrent_batches）在 v1 必须为 1。
struct RuntimeConfig {
  // 后端名称，首期 "ocl"。
  std::string backend = "ocl";

  // 设备序号，首期单设备固定 0。
  int32_t device_id = 0;

  // 后端队列数；首期 1。
  uint32_t queue_count = 1;

  // 设备内存预算（字节）；0 表示使用后端/设备默认值。
  uint64_t memory_budget_bytes = 0;

  // KV Cache 预算（字节）；0 表示由总预算减去权重/Activation/IO 后推导。
  uint64_t kv_cache_bytes = 0;

  // 单步 token 预算；0 表示读取 manifest 的 max_num_tokens。
  int32_t max_num_tokens = 0;

  // 并发请求上限；0 表示 min(max_num_tokens, 8)。
  int32_t max_concurrent_requests = 0;

  // KV 块大小（token 数）；0 表示由后端 attention_block_size 决定。
  int32_t kv_block_size = 0;

  // 是否校验权重逐项 SHA-256；默认开启。
  bool verify_weights_sha256 = true;

  // 是否允许加载 metadata-only 包；仅测试使用，默认关闭。
  bool allow_metadata_only = false;

  // 是否启用后台推理线程；默认关闭（单线程 RunOnce）。
  bool threaded = false;

  // 阶段三异步调度的在途 batch 数；v1 必须为 1。
  int32_t max_concurrent_batches = 1;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_RUNTIME_CONFIG_H_
