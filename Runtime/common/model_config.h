// 模块：Runtime 模型配置
//
// 职责：
//   定义从模型包 manifest 解析出的模型结构参数，供 Scheduler 计算上下文
//   上限、KVCache 计算容量、Executor 校验输入输出。
//
// 边界：
//   - 不包含运行参数（见 runtime_config.h）；
//   - 不解析 manifest；由 Loader 负责解析与校验；
//   - context_length 只是停止上限，不代表 KV 一定能容纳。
//
// 依赖：
//   Runtime/common/data_types.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_MODEL_CONFIG_H_
#define MOBILENET_RUNTIME_COMMON_MODEL_CONFIG_H_

#include <cstdint>
#include <vector>

#include "Runtime/common/data_types.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {

// 模型结构与运行上限。
//
// 功能：
//   Loader 从 manifest 的 model_config/inference_config 填充；Engine 传给
//   Scheduler 与 KVCache。
//
// 边界：
//   - 字段必须与模型包一致，Loader 校验失败即拒绝加载；
//   - dtype 使用后端边界类型 DataType；图内精度由 Graph 的 Precision 表达。
struct ModelConfig {
  // 层数。
  I32 num_layers = 0;

  // Query 头数。
  I32 num_heads = 0;

  // KV 头数（GQA）。
  I32 num_kv_heads = 0;

  // 每头维度。
  I32 head_dim = 0;

  // 隐藏层维度。
  I64 hidden_size = 0;

  // FFN 中间维度。
  I64 intermediate_size = 0;

  // 词表大小。
  I64 vocab_size = 0;

  // 模型支持的最大上下文；仅作为停止上限。
  I64 context_length = 0;

  // RMSNorm epsilon。
  FP32 rms_norm_eps = 0.0f;

  // RoPE theta。
  FP64 rope_theta = 0.0;

  // 权重/激活精度（模型包目标精度）。
  DataType dtype = DataType::kUnknown;

  // 单步 token 预算（manifest inference_config）。
  I32 max_num_tokens = 0;

  // 模型包声明的最大 batch（图 B 维）；并发由 token packing 表达。
  I32 max_batch_size = 0;

  // pad token id。
  I64 pad_token_id = 0;

  // bos token id。
  I64 bos_token_id = 0;

  // 模型默认停止 token id 列表。
  std::vector<I64> stop_token_ids;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_MODEL_CONFIG_H_
