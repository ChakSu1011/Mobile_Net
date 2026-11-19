// 模块：Runtime 请求、响应与序列对象
//
// 职责：
//   定义 Service → Engine → Scheduler 链路共用的请求/响应模型：
//   Request、Response、RequestStatus、SampleParams、Token、Sequence、
//   RequestSnapshot 与多模态输入描述。
//
// 边界：
//   - 只描述数据，不包含调度、采样、KV 或设备逻辑；
//   - 请求状态枚举命名为 RequestStatus，避免与错误类型 mobilenet::Status
//     冲突；
//   - 不持有设备资源；Sequence.logits 为借用指针，生命周期由 Executor 与
//     Response 回调约定；
//   - 阶段一/二只使用单序列、argmax、无 Beam；其余字段预留阶段三。
//
// 依赖：
//   Runtime/common/data_types.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_SEQUENCE_H_
#define MOBILENET_RUNTIME_COMMON_SEQUENCE_H_

#include <cstdint>
#include <vector>

#include "Runtime/common/data_types.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {

// 请求处理状态。
//
// 说明：
//   kInvalid 无效/未初始化；kWaiting 等待调度；kRunning 运行中；
//   kStopped 已停止（EOS/stop/长度/取消）；kRecompute 待重算；
//   kSwapped 已换出；kIllegal 非法/失败；kChunkedPrefill 分块 Prefill。
//   取消暂映射为 kStopped；如需区分可在后续版本增加 kCancelled。
enum class RequestStatus {
  kInvalid = 0,
  kWaiting = 1,
  kRunning = 2,
  kStopped = 3,
  kRecompute = 4,
  kSwapped = 5,
  kIllegal = 6,
  kChunkedPrefill = 7,
};

// 采样策略与生成约束。
//
// 功能：
//   由 Service 在 Request 中指定，并随 Response 回传；Scheduler/Engine 读取。
//
// 边界：
//   - 阶段一/二生效：max_tokens/min_tokens/stop_token_ids/ignore_eos；
//   - temperature/presence_penalty/frequency_penalty/num_beam/top_p/top_k/
//     logprobs 记录但不改变图内固定 argmax 的行为，阶段三启用；
//   - is_valid=false 时使用引擎默认参数。
struct SampleParams {
  // 是否启用本组参数。
  Bool is_valid = false;

  // 温度；阶段三。
  F32 temperature = 0.0f;

  // 出现惩罚；阶段三。
  F32 presence_penalty = 0.0f;

  // 频率惩罚；阶段三。
  F32 frequency_penalty = 0.0f;

  // Beam 宽度；阶段三。
  I32 num_beam = 1;

  // Nucleus 采样阈值；阶段三。
  F32 top_p = 0.0f;

  // Top-k 采样上限；阶段三。
  I32 top_k = 0;

  // 最大生成 token 数；0 表示由 context_length - prompt 推导。
  I32 max_tokens = 0;

  // 最小生成 token 数。
  I32 min_tokens = 0;

  // 额外停止 token。
  std::vector<I64> stop_token_ids;

  // 返回对数概率的数量；0 表示不返回。
  I32 logprobs = 0;

  // 是否忽略 EOS 停止。
  Bool ignore_eos = false;
};

// 多模态输入描述（阶段三）。
//
// 功能：
//   承载图片/音频等原始数据或其编码特征，以及已插入 prompt 的占位 token。
//
// 边界：
//   - 阶段一/二必须为空；非空时 Scheduler 返回不支持；
//   - data 为非拥有主机指针，生命周期由 Service 保证到 Merge 完成。
struct MultimodalInput {
  // 模态类型。
  enum class Modality {
    kImage = 0,
    kAudio = 1,
  };

  // 模态类型。
  Modality modality = Modality::kImage;

  // 原始数据或编码特征指针；允许为空。
  const void* data = nullptr;

  // 数据字节数。
  U64 byte_size = 0;

  // 数据类型。
  DataType dtype = DataType::kUnknown;

  // 数据形状。
  std::vector<I64> shape;

  // 已插入 prompt 的占位 token ID。
  std::vector<I64> placeholder_token_ids;
};

// Prefill/Decode 调度分组。
enum class RequestGroup {
  kPrefill = 0,
  kDecode = 1,
};

// 一次推理请求。
//
// 功能：
//   Service 构造后交给 Engine::AddRequest，Scheduler::Merge 复制并持有。
//
// 边界：
//   - request_id=0 表示由 Scheduler 分配；
//   - prompt_token_ids 不得为空；
//   - multimodal_inputs 阶段一/二必须为空；lora_id/die_id/chunk_size 为
//     预留字段，当前实现按单设备、无 LoRA、默认 chunk 处理。
struct Request {
  // 请求 ID；0 表示由 Scheduler 分配。
  U64 request_id = 0;

  // prompt token 序列。
  std::vector<I64> prompt_token_ids;

  // 采样策略与生成约束。
  SampleParams sample_param;

  // 多模态输入；阶段三预留。
  std::vector<MultimodalInput> multimodal_inputs;

  // LoRA ID；-1 表示无。
  I32 lora_id = -1;

  // Prefill/Decode 调度分组。
  RequestGroup group = RequestGroup::kPrefill;

  // die 编号；0 为默认，多 die 预留。
  I32 die_id = 0;

  // 分块 Prefill 大小；0 表示使用调度默认值。
  I32 chunk_size = 0;

  // 调度优先级；数值越大越优先。
  I32 priority = 0;

  // 截止时间（单调时钟纳秒）；0 表示不超时。
  U64 deadline_ns = 0;
};

// 候选或选中的输出 token。
//
// 边界：
//   - select=true 表示本轮选中；
//   - 阶段一/二 argmax 只产生一个选中 token，token_prob/log_prob 未计算
//     时保持 0。
struct Token {
  // 词表 ID。
  I64 token_id = -1;

  // 概率；未计算时为 0。
  F32 token_prob = 0.0f;

  // 对数概率；未计算时为 0。
  F32 log_prob = 0.0f;

  // 是否为选中 token。
  Bool select = false;
};

// 请求的一条输出序列。
//
// 功能：
//   Scheduler 维护其状态，随 Response 以值快照形式返回。
//
// 边界：
//   - history_token_ids 包含 prompt 与已生成 token；prompt_count 为前缀
//     长度，generated_count 为生成数量；
//   - current_token 为本轮 token；draft_token 为阶段三 draft，未进入 history；
//   - logits 为本步 logits 的主机可见指针，仅在 Response 回调期间有效。
struct Sequence {
  // 内部序列 ID。
  U64 sequence_id = 0;

  // 本步 logits 指针；可为空。
  const void* logits = nullptr;

  // logits 精度。
  DataType logits_dtype = DataType::kUnknown;

  // 本轮 token。
  Token current_token;

  // 阶段三 draft token。
  Token draft_token;

  // prompt + 已生成 token 的完整历史。
  std::vector<I64> history_token_ids;

  // 已生成 token 数。
  I32 generated_count = 0;

  // prompt token 数。
  I32 prompt_count = 0;

  // Beam 累计 logprob。
  F64 cumulative_score = 0.0;

  // 词表大小。
  I32 vocab_size = 0;
};

// 单个请求的单步推理结果。
//
// 功能：
//   Scheduler 在 Update/Stop/Fail 中填充，Engine 通过 ResponseCallback
//   交给 Service。
//
// 边界：
//   - status=kStopped/kIllegal 为终态；
//   - sequences 为本步值快照；Service 需长期保存时必须自行拷贝；
//   - SampleParams 为请求采样参数的副本。
struct Response {
  // 请求 ID。
  U64 request_id = 0;

  // 请求状态。
  RequestStatus status = RequestStatus::kInvalid;

  // 输出序列快照。
  std::vector<Sequence> sequences;

  // 请求采样参数副本。
  SampleParams sample_param;
};

// 请求状态快照。
//
// 功能：
//   供 Scheduler::Snapshot、日志与测试查询，不携带序列内容。
struct RequestSnapshot {
  // 请求 ID。
  U64 request_id = 0;

  // 当前状态。
  RequestStatus status = RequestStatus::kWaiting;

  // prompt token 数。
  I32 prompt_tokens = 0;

  // 已计算 token 数。
  I32 computed_tokens = 0;

  // 已生成 token 数。
  I32 generated_tokens = 0;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_SEQUENCE_H_
