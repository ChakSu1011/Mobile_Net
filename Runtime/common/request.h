// 模块：Runtime 请求定义
//
// 职责：
//   定义推理请求、采样参数、请求状态与请求快照；请求是 Engine 对外接口与
//   Scheduler 内部队列之间的共同数据对象。
//
// 边界：
//   - 不包含调度策略、KV 块表或 Graph/Backend 类型；
//   - 不持有文本；阶段一/二只处理 token id；
//   - v1 图内采样固定 argmax，采样参数只记录不生效（见字段注释）。
//
// 依赖：
//   C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_REQUEST_H_
#define MOBILENET_RUNTIME_COMMON_REQUEST_H_

#include <cstdint>
#include <vector>

namespace mobilenet {
namespace runtime {

// 请求在 Scheduler 中的生命周期状态。
//
// 说明：
//   kWaiting 等待调度；kRunning 已进入 Prefill/Decode；kPreempted 被抢占后
//   等待重算；kFinished/kCancelled/kFailed 为终态。
enum class RequestState {
  kWaiting = 0,
  kRunning = 1,
  kPreempted = 2,
  kFinished = 3,
  kCancelled = 4,
  kFailed = 5,
};

// 采样参数。
//
// 功能：
//   记录调用方传入的采样配置，供阶段二/三扩展。
//
// 边界：
//   - 阶段一/二图内 Sample 固定 argmax，除 stop_ids 外字段不改变引擎行为；
//   - 实现方必须在文档与日志中显式声明该限制，禁止静默忽略。
struct SamplingParams {
  // 温度；v1 不生效。
  double temperature = 0.0;

  // Top-k；v1 不生效。
  int32_t top_k = 0;

  // Top-p；v1 不生效。
  double top_p = 0.0;

  // 重复惩罚；v1 不生效。
  double repetition_penalty = 1.0;

  // 随机种子；v1 不生效。
  uint64_t seed = 0;

  // 额外停止 token；与模型 stop_token_ids 合并使用。
  std::vector<int64_t> stop_ids;
};

// 一次生成请求。
//
// 功能：
//   携带 prompt、生成上限、采样参数、优先级与超时，由 Service/Engine 构造，
//   交给 Scheduler.Merge。
//
// 边界：
//   - prompt_ids 不得为空；token id 合法性由 Scheduler 校验；
//   - max_new_tokens 为 0 时由 Scheduler 按上下文上限推导；
//   - 请求对象不持有调度状态或 KV 资源。
struct Request {
  // 调用方指定的请求 ID；0 表示由 Engine 自动分配。
  uint64_t id = 0;

  // prompt token ids，长度必须大于 0。
  std::vector<int64_t> prompt_ids;

  // 最大生成 token 数；0 表示按上下文上限推导。
  int32_t max_new_tokens = 0;

  // 采样参数；v1 仅 stop_ids 生效。
  SamplingParams sampling;

  // 优先级，数值越大越优先；默认 0。
  int32_t priority = 0;

  // 截止时间（单调时钟纳秒）；0 表示不超时。
  uint64_t deadline_ns = 0;
};

// 请求状态快照。
//
// 功能：
//   供 Service 查询、日志与测试断言；不暴露 Scheduler 内部结构。
//
// 边界：
//   - 只读；不包含 token 内容；
//   - computed_tokens 为已计算（含 KV）的 token 数，不等于已生成数。
struct RequestSnapshot {
  // 请求 ID。
  uint64_t request_id = 0;

  // 当前状态。
  RequestState state = RequestState::kWaiting;

  // prompt token 总数。
  int32_t prompt_tokens = 0;

  // 已计算的 token 数（prompt + 已计算的生成 token）。
  int32_t computed_tokens = 0;

  // 已生成的 token 数。
  int32_t generated_tokens = 0;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_REQUEST_H_
