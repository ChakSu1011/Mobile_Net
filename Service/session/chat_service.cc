// 模块：Session 对话编排实现
//
// 职责：
//   实现 prompt 预算/截断、Engine 请求、事件循环、流式解码、stop 截断、
//   超时与取消。
//
// 边界：
//   不管理会话历史；不解析 HTTP；不写网络。
//
// 主要逻辑：
//   1. Prepare 反复模板化+编码，必要时丢弃旧消息，直到满足 token 预算；
//   2. Run 预注册 request_id，提交 Engine，等待逐步事件；
//   3. 每个 token 经 StreamDecoder 转为文本，再由 StopTextFilter 截断；
//   4. 终态或 stop 命中后取消/清理，返回 finish_reason 与 usage。

#include "Service/session/chat_service.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "Service/session/log.h"

namespace mobilenet {
namespace service {

const C8 kContextLengthExceeded[] = "context_length_exceeded";

namespace {

// 把下标移动到 UTF-8 码点起始位置。
Usize Utf8BoundaryAtOrBefore(const std::string& text, Usize index) {
  if (index >= text.size()) {
    return text.size();
  }
  while (index > 0 &&
         (static_cast<U8>(text[index]) & 0xC0) == 0x80) {
    --index;
  }
  return index;
}

// 追加 UTF-8 替换字符。
void AppendReplacement(std::string* text) {
  text->append("\xEF\xBF\xBD");
}

}  // namespace

StopTextFilter::StopTextFilter(const std::vector<std::string>& stops)
    : stops_(stops) {
  for (const std::string& stop : stops_) {
    max_stop_bytes_ = std::max(max_stop_bytes_, stop.size());
  }
}

Status StopTextFilter::Feed(
    const std::string& delta,
    const std::function<Bool(const std::string&)>& emit, Bool* hit) {
  if (hit == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "StopTextFilter::Feed: hit must not be null");
  }
  *hit = false;
  if (emit == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "StopTextFilter::Feed: emit must not be null");
  }
  pending_.append(delta);
  if (stops_.empty()) {
    if (!pending_.empty()) {
      const std::string text = std::move(pending_);
      pending_.clear();
      if (!emit(text)) {
        return MakeStatus(ErrorCode::kCancelled,
                          "StopTextFilter: client disconnected");
      }
    }
    return Status::Ok();
  }

  Usize found = std::string::npos;
  for (const std::string& stop : stops_) {
    if (stop.empty()) {
      continue;
    }
    const Usize position = pending_.find(stop);
    if (position != std::string::npos &&
        (found == std::string::npos || position < found)) {
      found = position;
    }
  }
  if (found != std::string::npos) {
    if (found > 0) {
      const std::string text = pending_.substr(0, found);
      if (!emit(text)) {
        return MakeStatus(ErrorCode::kCancelled,
                          "StopTextFilter: client disconnected");
      }
    }
    pending_.clear();
    *hit = true;
    return Status::Ok();
  }

  const Usize keep = max_stop_bytes_ > 0 ? max_stop_bytes_ - 1 : 0;
  if (pending_.size() <= keep) {
    return Status::Ok();
  }
  Usize emit_length = Utf8BoundaryAtOrBefore(pending_,
                                             pending_.size() - keep);
  if (emit_length == 0) {
    return Status::Ok();
  }
  const std::string text = pending_.substr(0, emit_length);
  pending_.erase(0, emit_length);
  if (!text.empty() && !emit(text)) {
    return MakeStatus(ErrorCode::kCancelled,
                      "StopTextFilter: client disconnected");
  }
  return Status::Ok();
}

Status StopTextFilter::Flush(
    const std::function<Bool(const std::string&)>& emit) {
  if (pending_.empty()) {
    return Status::Ok();
  }
  const std::string text = std::move(pending_);
  pending_.clear();
  if (emit == nullptr || !emit(text)) {
    return MakeStatus(ErrorCode::kCancelled,
                      "StopTextFilter: client disconnected");
  }
  return Status::Ok();
}

ChatService::ChatService(const Tokenizer& tokenizer, IEngine& engine,
                         const ServerConfig& config,
                         PendingRegistry* registry)
    : tokenizer_(tokenizer),
      engine_(engine),
      config_(config),
      registry_(registry) {}

Bool ChatService::DropOldestMessage(
    std::vector<ChatMessage>* messages) {
  if (messages == nullptr || messages->size() <= 1) {
    return false;
  }
  // 保留首条 system 与最后一条消息；优先丢弃最旧的其它消息。
  for (Usize index = 0; index + 1 < messages->size(); ++index) {
    if (index == 0 && (*messages)[0].role == "system") {
      continue;
    }
    messages->erase(messages->begin() + static_cast<Isize>(index));
    return true;
  }
  // 只剩 system + 最后一条时，最后手段才丢弃 system。
  if ((*messages)[0].role == "system" && messages->size() > 1) {
    messages->erase(messages->begin());
    return true;
  }
  return false;
}

Status ChatService::Prepare(const ChatRequest& request,
                            std::vector<I64>* prompt_ids,
                            I32* max_tokens) {
  if (prompt_ids == nullptr || max_tokens == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ChatService::Prepare: output must not be null");
  }
  if (request.messages.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ChatService::Prepare: messages must not be empty");
  }
  if (request.stop.size() > 4) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ChatService::Prepare: too many stop strings");
  }
  for (const std::string& stop : request.stop) {
    if (stop.empty() || stop.size() > 256) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "ChatService::Prepare: invalid stop string");
    }
  }
  if (request.max_tokens < 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ChatService::Prepare: invalid max_tokens");
  }
  if (registry_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "ChatService::Prepare: registry is null");
  }

  // 编码前做字节级快速预判，避免对明显超限的超长文本执行完整 BPE。
  // 64 字节/token 是宽松上界；正常 Qwen3 prompt 远低于该值。
  U64 estimated_bytes = 0;
  for (const ChatMessage& message : request.messages) {
    estimated_bytes += message.role.size() + message.content.size() + 32;
  }
  const U64 max_prompt_bytes =
      static_cast<U64>(config_.max_context_tokens) * 64;
  if (estimated_bytes > max_prompt_bytes) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        std::string(kContextLengthExceeded) +
            ": prompt bytes exceed max_num_tokens budget");
  }

  I64 requested = request.max_tokens > 0 ? request.max_tokens
                                         : config_.default_max_tokens;
  requested = std::max<I64>(1, requested);

  std::vector<ChatMessage> working = request.messages;
  ChatTemplateOptions options;
  options.add_generation_prompt = true;
  options.thinking = request.thinking;

  I32 dropped = 0;
  while (!working.empty()) {
    std::string prompt;
    Status status = tokenizer_.ApplyChatTemplate(working, options, &prompt);
    if (!status.ok()) {
      return PrefixStatus(status, "ChatService::Prepare");
    }
    std::vector<I64> ids;
    EncodeOptions encode_options;
    status = tokenizer_.Encode(prompt, encode_options, &ids);
    if (!status.ok()) {
      return PrefixStatus(status, "ChatService::Prepare");
    }
    const I64 budget =
        static_cast<I64>(config_.max_context_tokens) -
        static_cast<I64>(ids.size());
    if (budget > 0) {
      if (dropped > 0) {
        LogWarn("chat prompt truncated: dropped " +
                std::to_string(dropped) +
                " messages, prompt_tokens=" + std::to_string(ids.size()));
      }
      *prompt_ids = std::move(ids);
      *max_tokens =
          static_cast<I32>(std::min<I64>(requested, budget));
      return Status::Ok();
    }
    if (!DropOldestMessage(&working)) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          std::string(kContextLengthExceeded) +
              ": prompt exceeds max_context_tokens=" +
              std::to_string(config_.max_context_tokens));
    }
    ++dropped;
  }
  return MakeStatus(ErrorCode::kInvalidArgument,
                    std::string(kContextLengthExceeded) +
                        ": empty prompt after truncation");
}

Bool ChatService::IsStopToken(I64 token_id,
                              const ChatRequest& request) const {
  for (const I64 id : request.stop_token_ids) {
    if (id == token_id) {
      return true;
    }
  }
  for (const I64 id : tokenizer_.eos_token_ids()) {
    if (id == token_id) {
      return true;
    }
  }
  return false;
}

Status ChatService::Run(const ChatRequest& request,
                        const std::vector<I64>& prompt_ids,
                        I32 max_tokens, StreamSink* sink,
                        ChatOutcome* outcome) {
  if (registry_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "ChatService::Run: registry is null");
  }
  if (sink == nullptr && outcome == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ChatService::Run: sink and outcome are both null");
  }

  const U64 request_id = registry_->AllocateId();
  std::shared_ptr<PendingRequest> pending;
  Status status = registry_->Register(request_id, &pending);
  if (!status.ok()) {
    return status;
  }

  runtime::Request engine_request;
  engine_request.request_id = request_id;
  engine_request.prompt_token_ids = prompt_ids;
  engine_request.sample_param.is_valid = true;
  engine_request.sample_param.max_tokens = max_tokens;
  engine_request.sample_param.min_tokens = 0;
  engine_request.sample_param.stop_token_ids = request.stop_token_ids;
  engine_request.sample_param.ignore_eos = false;

  U64 assigned_id = 0;
  status = engine_.AddRequest(engine_request, &assigned_id);
  if (!status.ok()) {
    registry_->Remove(request_id);
    return PrefixStatus(status, "ChatService::Run: AddRequest");
  }
  if (assigned_id != request_id) {
    // PendingRegistry 以预分配 id 为键；id 被改写会导致回调无法路由。
    (void)engine_.CancelRequest(assigned_id);
    registry_->Remove(request_id);
    return MakeStatus(ErrorCode::kInternal,
                      "ChatService::Run: engine reassigned request id");
  }

  std::unique_ptr<StreamDecoder> decoder =
      tokenizer_.CreateStreamDecoder(DecodeOptions{});
  if (decoder == nullptr) {
    (void)engine_.CancelRequest(request_id);
    registry_->Remove(request_id);
    return MakeStatus(ErrorCode::kStateError,
                      "ChatService::Run: decoder is null");
  }

  StopTextFilter filter(request.stop);
  std::string output;
  const auto emit = [&output, sink](const std::string& text) -> Bool {
    if (text.empty()) {
      return true;
    }
    output.append(text);
    if (sink != nullptr && !sink->OnDelta(text)) {
      return false;
    }
    return true;
  };

  const I32 prompt_tokens = static_cast<I32>(prompt_ids.size());
  I32 decoded_tokens = 0;
  I64 last_generated = 0;
  Bool stop_hit = false;
  Bool terminal_seen = false;
  Bool terminal_stop_token = false;
  Status run_status = Status::Ok();

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(
                            config_.request_timeout_ms);

  while (true) {
    std::vector<EngineEvent> batch;
    {
      std::unique_lock<std::mutex> lock(pending->mutex);
      const Bool ready = pending->cv.wait_until(lock, deadline, [&pending] {
        return !pending->events.empty() || pending->terminal ||
               pending->overflow || pending->failed;
      });
      if (!ready) {
        run_status = MakeStatus(ErrorCode::kTimeout,
                                "request_timeout: no engine response");
        break;
      }
      if (pending->failed) {
        run_status = MakeStatus(ErrorCode::kStateError,
                                "engine_unavailable: service stopping");
        break;
      }
      if (pending->overflow) {
        run_status = MakeStatus(ErrorCode::kBufferError,
                                "engine event queue overflow");
        break;
      }
      batch.assign(pending->events.begin(), pending->events.end());
      pending->events.clear();
      terminal_seen = pending->terminal;
    }

    for (const EngineEvent& event : batch) {
      // kIllegal 是终态但没有新 token；先处理错误，避免把上一个
      // 成功 token 再解码一次。
      if (event.status == runtime::RequestStatus::kIllegal) {
        run_status = MakeStatus(ErrorCode::kInternal,
                                "engine_unavailable: request failed");
        break;
      }
      if (event.token_id >= 0) {
        last_generated = event.generated_count;
        const Bool terminal_token =
            event.status == runtime::RequestStatus::kStopped;
        const Bool stop_token =
            terminal_token && IsStopToken(event.token_id, request);
        if (stop_token) {
          terminal_stop_token = true;
        }
        if (!stop_token) {
          std::string delta;
          Status push_status = decoder->Push(event.token_id, &delta);
          if (!push_status.ok()) {
            run_status = PrefixStatus(push_status, "ChatService::Run");
            break;
          }
          ++decoded_tokens;
          Bool hit = false;
          Status feed_status = filter.Feed(delta, emit, &hit);
          if (!feed_status.ok()) {
            run_status = feed_status;
            break;
          }
          if (hit) {
            stop_hit = true;
            break;
          }
        }
      }
    }
    if (!run_status.ok() || stop_hit || terminal_seen) {
      break;
    }
  }

  if (!run_status.ok() ||
      run_status.code() == ErrorCode::kCancelled) {
    (void)engine_.CancelRequest(request_id);
    registry_->Remove(request_id);
    if (run_status.code() == ErrorCode::kTimeout) {
      LogWarn("chat request timeout: request_id=" +
              std::to_string(request_id));
    }
    return run_status;
  }

  std::string finish_reason;
  if (stop_hit) {
    (void)engine_.CancelRequest(request_id);
    finish_reason = "stop";
    LogDebug("chat stop string hit: request_id=" +
             std::to_string(request_id));
  } else if (terminal_seen) {
    std::string tail;
    Status finish_status = decoder->Finish(&tail);
    if (!finish_status.ok()) {
      // 流结束仍有不完整 UTF-8：输出替换字符，避免整段响应失败。
      LogWarn("stream decode ended with incomplete UTF-8: request_id=" +
              std::to_string(request_id));
      AppendReplacement(&tail);
    }
    Bool tail_hit = false;
    Status feed_status = filter.Feed(tail, emit, &tail_hit);
    if (feed_status.ok()) {
      feed_status = filter.Flush(emit);
    }
    if (!feed_status.ok()) {
      registry_->Remove(request_id);
      return feed_status;
    }
    finish_reason =
        (tail_hit || terminal_stop_token ||
         last_generated < static_cast<I64>(max_tokens))
            ? "stop"
            : "length";
  } else {
    (void)engine_.CancelRequest(request_id);
    registry_->Remove(request_id);
    return MakeStatus(ErrorCode::kInternal,
                      "ChatService::Run: request ended without terminal");
  }

  registry_->Remove(request_id);

  const I32 used_completion = decoded_tokens;
  if (sink != nullptr) {
    if (!sink->OnFinish(finish_reason, prompt_tokens, used_completion)) {
      return MakeStatus(ErrorCode::kCancelled,
                        "ChatService::Run: client disconnected");
    }
  }
  if (outcome != nullptr) {
    outcome->text = std::move(output);
    outcome->finish_reason = finish_reason;
    outcome->prompt_tokens = prompt_tokens;
    outcome->completion_tokens = used_completion;
  }
  return Status::Ok();
}

Status ChatService::Complete(const ChatRequest& request,
                             ChatOutcome* outcome) {
  if (outcome == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ChatService::Complete: outcome must not be null");
  }
  std::vector<I64> prompt_ids;
  I32 max_tokens = 0;
  Status status = Prepare(request, &prompt_ids, &max_tokens);
  if (!status.ok()) {
    return status;
  }
  return Run(request, prompt_ids, max_tokens, nullptr, outcome);
}

Status ChatService::Stream(const ChatRequest& request, StreamSink* sink,
                           ChatOutcome* outcome) {
  if (sink == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ChatService::Stream: sink must not be null");
  }
  std::vector<I64> prompt_ids;
  I32 max_tokens = 0;
  Status status = Prepare(request, &prompt_ids, &max_tokens);
  if (!status.ok()) {
    return status;
  }
  return Run(request, prompt_ids, max_tokens, sink, outcome);
}

}  // namespace service
}  // namespace mobilenet
