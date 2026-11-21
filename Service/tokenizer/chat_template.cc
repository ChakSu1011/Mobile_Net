// 模块：Tokenizer Qwen3 Chat 模板格式化实现
//
// 职责：
//   复刻 Qwen3 Chat 模板在文本消息场景下的输出。
//
// 边界：
//   不支持 tools、tool_calls、多模态内容；这些输入返回 kUnsupportedOp。
//
// 主要逻辑：
//   1. 找到最后一条 user 消息下标 last_query_index；
//   2. 首条 system 输出 system 块，其余 system 按 user 处理；
//   3. assistant 消息解析 </think> 前后内容，按模板分支输出；
//   4. 按 add_generation_prompt 与 thinking 模式追加生成提示。

#include "Service/tokenizer/chat_template.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {
namespace {

// 删除字符串末尾的换行符。
void TrimTrailingNewlines(std::string* text) {
  while (!text->empty() && text->back() == '\n') {
    text->pop_back();
  }
}

// 删除字符串开头的换行符。
void TrimLeadingNewlines(std::string* text) {
  Usize offset = 0;
  while (offset < text->size() && (*text)[offset] == '\n') {
    ++offset;
  }
  text->erase(0, offset);
}

// 删除字符串首尾的换行符。
void TrimNewlines(std::string* text) {
  TrimLeadingNewlines(text);
  TrimTrailingNewlines(text);
}

// 计算最后一条 user 消息的下标；没有时返回 size - 1。
Usize FindLastQueryIndex(const std::vector<ChatMessage>& messages) {
  for (Usize index = messages.size(); index > 0; --index) {
    if (messages[index - 1].role == "user") {
      return index - 1;
    }
  }
  return messages.empty() ? 0 : messages.size() - 1;
}

// 按模板的 assistant 分支输出一条消息。
void AppendAssistantMessage(const ChatMessage& message, Bool is_last,
                            Bool after_last_query, std::string* prompt) {
  std::string content = message.content;
  std::string reasoning;
  const Usize first_think_end = content.find("</think>");
  if (first_think_end != std::string::npos) {
    std::string head = content.substr(0, first_think_end);
    TrimTrailingNewlines(&head);
    const Usize think_begin = head.rfind("<think>");
    if (think_begin != std::string::npos) {
      head = head.substr(think_begin + 7);
    }
    TrimLeadingNewlines(&head);
    reasoning = head;

    const Usize last_think_end = content.rfind("</think>");
    content = content.substr(last_think_end + 8);
    TrimLeadingNewlines(&content);
  }

  prompt->append("<|im_start|>assistant\n");
  if (after_last_query && (is_last || !reasoning.empty())) {
    prompt->append("<think>\n");
    TrimNewlines(&reasoning);
    prompt->append(reasoning);
    prompt->append("\n</think>\n\n");
    TrimLeadingNewlines(&content);
    prompt->append(content);
  } else {
    prompt->append(content);
  }
  prompt->append("<|im_end|>\n");
}

}  // namespace

Bool IsSupportedChatMessages(const std::vector<ChatMessage>& messages) {
  for (const ChatMessage& message : messages) {
    if (message.role != "system" && message.role != "user" &&
        message.role != "assistant") {
      return false;
    }
  }
  return true;
}

Status FormatQwen3Chat(const std::vector<ChatMessage>& messages,
                       const ChatTemplateOptions& options,
                       std::string* prompt) {
  if (prompt == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "FormatQwen3Chat: prompt must not be null");
  }
  if (messages.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "FormatQwen3Chat: messages must not be empty");
  }
  prompt->clear();

  const Usize last_query_index = FindLastQueryIndex(messages);
  for (Usize index = 0; index < messages.size(); ++index) {
    const ChatMessage& message = messages[index];
    if (message.role == "system" && index == 0) {
      prompt->append("<|im_start|>system\n");
      prompt->append(message.content);
      prompt->append("<|im_end|>\n");
    } else if (message.role == "user" ||
               (message.role == "system" && index != 0)) {
      // Qwen3 模板在非首条 system 时仍使用 message.role 作为角色名。
      prompt->append("<|im_start|>");
      prompt->append(message.role);
      prompt->append("\n");
      prompt->append(message.content);
      prompt->append("<|im_end|>\n");
    } else if (message.role == "assistant") {
      const Bool is_last = index + 1 == messages.size();
      AppendAssistantMessage(message, is_last,
                             index > last_query_index, prompt);
    } else {
      return MakeStatus(
          ErrorCode::kUnsupportedOp,
          "FormatQwen3Chat: unsupported role \"" + message.role + "\"");
    }
  }

  if (options.add_generation_prompt) {
    prompt->append("<|im_start|>assistant\n");
    if (options.thinking == ThinkingMode::kDisabled) {
      prompt->append("<think>\n\n</think>\n\n");
    }
  }
  return Status::Ok();
}

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet
