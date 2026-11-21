// 模块：Tokenizer Qwen3 Chat 模板格式化
//
// 职责：
//   按 Qwen3 tokenizer_config.json 中的 Chat 模板语义，把消息列表格式化为
//   模型输入文本。只实现 Qwen3 模板子集，不解释通用 Jinja。
//
// 边界：
//   - 只支持 system/user/assistant 文本消息；tool 与 tool_calls 不支持；
//   - 不做分词；调用方随后使用 Tokenizer::Encode；
//   - 输入消息不合法时返回 Status，不抛出异常。
//
// 依赖：
//   tokenizer.h（ChatMessage/ChatTemplateOptions）、Runtime/common/status.h。

#ifndef MOBILENET_SERVICE_TOKENIZER_CHAT_TEMPLATE_H_
#define MOBILENET_SERVICE_TOKENIZER_CHAT_TEMPLATE_H_

#include <string>
#include <vector>

#include "Runtime/common/status.h"
#include "Service/tokenizer/tokenizer.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {

// 判断消息列表是否符合 Qwen3 Chat 模板支持的范围。
//
// 参数：
//   messages：消息列表。
// 返回：
//   全部为 system/user/assistant 字符串消息返回 true。
Bool IsSupportedChatMessages(const std::vector<ChatMessage>& messages);

// 格式化 Qwen3 Chat 模板。
//
// 参数：
//   messages：消息列表；首条可为 system，其余为 user/assistant；
//   options：是否追加生成提示与思考模式；
//   prompt：输出参数，非空；成功时写入格式化文本。
// 返回：
//   成功返回 kOk；空消息或空指针返回 kInvalidArgument；tool 角色或
//   不支持的模板返回 kUnsupportedOp。
Status FormatQwen3Chat(const std::vector<ChatMessage>& messages,
                       const ChatTemplateOptions& options,
                       std::string* prompt);

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_TOKENIZER_CHAT_TEMPLATE_H_
