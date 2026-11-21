// 模块：Tokenizer Qwen3 预分词扫描器
//
// 职责：
//   实现 Qwen3 / GPT-2 风格正则的切分语义，把一段 UTF-8 文本切成预分词
//   片段，供 ByteLevel 与 BPE 处理。
//
// 边界：
//   - 只做切分，不做 NFC、ByteLevel 或词表查询；
//   - 输入必须是合法 UTF-8；非法输入返回 kInvalidArgument；
//   - 不支持插入/删除片段之外的正则扩展。
//
// 依赖：
//   unicode_util.h、Runtime/common/status.h、Utils/cpp/types.h。

#ifndef MOBILENET_SERVICE_TOKENIZER_PRETOKENIZER_H_
#define MOBILENET_SERVICE_TOKENIZER_PRETOKENIZER_H_

#include <string>
#include <vector>

#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {

// 按 Qwen3 正则语义切分文本。
//
// 正则：
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?\p{L}+
//   |\p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
//
// 参数：
//   text：合法 UTF-8 文本；
//   pieces：输出参数，非空；成功时按顺序写入切分后的 UTF-8 片段。
// 返回：
//   成功返回 kOk；非法 UTF-8 返回 kInvalidArgument；内部不变量破坏返回
//   kInternal。
Status SplitPretoken(const std::string& text, std::vector<std::string>* pieces);

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_TOKENIZER_PRETOKENIZER_H_
