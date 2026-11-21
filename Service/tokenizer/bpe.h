// 模块：Tokenizer BPE 编码
//
// 职责：
//   对一个预分词片段执行 ByteLevel 映射与 BPE 合并，输出 token ids。
//
// 边界：
//   - 不做 NFC、预分词、特殊 token 处理；
//   - 词表与 merge rank 必须已在加载期校验；
//   - 符号缺失视为模型资产不一致，返回 kInternal。
//
// 依赖：
//   byte_level.h、Runtime/common/status.h、Utils/cpp/types.h。

#ifndef MOBILENET_SERVICE_TOKENIZER_BPE_H_
#define MOBILENET_SERVICE_TOKENIZER_BPE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "Runtime/common/status.h"
#include "Service/tokenizer/byte_level.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {

// 对预分词片段执行 ByteLevel + BPE。
//
// 算法：
//   1. 把片段 UTF-8 字节映射为 ByteLevel 符号串；
//   2. 以每个符号（单个码点）为初始 BPE 单元；
//   3. 反复选择 merge rank 最小的相邻对，并合并所有相邻出现；
//   4. 把最终符号映射为词表 id。
//
// 参数：
//   segment：预分词片段，合法 UTF-8；
//   byte_level：ByteLevel 映射；
//   token_to_id：符号 → id 词表；
//   merge_rank：`first + '\0' + second` → rank；
//   ids：输出参数，非空；追加本片段的 token ids。
// 返回：
//   成功返回 kOk；词表缺符号返回 kInternal；参数非法返回
//   kInvalidArgument。
Status BpeEncode(const std::string& segment, const ByteLevelMap& byte_level,
                 const std::unordered_map<std::string, I64>& token_to_id,
                 const std::unordered_map<std::string, I32>& merge_rank,
                 std::vector<I64>* ids);

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_TOKENIZER_BPE_H_
