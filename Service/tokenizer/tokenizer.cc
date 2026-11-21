// 模块：Tokenizer 门面与编解码实现
//
// 职责：
//   组合配置加载、added token 扫描、NFC、预分词、BPE、解码与流式解码，
//   实现 Tokenizer 公共接口。
//
// 边界：
//   不处理 Engine/Session；不实现网络与文件写入；所有错误通过 Status 返回。
//
// 主要逻辑：
//   1. Encode 先在原始文本上按 trie 最长匹配 added tokens，再对普通片段
//      做 NFC、预分词与 BPE；
//   2. Decode 对 added token 直接输出 content，对普通 token 做 ByteLevel
//      反变换后拼接并校验 UTF-8；
//   3. StreamDecoder 只输出完整 UTF-8 前缀，保留不完整字节。

#include "Service/tokenizer/tokenizer.h"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utf8proc.h"

#include "Service/tokenizer/bpe.h"
#include "Service/tokenizer/chat_template.h"
#include "Service/tokenizer/pretokenizer.h"
#include "Service/tokenizer/tokenizer_config.h"
#include "Service/tokenizer/unicode_util.h"

namespace mobilenet {
namespace service {

namespace tokenizer_internal {

// added token 字节 trie。
//
// 功能：
//   Qwen3 的 added tokens 数量少、内容为短字符串；trie 支持在任意字节
//   位置做最长匹配。
class AddedTokenTrie {
 public:
  // 使用 added token 列表构建 trie。
  void Build(const std::vector<AddedToken>& tokens) {
    nodes_.clear();
    nodes_.push_back(Node{});
    for (I32 index = 0; index < static_cast<I32>(tokens.size()); ++index) {
      I32 node = 0;
      for (const C8 ch : tokens[static_cast<Usize>(index)].content) {
        const U8 byte = static_cast<U8>(ch);
        auto iterator = nodes_[static_cast<Usize>(node)].children.find(byte);
        if (iterator == nodes_[static_cast<Usize>(node)].children.end()) {
          const I32 next = static_cast<I32>(nodes_.size());
          nodes_.push_back(Node{});
          nodes_[static_cast<Usize>(node)].children.emplace(byte, next);
          iterator = nodes_[static_cast<Usize>(node)].children.find(byte);
        }
        node = iterator->second;
      }
      nodes_[static_cast<Usize>(node)].token_index = index;
    }
  }

  // 从 text[position] 开始做最长匹配。
  //
  // 返回：
  //   added token 下标；未匹配返回 -1。length 写入匹配字节数。
  I32 MatchLongest(const std::string& text, Usize position,
                   Usize* length) const {
    I32 node = 0;
    I32 best = -1;
    Usize best_length = 0;
    for (Usize cursor = position; cursor < text.size(); ++cursor) {
      const auto iterator = nodes_[static_cast<Usize>(node)].children.find(
          static_cast<U8>(text[cursor]));
      if (iterator == nodes_[static_cast<Usize>(node)].children.end()) {
        break;
      }
      node = iterator->second;
      if (nodes_[static_cast<Usize>(node)].token_index >= 0) {
        best = nodes_[static_cast<Usize>(node)].token_index;
        best_length = cursor - position + 1;
      }
    }
    if (length != nullptr) {
      *length = best_length;
    }
    return best;
  }

 private:
  // trie 节点。
  struct Node {
    // 子节点：字节 → 节点下标。
    std::unordered_map<U8, I32> children;

    // 终止节点对应的 added token 下标；-1 表示非终止。
    I32 token_index = -1;
  };

  // 节点数组；下标 0 为根。
  std::vector<Node> nodes_;
};

// Tokenizer 内部实现。
//
// 功能：
//   持有加载后的资产与 trie，提供编码/解码辅助方法。
//
// 边界：
//   加载后只读；所有方法可并发调用。
struct TokenizerImpl {
  // 加载后的资产。
  LoadedTokenizerAssets assets;

  // added token 匹配 trie。
  AddedTokenTrie added_trie;

  // id 是否为 added token。
  std::vector<Bool> is_added;

  // id 是否为 special token。
  std::vector<Bool> is_special;

  // 对一段普通文本做 NFC、预分词与 BPE。
  //
  // 参数：
  //   segment：普通文本片段；
  //   ids：输出参数，非空；追加编码结果。
  Status EncodeSegment(const std::string& segment,
                       std::vector<I64>* ids) const {
    std::string normalized = segment;
    if (assets.enable_nfc) {
      Status status = NormalizeNfc(segment, &normalized);
      if (!status.ok()) {
        return PrefixStatus(status, "Tokenizer::Encode");
      }
    }
    std::vector<std::string> pieces;
    Status status = SplitPretoken(normalized, &pieces);
    if (!status.ok()) {
      return PrefixStatus(status, "Tokenizer::Encode");
    }
    for (const std::string& piece : pieces) {
      status = BpeEncode(piece, assets.byte_level, assets.token_to_id,
                         assets.merge_rank, ids);
      if (!status.ok()) {
        return PrefixStatus(status, "Tokenizer::Encode");
      }
    }
    return Status::Ok();
  }

  // 把整段文本编码为 ids。
  Status EncodeText(const std::string& text, const EncodeOptions& options,
                    std::vector<I64>* ids) const {
    (void)options;  // Qwen3 的 ByteLevel post_processor 不追加特殊 token。
    Usize segment_start = 0;
    Usize cursor = 0;
    while (cursor < text.size()) {
      Usize match_length = 0;
      const I32 token_index =
          added_trie.MatchLongest(text, cursor, &match_length);
      if (token_index >= 0) {
        if (cursor > segment_start) {
          Status status = EncodeSegment(
              text.substr(segment_start, cursor - segment_start), ids);
          if (!status.ok()) {
            return status;
          }
        }
        ids->push_back(
            assets.added_tokens[static_cast<Usize>(token_index)].id);
        cursor += match_length;
        segment_start = cursor;
        continue;
      }
      // 前进一个完整 UTF-8 码点；输入已校验，不会失败。
      utf8proc_int32_t value = 0;
      const utf8proc_ssize_t length = utf8proc_iterate(
          reinterpret_cast<const utf8proc_uint8_t*>(text.data()) + cursor,
          static_cast<utf8proc_ssize_t>(text.size() - cursor), &value);
      if (length <= 0) {
        return MakeStatus(ErrorCode::kInvalidArgument,
                          "Tokenizer::Encode: invalid UTF-8 at byte " +
                              std::to_string(cursor));
      }
      cursor += static_cast<Usize>(length);
    }
    if (segment_start < text.size()) {
      Status status = EncodeSegment(text.substr(segment_start), ids);
      if (!status.ok()) {
        return status;
      }
    }
    return Status::Ok();
  }

  // 把单个 token 追加为字节。
  //
  // 参数：
  //   id：token id；
  //   options：解码选项；
  //   bytes：输出参数，非空；追加该 token 的字节表示；
  //   emitted：输出参数，可为空；写入是否实际产生字节。
  Status AppendTokenBytes(I64 id, const DecodeOptions& options,
                          std::string* bytes, Bool* emitted) const {
    if (id < 0 || id >= assets.vocab_size ||
        assets.id_to_token[static_cast<Usize>(id)].empty()) {
      if (options.skip_unknown_tokens) {
        if (emitted != nullptr) {
          *emitted = false;
        }
        return Status::Ok();
      }
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "Tokenizer::Decode: unknown token id " +
                            std::to_string(id));
    }
    const Bool added = is_added[static_cast<Usize>(id)];
    const Bool special = is_special[static_cast<Usize>(id)];
    if (added) {
      if (special && options.skip_special_tokens) {
        if (emitted != nullptr) {
          *emitted = false;
        }
        return Status::Ok();
      }
      bytes->append(assets.id_to_token[static_cast<Usize>(id)]);
      if (emitted != nullptr) {
        *emitted = true;
      }
      return Status::Ok();
    }
    std::string token_bytes;
    Status status =
        ByteLevelDecode(assets.id_to_token[static_cast<Usize>(id)],
                        assets.byte_level, &token_bytes);
    if (!status.ok()) {
      return PrefixStatus(status, "Tokenizer::Decode");
    }
    bytes->append(token_bytes);
    if (emitted != nullptr) {
      *emitted = true;
    }
    return Status::Ok();
  }
};

namespace {

// 计算 UTF-8 lead byte 的期望长度；非法返回 0。
Usize ExpectedUtf8Length(U8 lead) {
  if (lead < 0x80) {
    return 1;
  }
  if (lead >= 0xC2 && lead <= 0xDF) {
    return 2;
  }
  if (lead >= 0xE0 && lead <= 0xEF) {
    return 3;
  }
  if (lead >= 0xF0 && lead <= 0xF4) {
    return 4;
  }
  return 0;
}

// 提取可输出的 UTF-8 前缀；非法字节以 U+FFFD 替换。
//
// 说明：
//   HF ByteLevel decoder 对非法字节是 lossy 替换，不会让整条生成失败。
//   流式解码采用相同策略：非法序列替换后继续，不完整序列留到下一次。
//
// 参数：
//   bytes：输入字节缓冲；
//   output：输出参数，写入可发送文本（含替换字符）；
//   remainder：输出参数，写入不完整 UTF-8 尾部字节。
void ExtractUtf8LossyPrefix(const std::string& bytes, std::string* output,
                            std::string* remainder) {
  output->clear();
  Usize cursor = 0;
  while (cursor < bytes.size()) {
    const Usize expected =
        ExpectedUtf8Length(static_cast<U8>(bytes[cursor]));
    if (expected == 0) {
      output->append("\xEF\xBF\xBD");
      ++cursor;
      continue;
    }
    if (bytes.size() - cursor < expected) {
      break;
    }
    utf8proc_int32_t value = 0;
    const utf8proc_ssize_t length = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(bytes.data()) + cursor,
        static_cast<utf8proc_ssize_t>(expected), &value);
    if (length <= 0) {
      output->append("\xEF\xBF\xBD");
      ++cursor;
      continue;
    }
    output->append(bytes, cursor, static_cast<Usize>(length));
    cursor += static_cast<Usize>(length);
  }
  *remainder = bytes.substr(cursor);
}

}  // namespace
}  // namespace tokenizer_internal

using tokenizer_internal::TokenizerImpl;

Tokenizer::Tokenizer() : impl_(nullptr) {}

Tokenizer::~Tokenizer() = default;

Status Tokenizer::Load(const TokenizerConfig& config,
                       std::unique_ptr<Tokenizer>* tokenizer) {
  if (tokenizer == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Tokenizer::Load: tokenizer must not be null");
  }
  tokenizer_internal::LoadedTokenizerAssets assets;
  Status status = tokenizer_internal::LoadTokenizerAssets(config, &assets);
  if (!status.ok()) {
    return PrefixStatus(status, "Tokenizer::Load");
  }

  auto impl = std::make_shared<TokenizerImpl>();
  impl->assets = std::move(assets);
  impl->is_added.assign(
      static_cast<Usize>(impl->assets.vocab_size), false);
  impl->is_special.assign(
      static_cast<Usize>(impl->assets.vocab_size), false);
  for (const tokenizer_internal::AddedToken& token :
       impl->assets.added_tokens) {
    impl->is_added[static_cast<Usize>(token.id)] = true;
    impl->is_special[static_cast<Usize>(token.id)] = token.special;
  }
  impl->added_trie.Build(impl->assets.added_tokens);

  auto result = std::unique_ptr<Tokenizer>(new Tokenizer());
  result->impl_ = std::move(impl);
  *tokenizer = std::move(result);
  return Status::Ok();
}

Status Tokenizer::Encode(const std::string& text,
                         const EncodeOptions& options,
                         std::vector<I64>* ids) const {
  if (ids == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Tokenizer::Encode: ids must not be null");
  }
  ids->clear();
  if (impl_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "Tokenizer::Encode: tokenizer is not loaded");
  }
  std::string input = text;
  if (impl_->assets.strict_utf8) {
    // added token 在原始文本上匹配；普通片段在 EncodeSegment 中逐段 NFC，
    // 与 HF tokenizers 的 AddedVocabulary 语义一致。
    Status status = tokenizer_internal::ValidateUtf8(text);
    if (!status.ok()) {
      return PrefixStatus(status, "Tokenizer::Encode");
    }
  } else {
    Status status = tokenizer_internal::SanitizeUtf8(text, &input);
    if (!status.ok()) {
      return PrefixStatus(status, "Tokenizer::Encode");
    }
  }
  Status status = impl_->EncodeText(input, options, ids);
  if (!status.ok()) {
    ids->clear();
    return status;
  }
  return Status::Ok();
}

Status Tokenizer::Decode(const std::vector<I64>& ids,
                         const DecodeOptions& options,
                         std::string* text) const {
  if (text == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Tokenizer::Decode: text must not be null");
  }
  text->clear();
  if (impl_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "Tokenizer::Decode: tokenizer is not loaded");
  }
  for (const I64 id : ids) {
    Status status = impl_->AppendTokenBytes(id, options, text, nullptr);
    if (!status.ok()) {
      text->clear();
      return status;
    }
  }
  Status status = tokenizer_internal::ValidateUtf8(*text);
  if (!status.ok()) {
    return PrefixStatus(status, "Tokenizer::Decode");
  }
  return Status::Ok();
}

Status Tokenizer::ApplyChatTemplate(
    const std::vector<ChatMessage>& messages,
    const ChatTemplateOptions& options, std::string* prompt) const {
  if (prompt == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Tokenizer::ApplyChatTemplate: prompt must not be null");
  }
  if (impl_ == nullptr) {
    return MakeStatus(
        ErrorCode::kStateError,
        "Tokenizer::ApplyChatTemplate: tokenizer is not loaded");
  }
  if (!impl_->assets.chat_template_supported) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "Tokenizer::ApplyChatTemplate: Qwen3 chat template not available");
  }
  return tokenizer_internal::FormatQwen3Chat(messages, options, prompt);
}

std::unique_ptr<StreamDecoder> Tokenizer::CreateStreamDecoder(
    const DecodeOptions& options) const {
  if (impl_ == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<StreamDecoder>(new StreamDecoder(impl_, options));
}

I64 Tokenizer::TokenToId(const std::string& token) const {
  if (impl_ == nullptr) {
    return -1;
  }
  const auto iterator = impl_->assets.token_to_id.find(token);
  return iterator == impl_->assets.token_to_id.end() ? -1
                                                     : iterator->second;
}

std::string Tokenizer::IdToToken(I64 id) const {
  if (impl_ == nullptr || id < 0 || id >= impl_->assets.vocab_size) {
    return std::string();
  }
  return impl_->assets.id_to_token[static_cast<Usize>(id)];
}

I64 Tokenizer::bos_token_id() const {
  return impl_ == nullptr ? -1 : impl_->assets.bos_token_id;
}

const std::vector<I64>& Tokenizer::eos_token_ids() const {
  static const std::vector<I64> kEmpty;
  return impl_ == nullptr ? kEmpty : impl_->assets.eos_token_ids;
}

I64 Tokenizer::vocab_size() const {
  return impl_ == nullptr ? 0 : impl_->assets.vocab_size;
}

Bool Tokenizer::supports_chat_template() const {
  return impl_ != nullptr && impl_->assets.chat_template_supported;
}

StreamDecoder::StreamDecoder(
    std::shared_ptr<tokenizer_internal::TokenizerImpl> impl,
    DecodeOptions options)
    : impl_(std::move(impl)), options_(options) {}

StreamDecoder::~StreamDecoder() = default;

Status StreamDecoder::Push(I64 token_id, std::string* delta) {
  if (delta == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "StreamDecoder::Push: delta must not be null");
  }
  delta->clear();
  if (impl_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "StreamDecoder::Push: tokenizer is not loaded");
  }
  std::string bytes;
  Status status = impl_->AppendTokenBytes(token_id, options_, &bytes, nullptr);
  if (!status.ok()) {
    return status;
  }
  pending_bytes_.append(bytes);
  std::string prefix;
  std::string remainder;
  tokenizer_internal::ExtractUtf8LossyPrefix(pending_bytes_, &prefix,
                                             &remainder);
  pending_bytes_ = std::move(remainder);
  *delta = std::move(prefix);
  return Status::Ok();
}

Status StreamDecoder::Finish(std::string* tail) {
  if (tail == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "StreamDecoder::Finish: tail must not be null");
  }
  tail->clear();
  if (!pending_bytes_.empty()) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "StreamDecoder::Finish: incomplete UTF-8 bytes remain");
  }
  return Status::Ok();
}

void StreamDecoder::Reset() { pending_bytes_.clear(); }

}  // namespace service
}  // namespace mobilenet
