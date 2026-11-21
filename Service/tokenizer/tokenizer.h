// 模块：Tokenizer 公共接口
//
// 职责：
//   加载 HuggingFace tokenizer.json 资产，提供文本与 token ids 的
//   Encode/Decode、流式 Decode 与 Qwen3 Chat 模板格式化。首个验证模型为
//   Qwen3-0.6B。
//
// 边界：
//   - 只支持 HF BPE + ByteLevel + NFC + Qwen3 预分词正则；
//   - 不支持 SentencePiece/WordPiece/Tiktoken、通用 Jinja 模板、训练；
//   - 不依赖 Runtime Engine，可在进程内独立使用；
//   - 加载完成后对象不可变，const 接口线程安全；
//   - 不抛出异常，全部错误通过 Status 返回。
//
// 依赖：
//   Runtime/common/status.h、Utils/cpp/types.h、内置 utf8proc。

#ifndef MOBILENET_SERVICE_TOKENIZER_TOKENIZER_H_
#define MOBILENET_SERVICE_TOKENIZER_TOKENIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// 一条 Chat 消息。
//
// 边界：
//   role 仅支持 system/user/assistant；content 为纯文本。
struct ChatMessage {
  // 角色。
  std::string role;

  // 文本内容。
  std::string content;
};

// Qwen3 思考模式。
//
// 说明：
//   kDefault 不插入空思考块；kEnabled 同 kDefault；kDisabled 在生成提示
//   后插入 "<think>\n\n</think>\n\n"。
enum class ThinkingMode {
  kDefault = 0,
  kEnabled = 1,
  kDisabled = 2,
};

// Tokenizer 加载配置。
struct TokenizerConfig {
  // 模型目录；默认读取该目录下三个 JSON 文件。
  std::string model_dir;

  // tokenizer.json 路径；非空时覆盖 model_dir 下的默认路径。
  std::string tokenizer_json;

  // tokenizer_config.json 路径；非空时覆盖默认路径。
  std::string tokenizer_config_json;

  // generation_config.json 路径；非空时覆盖默认路径。
  std::string generation_config_json;

  // 是否严格校验输入 UTF-8；false 时非法字节替换为 U+FFFD。
  Bool strict_utf8 = true;

  // 是否执行 NFC 归一化；关闭仅用于调试。
  Bool enable_nfc = true;
};

// Encode 选项。
struct EncodeOptions {
  // 是否追加 post_processor 特殊 token；Qwen3 ByteLevel 无追加。
  Bool add_special_tokens = true;
};

// Decode 选项。
struct DecodeOptions {
  // 是否跳过 special=true 的 added tokens（如 <|im_end|>）。
  Bool skip_special_tokens = true;

  // 是否跳过词表范围外/无 token 的 id；true 时与 HF decode 一致，
  // 这些 id 输出空串。
  Bool skip_unknown_tokens = true;
};

// Chat 模板选项。
struct ChatTemplateOptions {
  // 是否在末尾追加 assistant 生成提示。
  Bool add_generation_prompt = true;

  // 思考模式。
  ThinkingMode thinking = ThinkingMode::kDefault;
};

namespace tokenizer_internal {
struct TokenizerImpl;
}  // namespace tokenizer_internal

// 增量 Detokenizer。
//
// 功能：
//   逐 token 接收 id，输出可安全发送的 UTF-8 文本增量；不完整的多字节
//   序列保留到后续 token。
//
// 边界：
//   - 每个实例只允许单线程使用；
//   - 可以比创建它的 Tokenizer 存活更久（持有内部共享指针）；
//   - Finish 后发现残留不完整 UTF-8 返回 kInvalidArgument。
class StreamDecoder {
 public:
  ~StreamDecoder();

  StreamDecoder(const StreamDecoder&) = delete;
  StreamDecoder& operator=(const StreamDecoder&) = delete;
  StreamDecoder(StreamDecoder&&) = delete;
  StreamDecoder& operator=(StreamDecoder&&) = delete;

  // 追加一个 token 并输出文本增量。
  //
  // 参数：
  //   token_id：词表 id；
  //   delta：输出参数，非空；写入本次可安全发送的 UTF-8 文本，可为空串。
  // 返回：
  //   成功返回 kOk；未知 id 或非法字节返回 kInvalidArgument。
  Status Push(I64 token_id, std::string* delta);

  // 结束流并输出剩余文本。
  //
  // 参数：
  //   tail：输出参数，非空；写入剩余完整文本，可为空串。
  // 返回：
  //   成功返回 kOk；仍有不完整 UTF-8 字节返回 kInvalidArgument。
  Status Finish(std::string* tail);

  // 清空缓冲，复用实例。
  void Reset();

 private:
  friend class Tokenizer;

  // 构造器由 Tokenizer 创建。
  StreamDecoder(std::shared_ptr<tokenizer_internal::TokenizerImpl> impl,
                DecodeOptions options);

  // 内部实现。
  std::shared_ptr<tokenizer_internal::TokenizerImpl> impl_;

  // 解码选项。
  DecodeOptions options_;

  // 尚未形成完整 UTF-8 字符的字节。
  std::string pending_bytes_;
};

// Tokenizer 门面。
//
// 功能：
//   加载模型自带 Tokenizer 资产，提供 Encode/Decode/流式解码/Chat 模板。
//
// 边界：
//   - 加载后不可变；const 方法线程安全；
//   - 不支持非 Qwen3 模板与不支持的 HF Tokenizer 变体；
//   - 所有错误通过 Status 返回。
class Tokenizer {
 public:
  // 加载 Tokenizer 资产。
  //
  // 参数：
  //   config：加载配置；
  //   tokenizer：输出参数，非空；成功时写入新对象。
  // 返回：
  //   成功返回 kOk；文件缺失/损坏返回 kIoError/kInvalidModelFile；
  //   不支持的类型/字段返回 kUnsupportedOp。
  static Status Load(const TokenizerConfig& config,
                     std::unique_ptr<Tokenizer>* tokenizer);

  ~Tokenizer();

  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&&) = delete;
  Tokenizer& operator=(Tokenizer&&) = delete;

  // 文本编码为 token ids。
  //
  // 参数：
  //   text：UTF-8 文本；
  //   options：编码选项；
  //   ids：输出参数，非空；成功时覆盖为编码结果。
  // 返回：
  //   成功返回 kOk；非法 UTF-8/空指针返回 kInvalidArgument；资产不一致
  //   返回 kInternal。
  Status Encode(const std::string& text, const EncodeOptions& options,
                std::vector<I64>* ids) const;

  // token ids 解码为文本。
  //
  // 参数：
  //   ids：token id 序列；
  //   options：解码选项；
  //   text：输出参数，非空；成功时覆盖为解码文本。
  // 返回：
  //   成功返回 kOk；未知 id/空指针返回 kInvalidArgument；词表损坏返回
  //   kInvalidModelFile。
  Status Decode(const std::vector<I64>& ids, const DecodeOptions& options,
                std::string* text) const;

  // 格式化 Qwen3 Chat 模板。
  //
  // 参数：
  //   messages：system/user/assistant 消息列表；
  //   options：生成提示与思考模式；
  //   prompt：输出参数，非空。
  // 返回：
  //   成功返回 kOk；空消息返回 kInvalidArgument；不支持模板/角色返回
  //   kUnsupportedOp。
  Status ApplyChatTemplate(const std::vector<ChatMessage>& messages,
                           const ChatTemplateOptions& options,
                           std::string* prompt) const;

  // 创建流式解码器。
  //
  // 参数：
  //   options：解码选项。
  // 返回：
  //   新解码器；实现保证非空。
  std::unique_ptr<StreamDecoder> CreateStreamDecoder(
      const DecodeOptions& options) const;

  // 查询 token 文本对应的 id。
  //
  // 参数：
  //   token：token 文本。
  // 返回：
  //   找到返回 id；未找到返回 -1。
  I64 TokenToId(const std::string& token) const;

  // 查询 id 对应的 token 文本。
  //
  // 参数：
  //   id：词表 id。
  // 返回：
  //   找到返回文本；未找到返回空串。
  std::string IdToToken(I64 id) const;

  // 返回 BOS id；未配置返回 -1。
  I64 bos_token_id() const;

  // 返回 EOS id 列表；未配置返回空。
  const std::vector<I64>& eos_token_ids() const;

  // 返回词表大小。
  I64 vocab_size() const;

  // 返回是否支持 Qwen3 Chat 模板。
  Bool supports_chat_template() const;

 private:
  Tokenizer();

  // 内部实现；共享指针允许 StreamDecoder 延长生命周期。
  std::shared_ptr<tokenizer_internal::TokenizerImpl> impl_;
};

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_TOKENIZER_TOKENIZER_H_
