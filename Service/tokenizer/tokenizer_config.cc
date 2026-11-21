// 模块：Tokenizer 资产加载与校验实现
//
// 职责：
//   解析三个 JSON 资产并构建词表、merges、added tokens、ByteLevel 映射与
//   EOS/BOS 信息。
//
// 边界：
//   不实现编码；不保留 JSON 树；不支持的类型直接报错。
//
// 主要逻辑：
//   1. 读取 tokenizer.json，校验 BPE/ByteLevel/NFC/Qwen3 预分词；
//   2. 构建词表与 merge rank；
//   3. 读取 added tokens 并校验标志；
//   4. 读取 tokenizer_config.json 与 generation_config.json；
//   5. 解析 EOS/BOS 并确定词表大小。

#include "Service/tokenizer/tokenizer_config.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Runtime/common/file_util.h"
#include "Runtime/common/json.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {
namespace {

// tokenizer.json 最大字节数。
constexpr U64 kTokenizerJsonMaxBytes = 64ull * 1024ull * 1024ull;

// 配置文件最大字节数。
constexpr U64 kConfigJsonMaxBytes = 4ull * 1024ull * 1024ull;

// 读取 UTF-8 文本文件。
//
// 参数：
//   path：文件路径；
//   max_bytes：允许的最大字节数；
//   text：输出参数，非空。
// 返回：
//   成功返回 kOk；文件错误返回 kIoError；超限返回 kInvalidModelFile。
Status ReadTextFile(const std::string& path, U64 max_bytes,
                    std::string* text) {
  if (text == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ReadTextFile: text must not be null");
  }
  U64 size = 0;
  Status status = runtime::GetFileSize(path, &size);
  if (!status.ok()) {
    return PrefixStatus(status, "ReadTextFile(" + path + ")");
  }
  if (size > max_bytes) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "ReadTextFile(" + path + "): file too large");
  }
  text->clear();
  if (size == 0) {
    return Status::Ok();
  }
  std::vector<U8> buffer;
  status = runtime::ReadFileRange(path, 0, size, &buffer);
  if (!status.ok()) {
    return PrefixStatus(status, "ReadTextFile(" + path + ")");
  }
  text->assign(reinterpret_cast<const C8*>(buffer.data()), buffer.size());
  return Status::Ok();
}

// 拼接默认路径。
std::string DefaultPath(const std::string& model_dir,
                        const std::string& override_path,
                        const std::string& file_name) {
  if (!override_path.empty()) {
    return override_path;
  }
  if (model_dir.empty()) {
    return std::string();
  }
  if (model_dir.back() == '/' || model_dir.back() == '\\') {
    return model_dir + file_name;
  }
  return model_dir + "/" + file_name;
}

// 解析 JSON 文件。
Status ParseJsonFile(const std::string& path, U64 max_bytes,
                     runtime::JsonValue* value) {
  if (value == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ParseJsonFile: value must not be null");
  }
  std::string text;
  Status status = ReadTextFile(path, max_bytes, &text);
  if (!status.ok()) {
    return status;
  }
  status = runtime::ParseJson(text, value);
  if (!status.ok()) {
    return PrefixStatus(status, "ParseJsonFile(" + path + ")");
  }
  return Status::Ok();
}

// 取得必需的对象成员。
//
// 参数：
//   object：JSON 对象；
//   key：成员名；
//   value：输出参数，非空。
// 返回：
//   成功返回 kOk；缺失或类型不符返回 kInvalidModelFile。
Status RequireMember(const runtime::JsonValue& object, const std::string& key,
                     const runtime::JsonValue** value) {
  if (value == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "RequireMember: value must not be null");
  }
  const runtime::JsonValue* member = object.Find(key);
  if (member == nullptr) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "RequireMember: missing key \"" + key + "\"");
  }
  *value = member;
  return Status::Ok();
}

// 读取可选的布尔字段；缺失返回默认值。
Bool ReadBoolOr(const runtime::JsonValue& object, const std::string& key,
                Bool default_value) {
  const runtime::JsonValue* member = object.Find(key);
  if (member == nullptr || !member->IsBool()) {
    return default_value;
  }
  return member->AsBool();
}

// 读取可选的字符串字段；缺失返回空串。
std::string ReadStringOr(const runtime::JsonValue& object,
                         const std::string& key) {
  const runtime::JsonValue* member = object.Find(key);
  if (member == nullptr || !member->IsString()) {
    return std::string();
  }
  return member->AsString();
}

// 校验 tokenizer.json 的 model 字段。
Status ValidateModelConfig(const runtime::JsonValue& model) {
  const runtime::JsonValue* type = nullptr;
  Status status = RequireMember(model, "type", &type);
  if (!status.ok()) {
    return status;
  }
  if (!type->IsString() || type->AsString() != "BPE") {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: only BPE tokenizer is supported");
  }
  const runtime::JsonValue* dropout = model.Find("dropout");
  if (dropout != nullptr && !dropout->IsNull()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: BPE dropout is not supported");
  }
  const runtime::JsonValue* unk = model.Find("unk_token");
  if (unk != nullptr && !unk->IsNull()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: unk_token is not supported");
  }
  if (ReadBoolOr(model, "byte_fallback", false)) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: byte_fallback is not supported");
  }
  if (ReadBoolOr(model, "ignore_merges", false)) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: ignore_merges is not supported");
  }
  if (!ReadStringOr(model, "continuing_subword_prefix").empty() ||
      !ReadStringOr(model, "end_of_word_suffix").empty()) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "TokenizerConfig: subword prefix/suffix is not supported");
  }
  return Status::Ok();
}

// 校验 normalizer 字段。
Status ValidateNormalizer(const runtime::JsonValue& root) {
  const runtime::JsonValue* normalizer = root.Find("normalizer");
  if (normalizer == nullptr || normalizer->IsNull()) {
    return Status::Ok();
  }
  if (!normalizer->IsObject()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: unsupported normalizer");
  }
  const runtime::JsonValue* type = normalizer->Find("type");
  if (type == nullptr || !type->IsString() || type->AsString() != "NFC") {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: only NFC normalizer is supported");
  }
  return Status::Ok();
}

// 校验预分词配置是否为 Qwen3 的 Split + ByteLevel。
Status ValidatePreTokenizer(const runtime::JsonValue& root) {
  const runtime::JsonValue* pre_tokenizer = root.Find("pre_tokenizer");
  if (pre_tokenizer == nullptr || !pre_tokenizer->IsObject()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: pre_tokenizer is required");
  }
  const runtime::JsonValue* type = pre_tokenizer->Find("type");
  if (type == nullptr || !type->IsString() ||
      type->AsString() != "Sequence") {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: pre_tokenizer must be Sequence");
  }
  const runtime::JsonValue* items = pre_tokenizer->Find("pretokenizers");
  if (items == nullptr || !items->IsArray() ||
      items->AsArray().size() != 2) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "TokenizerConfig: pre_tokenizer must contain Split and ByteLevel");
  }
  const runtime::JsonValue& split = items->AsArray()[0];
  const runtime::JsonValue& byte_level = items->AsArray()[1];
  if (!split.IsObject() || !byte_level.IsObject()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: invalid pretokenizer entry");
  }
  const runtime::JsonValue* split_type = split.Find("type");
  const runtime::JsonValue* byte_type = byte_level.Find("type");
  if (split_type == nullptr || !split_type->IsString() ||
      split_type->AsString() != "Split" || byte_type == nullptr ||
      !byte_type->IsString() || byte_type->AsString() != "ByteLevel") {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: unsupported pretokenizer order");
  }
  const runtime::JsonValue* pattern = split.Find("pattern");
  if (pattern == nullptr || !pattern->IsObject()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: Split pattern must be an object");
  }
  const runtime::JsonValue* regex = pattern->Find("Regex");
  if (regex == nullptr || !regex->IsString()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: Split pattern must be Regex");
  }
  if (regex->AsString().find("\\p{L}") == std::string::npos ||
      regex->AsString().find("\\p{N}") == std::string::npos) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "TokenizerConfig: Split regex is not the Qwen3 pattern");
  }
  const runtime::JsonValue* behavior = split.Find("behavior");
  if (behavior != nullptr &&
      (!behavior->IsString() || behavior->AsString() != "Isolated")) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: Split behavior must be Isolated");
  }
  const runtime::JsonValue* invert = split.Find("invert");
  if (invert != nullptr && (!invert->IsBool() || invert->AsBool())) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: Split invert must be false");
  }
  if (ReadBoolOr(byte_level, "add_prefix_space", false) ||
      ReadBoolOr(byte_level, "use_regex", true)) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "TokenizerConfig: unsupported ByteLevel options");
  }
  return Status::Ok();
}

// 校验 decoder 与 post_processor。
Status ValidateDecoderAndPostProcessor(const runtime::JsonValue& root) {
  const runtime::JsonValue* decoder = root.Find("decoder");
  if (decoder != nullptr && !decoder->IsNull()) {
    const runtime::JsonValue* type = decoder->Find("type");
    if (!decoder->IsObject() || type == nullptr || !type->IsString() ||
        type->AsString() != "ByteLevel") {
      return MakeStatus(ErrorCode::kUnsupportedOp,
                        "TokenizerConfig: only ByteLevel decoder is supported");
    }
  }
  const runtime::JsonValue* post = root.Find("post_processor");
  if (post != nullptr && !post->IsNull()) {
    const runtime::JsonValue* type = post->Find("type");
    if (!post->IsObject() || type == nullptr || !type->IsString() ||
        type->AsString() != "ByteLevel") {
      return MakeStatus(
          ErrorCode::kUnsupportedOp,
          "TokenizerConfig: only ByteLevel post_processor is supported");
    }
    if (ReadBoolOr(*post, "add_prefix_space", false) ||
        ReadBoolOr(*post, "use_regex", true)) {
      return MakeStatus(
          ErrorCode::kUnsupportedOp,
          "TokenizerConfig: unsupported post_processor options");
    }
  }
  return Status::Ok();
}

// 校验 truncation/padding 必须为空。
Status ValidateDisabledFeatures(const runtime::JsonValue& root) {
  const runtime::JsonValue* truncation = root.Find("truncation");
  if (truncation != nullptr && !truncation->IsNull()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: truncation is not supported");
  }
  const runtime::JsonValue* padding = root.Find("padding");
  if (padding != nullptr && !padding->IsNull()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: padding is not supported");
  }
  return Status::Ok();
}

// 解析基础词表。
Status ParseVocab(const runtime::JsonValue& vocab,
                  LoadedTokenizerAssets* assets) {
  if (!vocab.IsObject()) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "TokenizerConfig: model.vocab must be an object");
  }
  I64 max_id = -1;
  for (const runtime::JsonMember& member : vocab.AsObject()) {
    if (!member.value.IsInteger() || member.value.AsInt64() < 0) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: invalid vocab id for \"" +
                            member.key + "\"");
    }
    max_id = std::max(max_id, member.value.AsInt64());
  }
  if (max_id < 0) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "TokenizerConfig: vocab must not be empty");
  }
  assets->id_to_token.assign(static_cast<Usize>(max_id) + 1, std::string());
  for (const runtime::JsonMember& member : vocab.AsObject()) {
    const I64 id = member.value.AsInt64();
    if (!assets->id_to_token[static_cast<Usize>(id)].empty()) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: duplicate vocab id " +
                            std::to_string(id));
    }
    assets->id_to_token[static_cast<Usize>(id)] = member.key;
    if (!assets->token_to_id.emplace(member.key, id).second) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: duplicate vocab token");
    }
  }
  return Status::Ok();
}

// 解析 merges。
Status ParseMerges(const runtime::JsonValue& merges,
                   LoadedTokenizerAssets* assets) {
  if (!merges.IsArray()) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "TokenizerConfig: model.merges must be an array");
  }
  I32 rank = 0;
  for (const runtime::JsonValue& item : merges.AsArray()) {
    std::string first;
    std::string second;
    if (item.IsArray() && item.AsArray().size() == 2 &&
        item.AsArray()[0].IsString() && item.AsArray()[1].IsString()) {
      first = item.AsArray()[0].AsString();
      second = item.AsArray()[1].AsString();
    } else if (item.IsString()) {
      const std::string& text = item.AsString();
      const Usize separator = text.find(' ');
      if (separator == std::string::npos) {
        return MakeStatus(ErrorCode::kInvalidModelFile,
                          "TokenizerConfig: invalid merge entry");
      }
      first = text.substr(0, separator);
      second = text.substr(separator + 1);
    } else {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: invalid merge entry");
    }
    std::string key;
    key.reserve(first.size() + second.size() + 1);
    key.append(first);
    key.push_back('\0');
    key.append(second);
    if (!assets->merge_rank.emplace(std::move(key), rank).second) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: duplicate merge entry");
    }
    ++rank;
  }
  return Status::Ok();
}

// 解析 added tokens。
Status ParseAddedTokens(const runtime::JsonValue& root,
                        LoadedTokenizerAssets* assets) {
  const runtime::JsonValue* added = root.Find("added_tokens");
  if (added == nullptr || added->IsNull()) {
    return Status::Ok();
  }
  if (!added->IsArray()) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "TokenizerConfig: added_tokens must be an array");
  }
  std::unordered_set<I64> ids;
  std::unordered_set<std::string> contents;
  for (const runtime::JsonValue& item : added->AsArray()) {
    if (!item.IsObject()) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: invalid added token");
    }
    const runtime::JsonValue* id = item.Find("id");
    const runtime::JsonValue* content = item.Find("content");
    if (id == nullptr || !id->IsInteger() || id->AsInt64() < 0 ||
        content == nullptr || !content->IsString() ||
        content->AsString().empty()) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: added token needs id/content");
    }
    if (ReadBoolOr(item, "single_word", false) ||
        ReadBoolOr(item, "lstrip", false) ||
        ReadBoolOr(item, "rstrip", false) ||
        ReadBoolOr(item, "normalized", false)) {
      return MakeStatus(
          ErrorCode::kUnsupportedOp,
          "TokenizerConfig: added token flags are not supported");
    }
    AddedToken token;
    token.id = id->AsInt64();
    token.content = content->AsString();
    token.special = ReadBoolOr(item, "special", false);
    if (!ids.insert(token.id).second) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: duplicate added token id");
    }
    if (!contents.insert(token.content).second) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: duplicate added token content");
    }
    assets->added_tokens.push_back(std::move(token));
  }
  return Status::Ok();
}

// 把 token 文本解析为 id。
Status ResolveTokenId(const LoadedTokenizerAssets& assets,
                      const std::string& content, I64* id) {
  if (content.empty()) {
    return Status::Ok();
  }
  const auto iterator = assets.token_to_id.find(content);
  if (iterator == assets.token_to_id.end()) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "TokenizerConfig: token not found: " + content);
  }
  *id = iterator->second;
  return Status::Ok();
}

// 收集整数或整数数组。
Status CollectIds(const runtime::JsonValue& value, std::vector<I64>* ids) {
  if (value.IsInteger()) {
    ids->push_back(value.AsInt64());
    return Status::Ok();
  }
  if (value.IsArray()) {
    for (const runtime::JsonValue& item : value.AsArray()) {
      if (!item.IsInteger() || item.AsInt64() < 0) {
        return MakeStatus(ErrorCode::kInvalidModelFile,
                          "TokenizerConfig: invalid token id array");
      }
      ids->push_back(item.AsInt64());
    }
    return Status::Ok();
  }
  return MakeStatus(ErrorCode::kInvalidModelFile,
                    "TokenizerConfig: token id must be int or array");
}

// 校验 id 是否在词表范围内且有 token。
Status ValidateId(const LoadedTokenizerAssets& assets, I64 id,
                  const std::string& name) {
  if (id < 0 || id >= assets.vocab_size) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "TokenizerConfig: " + name + " id out of range");
  }
  if (assets.id_to_token[static_cast<Usize>(id)].empty()) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "TokenizerConfig: " + name + " id has no token");
  }
  return Status::Ok();
}

}  // namespace

Status LoadTokenizerAssets(const TokenizerConfig& config,
                           LoadedTokenizerAssets* assets) {
  if (assets == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LoadTokenizerAssets: assets must not be null");
  }
  if (config.model_dir.empty() && config.tokenizer_json.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LoadTokenizerAssets: model_dir or tokenizer_json "
                      "must be provided");
  }

  const std::string tokenizer_path =
      DefaultPath(config.model_dir, config.tokenizer_json, "tokenizer.json");
  const std::string tokenizer_config_path =
      DefaultPath(config.model_dir, config.tokenizer_config_json,
                  "tokenizer_config.json");
  const std::string generation_config_path =
      DefaultPath(config.model_dir, config.generation_config_json,
                  "generation_config.json");

  runtime::JsonValue root;
  Status status = ParseJsonFile(tokenizer_path, kTokenizerJsonMaxBytes, &root);
  if (!status.ok()) {
    return status;
  }
  if (!root.IsObject()) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "TokenizerConfig: tokenizer.json must be an object");
  }
  const runtime::JsonValue* model = nullptr;
  status = RequireMember(root, "model", &model);
  if (!status.ok()) {
    return status;
  }
  if (!model->IsObject()) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "TokenizerConfig: model must be an object");
  }
  status = ValidateModelConfig(*model);
  if (!status.ok()) {
    return status;
  }
  status = ValidateNormalizer(root);
  if (!status.ok()) {
    return status;
  }
  status = ValidatePreTokenizer(root);
  if (!status.ok()) {
    return status;
  }
  status = ValidateDecoderAndPostProcessor(root);
  if (!status.ok()) {
    return status;
  }
  status = ValidateDisabledFeatures(root);
  if (!status.ok()) {
    return status;
  }

  LoadedTokenizerAssets loaded;
  loaded.enable_nfc = config.enable_nfc;
  loaded.strict_utf8 = config.strict_utf8;
  loaded.byte_level = MakeByteLevelMap();

  const runtime::JsonValue* vocab = nullptr;
  status = RequireMember(*model, "vocab", &vocab);
  if (!status.ok()) {
    return status;
  }
  status = ParseVocab(*vocab, &loaded);
  if (!status.ok()) {
    return status;
  }
  const runtime::JsonValue* merges = nullptr;
  status = RequireMember(*model, "merges", &merges);
  if (!status.ok()) {
    return status;
  }
  status = ParseMerges(*merges, &loaded);
  if (!status.ok()) {
    return status;
  }
  status = ParseAddedTokens(root, &loaded);
  if (!status.ok()) {
    return status;
  }

  runtime::JsonValue tokenizer_config;
  status = ParseJsonFile(tokenizer_config_path, kConfigJsonMaxBytes,
                         &tokenizer_config);
  if (!status.ok()) {
    return status;
  }
  if (!tokenizer_config.IsObject()) {
    return MakeStatus(
        ErrorCode::kInvalidModelFile,
        "TokenizerConfig: tokenizer_config.json must be an object");
  }
  if (ReadBoolOr(tokenizer_config, "add_bos_token", false)) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "TokenizerConfig: add_bos_token is not supported");
  }
  if (ReadBoolOr(tokenizer_config, "split_special_tokens", false)) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "TokenizerConfig: split_special_tokens is not supported");
  }
  const std::string chat_template =
      ReadStringOr(tokenizer_config, "chat_template");
  loaded.chat_template_supported =
      chat_template.find("<|im_start|>") != std::string::npos &&
      chat_template.find("<|im_end|>") != std::string::npos;

  I64 config_vocab_size = -1;
  runtime::JsonValue generation_config;
  U64 generation_size = 0;
  status = runtime::GetFileSize(generation_config_path, &generation_size);
  if (status.ok()) {
    status = ParseJsonFile(generation_config_path, kConfigJsonMaxBytes,
                           &generation_config);
    if (!status.ok()) {
      return status;
    }
    if (!generation_config.IsObject()) {
      return MakeStatus(
          ErrorCode::kInvalidModelFile,
          "TokenizerConfig: generation_config.json must be an object");
    }
    const runtime::JsonValue* configured_size =
        generation_config.Find("vocab_size");
    if (configured_size != nullptr) {
      if (!configured_size->IsInteger() || configured_size->AsInt64() <= 0) {
        return MakeStatus(ErrorCode::kInvalidModelFile,
                          "TokenizerConfig: invalid generation vocab_size");
      }
      config_vocab_size = configured_size->AsInt64();
    }
  }

  I64 max_id = static_cast<I64>(loaded.id_to_token.size()) - 1;
  for (const AddedToken& token : loaded.added_tokens) {
    max_id = std::max(max_id, token.id);
  }
  if (config_vocab_size > 0 && config_vocab_size < max_id + 1) {
    return MakeStatus(
        ErrorCode::kInvalidModelFile,
        "TokenizerConfig: generation vocab_size smaller than token ids");
  }
  loaded.vocab_size =
      config_vocab_size > 0 ? config_vocab_size : max_id + 1;
  loaded.id_to_token.resize(static_cast<Usize>(loaded.vocab_size));

  for (const AddedToken& token : loaded.added_tokens) {
    if (token.id >= loaded.vocab_size) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: added token id out of range");
    }
    if (!loaded.id_to_token[static_cast<Usize>(token.id)].empty()) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "TokenizerConfig: added token id conflicts with vocab");
    }
    if (loaded.token_to_id.find(token.content) != loaded.token_to_id.end()) {
      return MakeStatus(
          ErrorCode::kInvalidModelFile,
          "TokenizerConfig: added token content conflicts with vocab");
    }
    loaded.id_to_token[static_cast<Usize>(token.id)] = token.content;
    loaded.token_to_id.emplace(token.content, token.id);
  }

  if (generation_config.IsObject()) {
    const runtime::JsonValue* eos = generation_config.Find("eos_token_id");
    if (eos != nullptr) {
      status = CollectIds(*eos, &loaded.eos_token_ids);
      if (!status.ok()) {
        return status;
      }
    }
    const runtime::JsonValue* bos = generation_config.Find("bos_token_id");
    if (bos != nullptr && bos->IsInteger()) {
      loaded.bos_token_id = bos->AsInt64();
    }
  }
  if (loaded.eos_token_ids.empty()) {
    const std::string eos_token =
        ReadStringOr(tokenizer_config, "eos_token");
    I64 eos_id = -1;
    status = ResolveTokenId(loaded, eos_token, &eos_id);
    if (!status.ok()) {
      return status;
    }
    if (eos_id >= 0) {
      loaded.eos_token_ids.push_back(eos_id);
    }
  }
  if (loaded.bos_token_id < 0) {
    const std::string bos_token =
        ReadStringOr(tokenizer_config, "bos_token");
    I64 bos_id = -1;
    status = ResolveTokenId(loaded, bos_token, &bos_id);
    if (!status.ok()) {
      return status;
    }
    loaded.bos_token_id = bos_id;
  }
  for (const I64 id : loaded.eos_token_ids) {
    status = ValidateId(loaded, id, "eos");
    if (!status.ok()) {
      return status;
    }
  }
  if (loaded.bos_token_id >= 0) {
    status = ValidateId(loaded, loaded.bos_token_id, "bos");
    if (!status.ok()) {
      return status;
    }
  }

  *assets = std::move(loaded);
  return Status::Ok();
}

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet
