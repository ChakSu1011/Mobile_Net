// 模块：Tokenizer 单元测试入口
//
// 职责：
//   加载 Qwen3-0.6B Tokenizer 资产，对照 tests/data/golden.json 验证
//   Encode/Decode/流式解码/Chat 模板，并覆盖错误路径、并发与性能基线。
//
// 边界：
//   - 只读取本地模型目录与金标准 JSON；资产缺失时返回 77（CTest 跳过）；
//   - 不访问网络、不依赖 Engine；
//   - 性能数值仅记录基线，不作为硬性通过条件。
//
// 用法：
//   tokenizer_test <模型目录> <golden.json>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "Runtime/common/file_util.h"
#include "Runtime/common/json.h"
#include "Service/tokenizer/tokenizer.h"
#include "Utils/cpp/types.h"

namespace {

// 全局失败计数。
I32 g_failures = 0;

// 检查条件并打印失败信息。
void Expect(Bool condition, const std::string& message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message.c_str());
    ++g_failures;
  }
}

// 读取文本文件。
std::string ReadTextFile(const std::string& path) {
  U64 size = 0;
  mobilenet::Status status = mobilenet::runtime::GetFileSize(path, &size);
  if (!status.ok()) {
    std::printf("FAIL: cannot stat %s: %s\n", path.c_str(),
                status.message().c_str());
    ++g_failures;
    return std::string();
  }
  std::vector<U8> buffer;
  status = mobilenet::runtime::ReadFileRange(path, 0, size, &buffer);
  if (!status.ok()) {
    std::printf("FAIL: cannot read %s: %s\n", path.c_str(),
                status.message().c_str());
    ++g_failures;
    return std::string();
  }
  return std::string(reinterpret_cast<const C8*>(buffer.data()), buffer.size());
}

// 把 JSON 整数数组转换为 ids。
std::vector<I64> ReadIds(const mobilenet::runtime::JsonValue& value) {
  std::vector<I64> ids;
  for (const mobilenet::runtime::JsonValue& item : value.AsArray()) {
    ids.push_back(item.AsInt64());
  }
  return ids;
}

// 把 JSON messages 转换为 ChatMessage。
std::vector<mobilenet::service::ChatMessage> ReadMessages(
    const mobilenet::runtime::JsonValue& value) {
  std::vector<mobilenet::service::ChatMessage> messages;
  for (const mobilenet::runtime::JsonValue& item : value.AsArray()) {
    mobilenet::service::ChatMessage message;
    const mobilenet::runtime::JsonValue* role = item.Find("role");
    const mobilenet::runtime::JsonValue* content = item.Find("content");
    if (role != nullptr && role->IsString()) {
      message.role = role->AsString();
    }
    if (content != nullptr && content->IsString()) {
      message.content = content->AsString();
    }
    messages.push_back(std::move(message));
  }
  return messages;
}

// 解析思考模式字符串。
mobilenet::service::ThinkingMode ParseThinking(const std::string& mode) {
  if (mode == "false") {
    return mobilenet::service::ThinkingMode::kDisabled;
  }
  if (mode == "true") {
    return mobilenet::service::ThinkingMode::kEnabled;
  }
  return mobilenet::service::ThinkingMode::kDefault;
}

// 验证单个 Encode/Decode 用例。
void CheckEncodeCase(const mobilenet::service::Tokenizer& tokenizer,
                     const mobilenet::runtime::JsonValue& item) {
  const mobilenet::runtime::JsonValue* name = item.Find("name");
  const mobilenet::runtime::JsonValue* text = item.Find("text");
  const mobilenet::runtime::JsonValue* ids = item.Find("ids");
  const mobilenet::runtime::JsonValue* decoded = item.Find("decoded");
  const mobilenet::runtime::JsonValue* keep_special =
      item.Find("decoded_keep_special");
  if (name == nullptr || text == nullptr || ids == nullptr ||
      decoded == nullptr || keep_special == nullptr) {
    Expect(false, "invalid golden encode case");
    return;
  }
  const std::string case_name = name->AsString();
  const std::vector<I64> expected_ids = ReadIds(*ids);

  mobilenet::service::EncodeOptions encode_options;
  std::vector<I64> actual_ids;
  mobilenet::Status status =
      tokenizer.Encode(text->AsString(), encode_options, &actual_ids);
  Expect(status.ok(), case_name + ": encode status " +
                          status.message());
  Expect(actual_ids == expected_ids, case_name + ": encode ids mismatch");

  mobilenet::service::DecodeOptions decode_options;
  decode_options.skip_special_tokens = true;
  std::string actual_text;
  status = tokenizer.Decode(expected_ids, decode_options, &actual_text);
  Expect(status.ok(), case_name + ": decode status " + status.message());
  Expect(actual_text == decoded->AsString(),
         case_name + ": decode text mismatch");

  mobilenet::service::DecodeOptions keep_special_options;
  keep_special_options.skip_special_tokens = false;
  std::string actual_keep_special;
  status = tokenizer.Decode(expected_ids, keep_special_options,
                            &actual_keep_special);
  Expect(status.ok(),
         case_name + ": decode keep-special status " + status.message());
  Expect(actual_keep_special == keep_special->AsString(),
         case_name + ": decode keep-special mismatch");

  // 流式解码：逐 token 推送，拼接结果与整段解码一致。
  std::unique_ptr<mobilenet::service::StreamDecoder> stream =
      tokenizer.CreateStreamDecoder(decode_options);
  Expect(stream != nullptr, case_name + ": stream decoder is null");
  if (stream != nullptr) {
    std::string streamed;
    for (const I64 id : expected_ids) {
      std::string delta;
      status = stream->Push(id, &delta);
      Expect(status.ok(), case_name + ": stream push status " +
                              status.message());
      streamed.append(delta);
    }
    std::string tail;
    status = stream->Finish(&tail);
    Expect(status.ok(), case_name + ": stream finish status " +
                            status.message());
    streamed.append(tail);
    Expect(streamed == decoded->AsString(),
           case_name + ": streaming decode mismatch");
  }
}

// 验证单个 Chat 模板用例。
void CheckChatCase(const mobilenet::service::Tokenizer& tokenizer,
                   const mobilenet::runtime::JsonValue& item) {
  const mobilenet::runtime::JsonValue* name = item.Find("name");
  const mobilenet::runtime::JsonValue* messages_value =
      item.Find("messages");
  const mobilenet::runtime::JsonValue* add_prompt =
      item.Find("add_generation_prompt");
  const mobilenet::runtime::JsonValue* thinking =
      item.Find("enable_thinking");
  const mobilenet::runtime::JsonValue* prompt = item.Find("prompt");
  if (name == nullptr || messages_value == nullptr || add_prompt == nullptr ||
      thinking == nullptr || prompt == nullptr) {
    Expect(false, "invalid golden chat case");
    return;
  }
  mobilenet::service::ChatTemplateOptions options;
  options.add_generation_prompt = add_prompt->AsBool();
  options.thinking = ParseThinking(thinking->AsString());
  std::string actual_prompt;
  mobilenet::Status status = tokenizer.ApplyChatTemplate(
      ReadMessages(*messages_value), options, &actual_prompt);
  Expect(status.ok(), name->AsString() + ": chat status " +
                          status.message());
  Expect(actual_prompt == prompt->AsString(),
         name->AsString() + ": chat prompt mismatch");
}

// 验证错误路径与元信息。
void CheckErrorPaths(const mobilenet::service::Tokenizer& tokenizer) {
  mobilenet::service::EncodeOptions encode_options;
  std::vector<I64> ids;
  mobilenet::Status status =
      tokenizer.Encode(std::string("\xFF", 1), encode_options, &ids);
  Expect(!status.ok(), "invalid UTF-8 should fail");

  mobilenet::service::DecodeOptions skip_unknown;
  std::string text;
  status = tokenizer.Decode({200000}, skip_unknown, &text);
  Expect(status.ok() && text.empty(), "unknown id should be skipped");
  skip_unknown.skip_unknown_tokens = false;
  status = tokenizer.Decode({200000}, skip_unknown, &text);
  Expect(!status.ok(), "unknown id should fail when skip disabled");

  std::vector<mobilenet::service::ChatMessage> empty_messages;
  mobilenet::service::ChatTemplateOptions chat_options;
  status = tokenizer.ApplyChatTemplate(empty_messages, chat_options, &text);
  Expect(!status.ok(), "empty chat messages should fail");

  std::vector<mobilenet::service::ChatMessage> tool_messages = {
      {"tool", "x"}};
  status = tokenizer.ApplyChatTemplate(tool_messages, chat_options, &text);
  Expect(!status.ok(), "tool role should fail");

  Expect(tokenizer.TokenToId("<|im_start|>") == 151644,
         "TokenToId(<|im_start|>) mismatch");
  Expect(tokenizer.IdToToken(151644) == "<|im_start|>",
         "IdToToken(151644) mismatch");
  // 单字节非法 token：严格 Decode 报错，流式解码替换为 U+FFFD。
  mobilenet::service::DecodeOptions strict_options;
  strict_options.skip_unknown_tokens = false;
  status = tokenizer.Decode({94}, strict_options, &text);
  Expect(!status.ok(), "invalid byte token should fail strict Decode");
  std::unique_ptr<mobilenet::service::StreamDecoder> stream =
      tokenizer.CreateStreamDecoder(mobilenet::service::DecodeOptions{});
  Expect(stream != nullptr, "stream decoder is null");
  if (stream != nullptr) {
    std::string delta;
    status = stream->Push(94, &delta);
    Expect(status.ok(), "stream invalid byte should not fail");
    Expect(delta == "\xEF\xBF\xBD",
           "stream invalid byte should be U+FFFD");
  }

  Expect(tokenizer.bos_token_id() == 151643, "bos id mismatch");
  Expect(tokenizer.eos_token_ids() == std::vector<I64>({151645, 151643}),
         "eos ids mismatch");
}

// 并发编码/解码，验证只读线程安全。
void CheckConcurrency(const mobilenet::service::Tokenizer& tokenizer) {
  const std::string text = "你好Hello123 \U0001f600";
  mobilenet::service::EncodeOptions encode_options;
  std::vector<I64> baseline;
  mobilenet::Status status = tokenizer.Encode(text, encode_options, &baseline);
  Expect(status.ok(), "concurrency baseline encode failed");

  std::atomic<I32> failures{0};
  std::vector<std::thread> threads;
  for (I32 thread_index = 0; thread_index < 4; ++thread_index) {
    threads.emplace_back([&tokenizer, &text, &baseline, &failures]() {
      for (I32 iteration = 0; iteration < 50; ++iteration) {
        mobilenet::service::EncodeOptions options;
        std::vector<I64> ids;
        mobilenet::Status encode_status =
            tokenizer.Encode(text, options, &ids);
        if (!encode_status.ok() || ids != baseline) {
          ++failures;
          continue;
        }
        mobilenet::service::DecodeOptions decode_options;
        std::string decoded;
        mobilenet::Status decode_status =
            tokenizer.Decode(ids, decode_options, &decoded);
        if (!decode_status.ok() || decoded != text) {
          ++failures;
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  Expect(failures.load() == 0, "concurrent encode/decode mismatch");
}

// 验证 Runtime JSON 解析器的大对象/重复键行为。
//
// 背景：tokenizer.json 的 model.vocab 有 151k 个成员，原解析器用线性
// 扫描检测重复键，加载耗时 40+ 秒；改为哈希集合后应在数百毫秒内完成。
void CheckJsonParserRegression() {
  mobilenet::runtime::JsonValue value;
  mobilenet::Status status =
      mobilenet::runtime::ParseJson("{\"a\":1,\"a\":2}", &value);
  Expect(!status.ok(), "duplicate JSON keys should fail");

  std::string large = "{";
  for (I32 index = 0; index < 20000; ++index) {
    if (index > 0) {
      large.push_back(',');
    }
    large.append("\"k");
    large.append(std::to_string(index));
    large.append("\":");
    large.append(std::to_string(index));
  }
  large.push_back('}');
  const auto start = std::chrono::steady_clock::now();
  status = mobilenet::runtime::ParseJson(large, &value);
  const auto end = std::chrono::steady_clock::now();
  Expect(status.ok(), "large JSON parse failed");
  Expect(value.IsObject() && value.AsObject().size() == 20000,
         "large JSON member count mismatch");
  const FP64 elapsed_ms =
      std::chrono::duration<FP64, std::milli>(end - start).count();
  std::printf("PERF: json_large_keys=20000 parse=%.2f ms\n", elapsed_ms);

  std::string deep;
  for (I32 index = 0; index < 300; ++index) {
    deep.push_back('[');
  }
  for (I32 index = 0; index < 300; ++index) {
    deep.push_back(']');
  }
  status = mobilenet::runtime::ParseJson(deep, &value);
  Expect(!status.ok(), "deep JSON should fail");

  status = mobilenet::runtime::ParseJson("1 2", &value);
  Expect(!status.ok(), "trailing JSON data should fail");
}

// 记录编码性能基线。
void ReportPerformance(const mobilenet::service::Tokenizer& tokenizer,
                       FP64 load_ms) {
  std::string text;
  while (text.size() < 1024) {
    text += "The quick brown fox jumps over the lazy dog. ";
  }
  mobilenet::service::EncodeOptions options;
  std::vector<I64> ids;
  mobilenet::Status status = tokenizer.Encode(text, options, &ids);
  Expect(status.ok(), "performance encode failed");

  constexpr I32 kIterations = 50;
  const auto start = std::chrono::steady_clock::now();
  for (I32 iteration = 0; iteration < kIterations; ++iteration) {
    status = tokenizer.Encode(text, options, &ids);
    if (!status.ok()) {
      Expect(false, "performance encode iteration failed");
      return;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const FP64 elapsed_ms =
      std::chrono::duration<FP64, std::milli>(end - start).count();
  std::printf("PERF: load=%.2f ms, encode_1KB_avg=%.3f ms, tokens=%lld\n",
              load_ms, elapsed_ms / kIterations,
              static_cast<long long>(ids.size()));
}

}  // namespace

// 测试入口。
//
// 参数：
//   argv[1]：Qwen3 模型目录；
//   argv[2]：golden.json 路径。
// 返回：
//   成功 0；失败 1；资产缺失 77。
I32 main(I32 argc, C8** argv) {
  if (argc < 3) {
    std::printf("usage: tokenizer_test <model_dir> <golden.json>\n");
    return 1;
  }
  const std::string model_dir = argv[1];
  const std::string golden_path = argv[2];

  const auto load_start = std::chrono::steady_clock::now();
  mobilenet::service::TokenizerConfig config;
  config.model_dir = model_dir;
  std::unique_ptr<mobilenet::service::Tokenizer> tokenizer;
  mobilenet::Status status =
      mobilenet::service::Tokenizer::Load(config, &tokenizer);
  const auto load_end = std::chrono::steady_clock::now();
  if (!status.ok()) {
    std::printf("SKIP: cannot load tokenizer: %s\n", status.message().c_str());
    return 77;
  }
  const FP64 load_ms =
      std::chrono::duration<FP64, std::milli>(load_end - load_start).count();

  const std::string golden_text = ReadTextFile(golden_path);
  if (golden_text.empty()) {
    std::printf("SKIP: cannot read golden data: %s\n", golden_path.c_str());
    return 77;
  }
  mobilenet::runtime::JsonValue golden;
  status = mobilenet::runtime::ParseJson(golden_text, &golden);
  Expect(status.ok(), "golden JSON parse failed: " + status.message());

  std::printf("INFO: vocab=%lld eos=%zu chat=%d\n",
              static_cast<long long>(tokenizer->vocab_size()),
              tokenizer->eos_token_ids().size(),
              tokenizer->supports_chat_template() ? 1 : 0);

  // 校验金标准与本地 tokenizer.json 的来源一致。
  const mobilenet::runtime::JsonValue* meta = golden.Find("meta");
  const mobilenet::runtime::JsonValue* golden_sha =
      meta != nullptr ? meta->Find("tokenizer_json_sha256") : nullptr;
  if (golden_sha != nullptr && golden_sha->IsString()) {
    std::string actual_sha;
    const mobilenet::Status hash_status =
        mobilenet::runtime::HashFileSha256(
            model_dir + "/tokenizer.json", 0, &actual_sha);
    Expect(hash_status.ok(),
           "tokenizer.json hash failed: " + hash_status.message());
    Expect(actual_sha == golden_sha->AsString(),
           "tokenizer.json hash does not match golden data");
  } else {
    Expect(false, "golden meta tokenizer_json_sha256 missing");
  }

  const mobilenet::runtime::JsonValue* encode_cases =
      golden.Find("encode_cases");
  const mobilenet::runtime::JsonValue* chat_cases =
      golden.Find("chat_cases");
  Expect(encode_cases != nullptr && encode_cases->IsArray(),
         "golden encode_cases missing");
  Expect(chat_cases != nullptr && chat_cases->IsArray(),
         "golden chat_cases missing");
  if (encode_cases != nullptr && encode_cases->IsArray()) {
    for (const mobilenet::runtime::JsonValue& item :
         encode_cases->AsArray()) {
      CheckEncodeCase(*tokenizer, item);
    }
  }
  if (chat_cases != nullptr && chat_cases->IsArray()) {
    for (const mobilenet::runtime::JsonValue& item : chat_cases->AsArray()) {
      CheckChatCase(*tokenizer, item);
    }
  }

  CheckErrorPaths(*tokenizer);
  CheckConcurrency(*tokenizer);
  CheckJsonParserRegression();
  ReportPerformance(*tokenizer, load_ms);

  if (g_failures != 0) {
    std::printf("%d tokenizer checks failed\n", g_failures);
    return 1;
  }
  std::printf("all tokenizer checks passed\n");
  return 0;
}
