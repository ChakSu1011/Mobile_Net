# Tokenizer 模块 Skill

## 适用场景

- 为 Runtime Engine 提供文本 ↔ token ids 转换。
- 接入 Qwen3-0.6B 或同族 HF `tokenizer.json`（BPE + ByteLevel + NFC）。
- 需要进程内、CPU、线程安全、无网络依赖的 Tokenizer。
- 需要流式 Detokenize 或 Qwen3 Chat 模板格式化。

不适用：SentencePiece/WordPiece/Tiktoken 词表、通用 Jinja 模板、训练或
词表修改、多模态 token。

## 输入

- 模型目录，至少包含：
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `generation_config.json`（可选；缺失时 EOS 从 tokenizer_config 读取）
- Qwen3 示例：`Models/Qwen3/`
- 文本为 UTF-8；`strict_utf8=true` 时非法字节直接报错。

## 步骤

1. 构建：
   ```bash
   cd d:/Mobile_Net
   cmake -S Service -B Service/build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_C_COMPILER=d:/Mobile_Net/Service/tools/zig_cc.cmd \
     -DCMAKE_CXX_COMPILER=d:/Mobile_Net/Graph/tools/zig_cxx.cmd
   cmake --build Service/build
   ```
2. 加载：
   ```cpp
   TokenizerConfig config;
   config.model_dir = "Models/Qwen3";
   std::unique_ptr<Tokenizer> tokenizer;
   Status status = Tokenizer::Load(config, &tokenizer);
   ```
3. 编码/解码：
   ```cpp
   EncodeOptions encode_options;
   std::vector<I64> ids;
   tokenizer->Encode("你好，世界！", encode_options, &ids);

   DecodeOptions decode_options;  // skip_special_tokens=true by default
   std::string text;
   tokenizer->Decode(ids, decode_options, &text);
   ```
4. 流式：
   ```cpp
   auto decoder = tokenizer->CreateStreamDecoder(decode_options);
   for (I64 id : ids) {
     std::string delta;
     decoder->Push(id, &delta);   // 只输出完整 UTF-8 前缀
   }
   std::string tail;
   decoder->Finish(&tail);
   ```
5. Chat 模板：
   ```cpp
   std::vector<ChatMessage> messages = {
       {"system", "You are helpful."},
       {"user", "Hello"},
   };
   ChatTemplateOptions options;
   options.thinking = ThinkingMode::kDisabled;
   std::string prompt;
   tokenizer->ApplyChatTemplate(messages, options, &prompt);
   ```
6. 测试：
   ```bash
   ./Service/build/tokenizer/tokenizer_test.exe \
     Models/Qwen3 Service/tokenizer/tests/data/golden.json
   ```
   重新生成金标准（开发机需 `tokenizers`/`transformers`）：
   ```bash
   python Service/tokenizer/tests/gen_golden.py \
     --model-dir Models/Qwen3 \
     --output Service/tokenizer/tests/data/golden.json
   ```

## 输出

- `Status` + `std::vector<I64>`（Encode）
- `Status` + `std::string`（Decode/流式 delta/Chat prompt）
- 词表元信息：`vocab_size()`、`eos_token_ids()`、`bos_token_id()`、
  `TokenToId/IdToToken`。
- Qwen3-0.6B：Tokenizer 有效词表 151,669；EOS `{151645, 151643}`；
  BOS `151643`。

## 常见陷阱

1. **added token 必须先于 NFC 匹配**：HF 的 Qwen3 added tokens 均
   `normalized=false`；先整体 NFC 再做 added token 匹配会在极少数字符上
   与 HF 不一致。
2. **NFC 不可省略**：`café`（NFC）与 `cafe\u0301`（NFD）编码结果不同，
   Qwen3 的 normalizer 是 NFC。
3. **预分词是正则语义，不是“最长匹配”**：`\s+(?!\S)` 需要贪婪回溯，
   `a   b` 会切成 `a`、`  `、` b`；`  !` 会切成 ` `、` !`。
4. **ByteLevel 是字节映射，不是 UTF-8 转义**：先按 GPT-2 byte↔unicode
   映射，再做 BPE；不要直接把 token 文本当普通字符串拼接。
5. **special 与 added 不是一回事**：`<|im_end|>` 是 `special=true`，
   `<tool_call>`/`<think>` 是 `special=false`；`skip_special_tokens`
   只跳过前者。
6. **未知 id 默认跳过**：与 HF `decode` 一致；若需要严格报错，设置
   `DecodeOptions.skip_unknown_tokens=false`。
7. **流式不能输出半个 UTF-8 字符**：保留不完整字节到下一 token；
   `Finish` 时仍有残留说明数据异常。
8. **JSON 解析性能**：`tokenizer.json` 有 151k 词表成员；Runtime JSON
   解析器的重复键检测已改为哈希集合，加载从约 44 s 降至亚秒级。若未来
   替换解析器，务必保留该行为。
9. **线程安全**：`Tokenizer` 加载后只读；`StreamDecoder` 每流独占，
   不要跨线程共享。
10. **词表大小口径**：Tokenizer 有效词表 151,669；模型 embedding
    `vocab_size=151,936`，其中 151,669–151,935 为保留/未定义 id。

## 验证基线

- 82 条 Encode 金标准（含 60 条确定性随机语料）全部通过。
- Decode/流式/keep-special 与 HF 一致。
- 8 条 Chat 模板用例与 HF `apply_chat_template` 逐字节一致。
- 4 线程并发 Encode/Decode 通过。
- 实测：加载约 0.2–0.7 s；1 KB 文本 Encode 约 0.2–0.5 ms。
