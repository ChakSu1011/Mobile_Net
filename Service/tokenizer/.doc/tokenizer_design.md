# Tokenizer 模块设计文档

> 模块目录：`Service/tokenizer/`
> 文档版本：v1.0（2026-11-21）
> 对应架构文档：`MobileNet_design.md` §4.5.1、FR-2.1、TC-TOK-01
> 首个验证模型：Qwen3-0.6B

## 1. 背景与目标

### 1.1 背景

Runtime Engine 以 token 为输入与输出：`Request.prompt_token_ids` 进入
Scheduler，`Response.sequences[].current_token.token_id` 逐步返回生成的
token。阶段二需要文本入口，因此必须在 Engine 之外提供 CPU Tokenizer 服务，
把用户文本转换为 token ids，并把生成 token 转换回文本。

项目已有的 Qwen3-0.6B 模型目录包含 HuggingFace `tokenizers` 格式的
`tokenizer.json`、`tokenizer_config.json` 与 `generation_config.json`。
Qwen3 使用 byte-level BPE、NFC 归一化与 GPT-2 风格的预分词正则；
Tokenizer 有效词表为 151,669（基础词表 151,643 + 26 个 added tokens），
模型 embedding 的 vocab_size 为 151,936（含保留 id）。不能使用简单的
空格切分或 SentencePiece 实现。

### 1.2 目标

1. 在 `Service/tokenizer/` 提供 C++17、CPU、进程内的 Tokenizer 库。
2. 加载模型目录下的 HF Tokenizer 资产，完成文本与 token ids 的双向转换。
3. 提供流式 Detokenize，保证 UTF-8 多字节字符不被截断。
4. 提供 Qwen3 Chat 模板格式化，供 Session 模块构造对话 prompt。
5. 与 HuggingFace `tokenizers` 在固定金标准语料上 token ids 完全一致。
6. 满足项目 memory.md 的编码规范、中文注释、四件套与六段式文档要求。

### 1.3 非目标

- 不支持 SentencePiece、WordPiece、Tiktoken 词表。
- 不实现通用 Jinja 模板引擎；仅支持 Qwen3 Chat 模板。
- 不支持训练、词表修改、Tokenizer 模型导出。
- 不支持多模态输入与多模态占位 token。
- 不提供跨进程 RPC；阶段二以进程内库形式使用。
- 不提供分词缓存持久化；加载性能优化留到后续版本。

### 1.4 与整体架构的关系

```text
Session ── Encode/Decode/ApplyChatTemplate ──▶ Tokenizer
   │                                              │
   └──────────── Request/Response ──────────────▶ Engine
```

Tokenizer 不依赖 Engine，可独立测试；Session 负责在启动时同时加载
Tokenizer 与 Engine，并把同一个模型目录下的资产对应起来。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求描述 | 验收要点 | 关联用例 |
| --- | --- | --- | --- |
| TOK-F-01 | 从模型目录加载 `tokenizer.json`、`tokenizer_config.json`、`generation_config.json`，校验 BPE、NFC、ByteLevel、Qwen3 预分词正则与 Chat 模板 | 正常加载；缺失/损坏/不支持类型返回明确错误 | TC-TOK-01、TC-TOK-06 |
| TOK-F-02 | `Encode(text)`：UTF-8 校验 → added-token 扫描 → NFC → GPT-2 语义预分词 → ByteLevel → BPE → ids | 与 HF `tokenizers` 金标准 ids 完全一致 | TC-TOK-02 |
| TOK-F-03 | `Decode(ids)`：added token 处理、ByteLevel 反变换、UTF-8 拼接；`skip_special_tokens` 可配置 | 无特殊 token 时往返等于 NFC 后文本；特殊 token 跳过行为与 HF 一致 | TC-TOK-03 |
| TOK-F-04 | `StreamDecoder`：逐 token 增量解码，缓冲不完整 UTF-8 字节 | 任意 token 边界切分后拼接与整段 Decode 逐字节一致 | TC-TOK-04 |
| TOK-F-05 | `ApplyChatTemplate`：支持 system/user/assistant 消息、`add_generation_prompt`、`enable_thinking` 三态 | 与 HF `apply_chat_template` 输出逐字节一致 | TC-TOK-05 |
| TOK-F-06 | 特殊 token 查询：EOS/BOS id、token 文本与 id 互查、词表大小 | 与配置一致；未知 token 返回 -1 | TC-TOK-01、TC-TOK-06 |
| TOK-F-07 | 错误处理：非法 UTF-8、未知 id、非法配置、不支持的模板/模型类型 | 不崩溃，返回 `Status` 与可定位信息 | TC-TOK-06 |
| TOK-F-08 | 线程安全：加载完成后只读；多个线程可并发 Encode/Decode | 并发压力测试无数据竞争、结果一致 | TC-TOK-07 |

说明：基础版不支持 added token 的 `single_word/lstrip/rstrip/normalized`
标志；Qwen3 的 26 个 added tokens 均未启用这些标志。加载时若遇到启用标志，
返回 `kUnsupportedOp`，避免静默产生与 HF 不一致的结果。

### 2.2 非功能需求

| 编号 | 类别 | 要求 | 验证方式 |
| --- | --- | --- | --- |
| TOK-NF-01 | 正确性 | 金标准语料上 Encode ids 100% 一致；无特殊 token 文本 Decode ∘ Encode 等于 NFC 文本 | TC-TOK-02、TC-TOK-03 |
| TOK-NF-02 | 流式正确性 | 流式输出不出现半个 UTF-8 字符；与整段解码逐字节一致 | TC-TOK-04 |
| TOK-NF-03 | 线程安全 | 加载后不可变；并发 Encode/Decode 无数据竞争 | TC-TOK-07 |
| TOK-NF-04 | 可移植性 | C++17；Windows 主机与 Android arm64 NDK 可编译；无网络访问 | 构建检查 |
| TOK-NF-05 | 性能 | 加载与编码耗时先记录基线；初始目标为与 HF `tokenizers` 同量级（1 KB 文本编码耗时不高于其 2 倍），实测后锁定 | TC-TOK-08 |
| TOK-NF-06 | 内存 | 常驻内存记录基线；峰值内存（加载期）记录并评估手机端可行性 | TC-TOK-08 |
| TOK-NF-07 | 工程规范 | Google C++ Style；模块/类/函数级中文注释覆盖功能边界、输入输出、主要逻辑 | 代码审查 |

### 2.3 约束条件

- 运行时不得访问网络；模型与 Tokenizer 资产全部本地读取。
- 词表与 merges 使用模型自带 `tokenizer.json`，不做二次转换。
- Unicode 归一化与类别查询使用内置 `utf8proc` v2.10.0（MIT），
  第三方代码按项目规则附带许可证与来源说明。
- 复用 `Runtime/common/json.h` 解析 JSON；本模块不引入 JSON 序列化。
- 依赖修复：原 JSON 解析器对对象重复键做 O(n²) 线性扫描，150k 词表加载
  耗时约 44 s；已改为哈希集合检测（语义不变），加载降至亚秒级。该修改
  位于 `Runtime/common/json.cc`，Runtime 现有快速测试全部通过。
- 所有 Host 端整数类型使用 `Utils/cpp/types.h` 别名。

### 2.4 验收标准

- AC-TOK-1：Qwen3-0.6B 金标准语料全部通过 Encode/Decode 对照。
- AC-TOK-2：流式解码在所有 token 切分点与整段解码一致。
- AC-TOK-3：Qwen3 Chat 模板在 system/user/assistant、多轮、
  `enable_thinking` 三态下与 HF 输出逐字节一致。
- AC-TOK-4：错误路径返回稳定错误码，不崩溃。
- AC-TOK-5：四件套（设计文档、代码、测试结果、skill）齐全。

## 3. 概要设计

### 3.1 总体架构

```text
                    ┌─────────────────────────────┐
                    │          Tokenizer          │
                    │  Load / Encode / Decode     │
                    │  ApplyChatTemplate          │
                    └───────┬───────────┬─────────┘
                            │           │
         ┌──────────────────┘           └──────────────────┐
         ▼                                                 ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
│  Config Loader  │   │  Encode Pipeline │   │   Decode Pipeline   │
│ tokenizer.json  │   │ UTF-8 校验       │   │ added token 过滤    │
│ tokenizer_config│   │ added token 扫描 │   │ 词表查表            │
│ generation_conf │   │ NFC              │   │ ByteLevel 反变换    │
└─────────────────┘   │ 预分词 Split     │   │ UTF-8 拼接/缓冲     │
                      │ ByteLevel        │   └─────────────────────┘
                      │ BPE 合并         │
                      └─────────────────┘
```

### 3.2 模块划分

| 文件 | 职责 | 边界 |
| --- | --- | --- |
| `tokenizer.h/.cc` | 公共门面与生命周期；组合各子模块 | 不实现 BPE/Unicode 细节 |
| `tokenizer_config.h/.cc` | 读取并校验三个 JSON 资产 | 不做编码 |
| `bpe.h/.cc` | 词表、merges、BPE 合并、ids 映射 | 不做归一化/预分词 |
| `byte_level.h/.cc` | GPT-2 byte ↔ unicode 双向映射 | 不处理特殊 token |
| `pretokenizer.h/.cc` | Qwen3 GPT-2 正则语义扫描器 | 不做 NFC/ByteLevel |
| `unicode_util.h/.cc` | UTF-8 解码、NFC、Unicode 类别、White_Space | 不依赖 Tokenizer 类型 |
| `chat_template.h/.cc` | Qwen3 Chat 模板格式化 | 不做分词 |
| `stream_decoder.h/.cc` | 增量 Detokenize 与 UTF-8 缓冲 | 不持有 Engine/Session |
| `third_party/utf8proc/` | MIT Unicode 库 | 不修改第三方源码 |

### 3.3 关键流程

#### 3.3.1 Encode

```text
输入 text
  → UTF-8 合法性检查（strict_utf8=true 时非法即错误）
  → added token 最长匹配（在原始文本上；Qwen3 added tokens 均 normalized=false）
  → 每个普通片段做 NFC
  → Qwen3 Split 扫描器切成预分词片段
  → 每个片段 ByteLevel 映射为 Unicode 符号串
  → BPE 按 merge rank 合并
  → 符号查词表得到 ids
  → 按 post_processor 追加特殊 token（Qwen3 ByteLevel：无）
```

#### 3.3.2 Decode

```text
ids
  → 逐 id 查 token
  → added token：special 且 skip_special_tokens 时跳过；否则直接输出 content
  → 普通 token：每个 Unicode 码点经 ByteLevel 反查得到字节
  → 字节拼接后校验 UTF-8
  → 输出文本
```

#### 3.3.3 StreamingDecode

```text
每个 id 的字节追加到 pending_bytes
  → 从 pending_bytes 头部提取最长完整 UTF-8 前缀
  → 输出前缀（delta），保留不完整后缀
Finish 时若仍有不完整字节 → 返回错误
```

### 3.4 技术选型与取舍

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 实现语言 | C++17 | 与 Runtime/Engine 进程内直连，避免跨语言绑定 |
| Unicode | utf8proc v2.10.0（MIT） | NFC 与 Unicode 类别完整、经过验证，显著降低自研表风险 |
| JSON | `Runtime/common/json.h` | 项目已有严格解析器；避免重复实现 |
| 预分词 | 手写扫描器 | C++ `std::regex` 不支持 `\p{L}` 等 Unicode 属性；引入正则库过重 |
| added tokens | 字节 trie | Qwen3 有 26 个 added tokens，需最长匹配；trie 线性扫描 |
| 模板 | Qwen3 原生格式化 | 不引入 Jinja 引擎；与 Qwen3 模板逐字节对齐 |
| 流式 | 字节缓冲 + UTF-8 前缀提取 | 不依赖替换字符，保证不截断字符 |

## 4. 详细设计

### 4.1 公共接口

```cpp
namespace mobilenet {
namespace service {

struct ChatMessage {
  std::string role;
  std::string content;
};

enum class ThinkingMode { kDefault = 0, kEnabled = 1, kDisabled = 2 };

struct TokenizerConfig {
  std::string model_dir;              // 含三个 JSON 的目录
  std::string tokenizer_json;         // 非空时覆盖 model_dir/tokenizer.json
  std::string tokenizer_config_json;  // 非空时覆盖
  std::string generation_config_json; // 非空时覆盖
  Bool strict_utf8 = true;
  Bool enable_nfc = true;
};

struct EncodeOptions {
  Bool add_special_tokens = true;
};

struct DecodeOptions {
  Bool skip_special_tokens = true;
  Bool skip_unknown_tokens = true;  // 与 HF decode 一致：未知 id 输出空串
};

struct ChatTemplateOptions {
  Bool add_generation_prompt = true;
  ThinkingMode thinking = ThinkingMode::kDefault;
};

class StreamDecoder {
 public:
  ~StreamDecoder();
  Status Push(I64 token_id, std::string* delta);
  Status Finish(std::string* tail);
  void Reset();
 private:
  friend class Tokenizer;
  StreamDecoder(std::shared_ptr<const Tokenizer::Impl> impl,
                DecodeOptions options);
  std::shared_ptr<const Tokenizer::Impl> impl_;
  DecodeOptions options_;
  std::string pending_bytes_;
};

class Tokenizer {
 public:
  static Status Load(const TokenizerConfig& config,
                     std::unique_ptr<Tokenizer>* tokenizer);
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  Status Encode(const std::string& text, const EncodeOptions& options,
                std::vector<I64>* ids) const;
  Status Decode(const std::vector<I64>& ids, const DecodeOptions& options,
                std::string* text) const;
  Status ApplyChatTemplate(const std::vector<ChatMessage>& messages,
                           const ChatTemplateOptions& options,
                           std::string* prompt) const;
  std::unique_ptr<StreamDecoder> CreateStreamDecoder(
      const DecodeOptions& options) const;

  I64 TokenToId(const std::string& token) const;
  std::string IdToToken(I64 id) const;
  I64 bos_token_id() const;
  const std::vector<I64>& eos_token_ids() const;
  I64 vocab_size() const;
  Bool supports_chat_template() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace service
}  // namespace mobilenet
```

接口约束：

- `Load` 成功返回后对象不可变；所有 const 方法线程安全。
- `StreamDecoder` 持有 `Impl` 的共享指针，可安全存活到 Tokenizer 析构之后；
  但不得跨线程并发使用同一个 `StreamDecoder`。
- `Push` 在 `skip_unknown_tokens=false` 时对未知 id 返回
  `kInvalidArgument`；默认跳过未知 id（与 HF decode 一致）。流式解码遇到
  非法 UTF-8 字节时以 U+FFFD 替换并继续（与 HF lossy 行为一致）；
  `Finish` 若仍有不完整尾部返回 `kInvalidArgument`，Session 层以 U+FFFD
  兜底。`kUnsupportedOp` 表示配置不支持。
- `ApplyChatTemplate` 只做文本格式化，不做分词；Session 随后调用 `Encode`。

### 4.2 配置与数据加载

#### 4.2.1 文件读取

- 默认路径：`model_dir/tokenizer.json`、
  `model_dir/tokenizer_config.json`、`model_dir/generation_config.json`。
- 使用二进制读取，限制单文件大小（`tokenizer.json` ≤ 64 MiB，
  其它 ≤ 4 MiB），防止误加载超大文件。
- 使用 `ParseJson`；解析失败返回 `kInvalidModelFile`，消息包含文件名与
  解析器的字节偏移。

#### 4.2.2 `tokenizer.json` 校验

| 字段 | 要求 |
| --- | --- |
| `model.type` | `"BPE"`，否则 `kUnsupportedOp` |
| `model.vocab` | 对象；token → id；id 在 `[0, vocab_size)` 且不重复 |
| `model.merges` | 数组；元素为 `[a, b]` 数组或 `"a b"` 字符串；按顺序记录 rank |
| `added_tokens` | 数组；`id/content/special` 必填；Qwen3 的标志必须为 false |
| `normalizer` | `null` 或 `{"type":"NFC"}`；否则 `kUnsupportedOp` |
| `pre_tokenizer` | Sequence 包含 Split（Qwen3 正则）与 ByteLevel；校验失败报错 |
| `decoder` | ByteLevel 或 null；否则 `kUnsupportedOp` |
| `post_processor` | ByteLevel 或 null；非 ByteLevel 的 post processor 不支持 |
| `truncation`/`padding` | 必须为 null 或不存在；基础版不实现 |

预分词正则校验：要求 Split 的 `pattern.Regex` 等于 Qwen3 的固定模式
（允许字节级差异但必须包含 `\p{L}`、`\p{N}`、`\s` 与 contraction 分支），
`behavior` 为 `Isolated`，`invert=false`。ByteLevel 的
`add_prefix_space=false`、`use_regex=false`。

#### 4.2.3 `tokenizer_config.json` 与 `generation_config.json`

- `tokenizer_config.json`：
  - `chat_template`：字符串；若包含 `<|im_start|>` 与 `<|im_end|>`，
    标记 `supports_chat_template=true`；否则 Chat 接口返回 `kUnsupportedOp`。
  - `add_bos_token`：Qwen3 为 false；true 时返回 `kUnsupportedOp`
    （基础版不实现 BOS 自动添加）。
  - `eos_token`、`pad_token`：记录内容与 id。
  - `split_special_tokens`：必须为 false；true 时返回 `kUnsupportedOp`。
- `generation_config.json`：
  - `eos_token_id`：整数或整数数组；转换为 `eos_token_ids_`。
  - `bos_token_id`、`pad_token_id`：记录。
  - `vocab_size`：若存在，用于 `vocab_size_`；否则 `max_token_id + 1`。
- 优先级：`generation_config.eos_token_id` > `tokenizer_config.eos_token` >
  空。

#### 4.2.4 内存结构

| 结构 | 类型 | 说明 |
| --- | --- | --- |
| `id_to_token_` | `std::vector<std::string>` | 大小 `vocab_size`；空串表示无 token |
| `token_to_id_` | `std::unordered_map<std::string, I64>` | 基础词表 + added tokens |
| `merge_rank_` | `std::unordered_map<std::string, I32>` | key 为 `a + '\0' + b` |
| `added_tokens_` | `std::vector<AddedToken>` | id/content/special |
| `added_trie_` | 字节 trie | 最长匹配 |
| `byte_to_char_` | `std::array<I32,256>` | byte → Unicode 码点 |
| `char_to_byte_` | `std::unordered_map<I32,U8>` | Unicode 码点 → byte |
| `eos_token_ids_` | `std::vector<I64>` | 来自 generation config |
| `chat_template_supported_` | `Bool` | 是否支持 Qwen3 Chat |

加载完成后释放 `JsonValue` 树，仅保留上述结构。

### 4.3 Unicode 与 UTF-8

#### 4.3.1 UTF-8 校验

- `unicode_util` 提供 `Status DecodeUtf8(const std::string&, std::vector<Codepoint>*)`，
  `Codepoint` 包含 `I32 value`、`Usize byte_offset`、`Usize byte_length`。
- 非法 lead/continuation、过长编码、代理区码点、超过 U+10FFFF 均返回
  `kInvalidArgument`。
- `strict_utf8=false` 时，非法字节替换为 U+FFFD；基础版默认 true。

#### 4.3.2 NFC

- `NormalizeNfc(text, out)` 使用
  `utf8proc_map(..., UTF8PROC_STABLE | UTF8PROC_COMPOSE)`。
- 失败返回 `kInvalidArgument`；调用方负责释放 utf8proc 分配的内存。
- `enable_nfc=false` 仅用于调试，与 HF 结果可能不一致，文档标注。

#### 4.3.3 类别与空白

- `IsLetter`：utf8proc category 为 LU/LL/LT/LM/LO。
- `IsNumber`：ND/NL/NO。
- `IsWhitespace`：严格按 Unicode White_Space 属性：U+0009–U+000D、
  U+0020、U+0085、U+00A0、U+1680、U+2000–U+200A、U+2028、U+2029、
  U+202F、U+205F、U+3000。
- `IsNewline`：`\r`（U+000D）或 `\n`（U+000A）。

### 4.4 Qwen3 预分词扫描器

#### 4.4.1 正则模式

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)
| [^\r\n\p{L}\p{N}]?\p{L}+
| \p{N}
|  ?[^\s\p{L}\p{N}]+[\r\n]*
| \s*[\r\n]+
| \s+(?!\S)
| \s+
```

扫描器按“左端最先、备选顺序优先”的 regex 语义在每个位置依次尝试：

1. 大小写不敏感的 contraction：`'s`、`'t`、`'re`、`'ve`、`'m`、`'ll`、`'d`。
2. 可选一个“非 CR/LF、非字母、非数字”字符，后接 1 个以上字母。
3. 单个数字字符。
4. 可选一个 ASCII 空格，后接 1 个以上“非空白、非字母、非数字”字符，
   再后接任意 CR/LF。
5. 任意空白后接 1 个以上 CR/LF；匹配到最后一个换行组结束。
6. 1 个以上空白且其后不是非空白：贪婪匹配整个空白串；若空白串后仍有
   非空白，则回退 1 个空白字符（`\s+(?!\S)` 的贪婪回溯语义）。
7. 1 个以上空白。

#### 4.4.2 关键边界

| 输入 | 切分结果 | 说明 |
| --- | --- | --- |
| `a   b` | `a`、`  `、` b` | 空格串回溯，最后一个空格与 b 合并 |
| `  !` | ` `、` !` | 首个空格单独；第二段为空格+标点 |
| `'s` | `'s` | contraction |
| ` 's` | ` '`、`s` | 左端最先：空格+标点分支先命中 |
| `123` | `1`、`2`、`3` | `\p{N}` 单字符 |
| `你好，世界！` | `你好`、`，`、`世界`、`！` | CJK 字母与全角标点 |
| `emoji 😀🚀` | `emoji`、` Ġ...`、`...` | Emoji 属标点分支 |
| `a\n\nb` | `a`、`\n\n`、`b` | 换行分支 |

若扫描器未命中任何备选（理论上不应发生），返回 `kInternal` 并指明字节偏移。

### 4.5 ByteLevel 映射

采用 GPT-2 `bytes_to_unicode`：

```text
可见字节：'!'..'~'、'¡'..'¬'、'®'..'ÿ' 映射为自身码点；
其余字节按 256 + 递增序号映射到 U+0100 起的码点。
```

- Encode：UTF-8 字节 → 映射码点 → UTF-8 字符串（作为 BPE 符号）。
- Decode：符号中的每个码点 → 原始字节；码点不在反向表时返回
  `kInvalidModelFile`（词表损坏）。
- ByteLevel 映射与 UTF-8 编码是两个独立步骤，注意不要混淆。

### 4.6 BPE 合并

- 初始符号为 ByteLevel 映射后的每个 Unicode 字符（单个码点的 UTF-8 串）。
- `merge_rank_` key 为 `first + '\0' + second`；ByteLevel 符号不含 NUL，
  分隔安全。
- 每轮扫描相邻符号对，选择 rank 最小的一对；若不存在则终止。
- 将该对在符号序列中的所有相邻出现合并（左到右），重复直到无可用 merge。
- 最终符号逐个查 `token_to_id_`；缺失返回 `kInternal` 并输出符号与预分词
  片段，便于定位词表/merges 不一致。
- 复杂度：预分词片段通常很短；基础版使用 O(n²) 扫描。实测若成为瓶颈，
  后续引入优先队列或缓存。

### 4.7 Added Token 与特殊 Token

- 加载时为所有 added tokens 建立字节 trie；`added_tokens` 的内容均为
  UTF-8。Qwen3 的内容为 ASCII。
- 扫描在原始文本上进行（NFC 之前）；普通片段再逐段 NFC。
- 在每个字节位置从 trie 根向下走，记录最深的终止节点；命中后输出 token id，
  跳过对应字节；未命中则前进一个完整 UTF-8 码点并累积普通文本。
- 最长匹配优先；相同长度下按 id 升序（Qwen3 无重叠 token）。
- `special` 标志用于 Decode 的 `skip_special_tokens`：
  - `special=true` 且跳过：不输出；
  - `special=true` 且不跳过：输出 content；
  - `special=false`：始终输出 content。
- 加载时若任一 added token 启用
  `single_word/lstrip/rstrip/normalized`，返回 `kUnsupportedOp`。

### 4.8 Chat 模板

实现 Qwen3 `chat_template` 的以下子集：

1. 消息角色：`system`、`user`、`assistant`；其他角色（含 `tool`）
   返回 `kUnsupportedOp`。
2. 首条 `system`：`<|im_start|>system\n{content}<|im_end|>\n`。
3. 非首条 `system`：按模板使用 `<|im_start|>system` 角色名（与 user
   分支相同，但角色名保持 `system`）。
4. `user`：`<|im_start|>user\n{content}<|im_end|>\n`。
5. `assistant`：
   - 若 content 含 `</think>`，按模板拆分 reasoning/content；
   - 若该 assistant 位于最后一条 user 之后（`index > last_query_index`），
     按模板的 thinking 分支输出；
   - 否则输出 `<|im_start|>assistant\n{content}<|im_end|>\n`；
   - 基础版不支持 `tool_calls`；若消息结构中出现 `tool_calls` 字段，
     返回 `kUnsupportedOp`。
6. `add_generation_prompt=true`：
   - `thinking=kDefault/kEnabled`：`<|im_start|>assistant\n`
   - `thinking=kDisabled`：`<|im_start|>assistant\n<think>\n\n</think>\n\n`
7. `add_generation_prompt=false`：只输出历史消息。
8. 空消息数组返回 `kInvalidArgument`。
9. 模板不支持时返回 `kUnsupportedOp`。

参考输出（Qwen3-0.6B，HF `apply_chat_template`）：

```text
system+user, add_generation_prompt=True, enable_thinking=None:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

### 4.9 流式解码

- `StreamDecoder` 内部维护 `pending_bytes_`。
- `Push`：
  1. 将 token 对应字节（普通 token 的 ByteLevel 反变换，或 added token 的
     content）追加到 `pending_bytes_`。
  2. 从头部提取最长完整 UTF-8 前缀：
     - 逐字节判断 lead byte 与期望长度；
     - 若剩余字节不足一个完整序列，停止（保留）；
     - 若 continuation/lead 非法，输出 U+FFFD 并前进一个字节；
     - 不完整序列保留到下一次 `Push`。
  3. 将完整前缀写入 `delta`（可为空），从 `pending_bytes_` 删除。
- `Finish`：若 `pending_bytes_` 非空，返回 `kInvalidArgument`
  （消息为“流结束时仍有不完整 UTF-8 字节”）；否则返回空串。Session 在
  终态捕获该错误并追加 U+FFFD，避免整条响应失败。
- `Reset` 清空缓冲。
- 跳过特殊 token 时不产生字节；`spaces_between_special_tokens` 不在基础版
  提供。

### 4.10 错误处理

| 场景 | 错误码 | 附加信息 |
| --- | --- | --- |
| 资产文件缺失/不可读 | `kIoError` | 路径 |
| JSON 解析失败 | `kInvalidModelFile` | 文件名 + 字节偏移 |
| 非 BPE/非 NFC/非 ByteLevel/未知正则 | `kUnsupportedOp` | 字段名与实际值 |
| 词表 id 越界/重复 | `kInvalidModelFile` | token/id |
| added token 标志不支持 | `kUnsupportedOp` | token 内容 |
| 非法 UTF-8 输入 | `kInvalidArgument` | 字节偏移 |
| 未知 token id（skip_unknown_tokens=false） | `kInvalidArgument` | id |
| 未知 token id（默认） | 无 | 输出空串，与 HF decode 一致 |
| 词表符号缺失（内部不一致） | `kInternal` | 预分词片段与符号 |
| 流结束仍不完整 | `kInvalidArgument` | 残留字节数 |
| Chat 模板不支持/角色非法 | `kUnsupportedOp` / `kInvalidArgument` | 角色名 |

所有 `Status` 消息使用 `模块::阶段` 前缀，例如
`Tokenizer::Encode: ...`。

### 4.11 性能与内存

- 加载期：
  - JSON 树峰值内存与 `tokenizer.json` 大小相关；加载完成后释放树。
  - 词表 map 预留容量，避免多次 rehash。
- 编码期：
  - 预分词与 BPE 使用栈/局部 vector；避免每个符号单独堆分配。
  - 基础版不做全局缓存；多线程并发安全。
- 解码期：
  - 批量 Decode 一次拼接；流式 Decode 只保留不完整字节。
- 基线指标在 TC-TOK-08 记录：加载耗时、1 KB 文本 Encode 耗时、常驻内存、
  峰值内存。初始目标仅作参考，实测后写入文档并锁定。

## 5. 实施步骤

| 阶段 | 任务 | 交付 | 依赖 | 风险 |
| --- | --- | --- | --- | --- |
| T1 | 设计文档确认与落盘 | 本文档 | 用户确认 | 无 |
| T2 | CMake 骨架、utf8proc 内置、配置读取 | `CMakeLists.txt`、`tokenizer_config.*` | T1 | 构建工具链差异 |
| T3 | Unicode/UTF-8/ByteLevel/预分词 | `unicode_util.*`、`byte_level.*`、`pretokenizer.*` | T2 | 正则语义偏差 |
| T4 | BPE 与 added token | `bpe.*` | T3 | 词表/merges 一致性 |
| T5 | Decode 与 StreamDecoder | `stream_decoder.*`、`tokenizer.cc` | T4 | UTF-8 边界 |
| T6 | Qwen3 Chat 模板 | `chat_template.*` | T5 | 模板分支差异 |
| T7 | 金标准数据与单元测试 | `tests/gen_golden.py`、`tests/data/golden.json`、`tokenizer_test.cc` | T4–T6 | HF 版本差异 |
| T8 | 性能/内存基线、skill、测试结果 | `.skill/SKILL.md`、`tests/test_results.md` | T7 | 手机端内存 |

## 6. 测试与验证设计

### 6.1 测试策略

- 以 HuggingFace `tokenizers` 为金标准，固定版本与资产哈希。
- 单元测试覆盖正常路径、边界条件、错误路径与流式切分。
- 并发测试验证线程安全。
- 性能测试只记录基线，不与未校准的目标做硬性通过判定。

### 6.2 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-21 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | zig clang wrapper（`Graph/tools/zig_cxx.cmd`） |
| C++ 标准 | C++17 |
| Python | 3.13.3 |
| 金标准库 | `tokenizers 0.22.2`、`transformers 5.9.0` |
| 模型资产 | `Models/Qwen3/tokenizer.json`（11.4 MB） |
| 实测 | 加载亚秒级；1 KB 文本 Encode 亚毫秒级；82 条 Encode 金标准全部通过 |

### 6.3 用例设计

| 编号 | 用例 | 输入 | 期望 |
| --- | --- | --- | --- |
| TC-TOK-01 | 加载与元信息 | Qwen3 模型目录 | `kOk`；Tokenizer vocab=151669；EOS=[151645,151643]；支持 Chat |
| TC-TOK-02 | Encode 金标准 | ASCII/CJK/Emoji/代码/空白/NFC/NFD/特殊 token 混排 | ids 与 HF 完全一致 |
| TC-TOK-03 | Decode 金标准 | 上述 ids | 与 HF `decode(skip_special_tokens=true)` 一致 |
| TC-TOK-04 | 流式切分 | 每个用例在所有 token 边界切分 | 拼接结果与整段解码逐字节一致 |
| TC-TOK-05 | Chat 模板 | 单轮/多轮/系统消息/`enable_thinking` 三态 | 与 HF `apply_chat_template` 逐字节一致 |
| TC-TOK-06 | 错误路径 | 缺失文件、损坏 JSON、未知 id、非法 UTF-8、不支持配置 | 返回预期错误码，不崩溃 |
| TC-TOK-07 | 并发 | 4 线程 × 100 次相同/不同文本 Encode/Decode | 结果一致，无崩溃 |
| TC-TOK-08 | 性能/内存 | 1 KB 文本 100 次编码；加载一次 | 记录实测值 |

### 6.4 通过标准

- TC-TOK-01 至 TC-TOK-07 全部通过；TC-TOK-08 记录数据并给出结论。
- 金标准不一致时不得放宽断言；必须修复实现或记录明确的、
  用户确认的例外。

### 6.5 测试命令

```bash
cd d:/Mobile_Net
cmake -S Service -B Service/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=d:/Mobile_Net/Graph/tools/zig_cxx.cmd
cmake --build Service/build
ctest --test-dir Service/build --output-on-failure
```

实际输出与结论写入 `Service/tokenizer/tests/test_results.md`。

### 6.6 未覆盖项与风险

- Android 真机 / NDK 构建若环境不可用，标注未覆盖。
- 非 Qwen3 的 HF Tokenizer 变体不在基础版范围。
- `tokenizer.json` 加载峰值内存尚未在手机端实测。
- Unicode 版本差异（utf8proc v2.10.0 与 HF tokenizers 内置版本）可能影响
  极少数字符；金标准语料需记录覆盖范围。

## 附录 A：参考资料

| 资料 | 用途 |
| --- | --- |
| `Models/Qwen3/tokenizer.json` | 词表、merges、added tokens、归一化与预分词配置 |
| `Models/Qwen3/tokenizer_config.json` | Chat 模板、EOS/BOS/PAD 配置 |
| `Models/Qwen3/generation_config.json` | EOS ids |
| HF `tokenizers` 文档 | ByteLevel、BPE、pre-tokenizer 语义 |
| utf8proc v2.10.0（MIT） | NFC 与 Unicode 类别 |
| SGLang `detokenizer_manager.py` | 增量解码与 UTF-8 边界处理参考 |
| vLLM `tokenizer` 相关实现 | 服务侧特殊 token 处理参考 |

## 附录 B：需求追溯

| 需求 | 设计章节 | 测试用例 |
| --- | --- | --- |
| TOK-F-01 | §4.2 | TC-TOK-01、TC-TOK-06 |
| TOK-F-02 | §4.3–§4.6 | TC-TOK-02 |
| TOK-F-03 | §4.7 | TC-TOK-03 |
| TOK-F-04 | §4.9 | TC-TOK-04 |
| TOK-F-05 | §4.8 | TC-TOK-05 |
| TOK-F-06 | §4.2、§4.7 | TC-TOK-01、TC-TOK-06 |
| TOK-F-07 | §4.10 | TC-TOK-06 |
| TOK-F-08 | §4.11 | TC-TOK-07 |
| TOK-NF-05/06 | §4.11 | TC-TOK-08 |
