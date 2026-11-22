# Tokenizer 模块测试结果

## 1. 测试范围

本轮验证 `Service/tokenizer/`：

- Tokenizer 资产加载与校验（Qwen3-0.6B）；
- Encode/Decode 与 HF `tokenizers` 金标准对照；
- 流式 Decode 与 UTF-8 边界；
- Qwen3 Chat 模板与 HF `apply_chat_template` 对照；
- 错误路径、并发只读安全；
- Runtime JSON 解析器大对象回归（词表加载性能修复）；
- 加载/编码性能基线。

未覆盖：Android NDK/真机、非 Qwen3 Tokenizer 变体、加载峰值内存的
手机端实测。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-21 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | zig clang wrapper（`Graph/tools/zig_cxx.cmd`，Clang 22.1.8） |
| C 编译器 | `Service/tools/zig_cc.cmd`（zig cc） |
| 语言标准 | C++17 / C99 |
| Python | 3.13.3 |
| 金标准库 | `tokenizers 0.22.2`、`transformers 5.9.0` |
| 模型资产 | `Models/Qwen3/tokenizer.json`（11.4 MB，SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`） |
| 第三方库 | utf8proc v2.10.0（MIT，`third_party/utf8proc/`） |

## 3. 构建

```bash
cd d:/Mobile_Net
cmake -S Service -B Service/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=d:/Mobile_Net/Service/tools/zig_cc.cmd \
  -DCMAKE_CXX_COMPILER=d:/Mobile_Net/Graph/tools/zig_cxx.cmd
cmake --build Service/build
```

构建结果：`tokenizer_test.exe`、`libmobilenet_tokenizer.a` 与
`libmobilenet_utf8proc.a` 生成成功，无编译错误。

## 4. 金标准生成

```bash
python Service/tokenizer/tests/gen_golden.py \
  --model-dir Models/Qwen3 \
  --output Service/tokenizer/tests/data/golden.json
```

实际输出：

```text
wrote Service\tokenizer\tests\data\golden.json (82 encode, 9 chat)
```

金标准包含 22 条固定语料 + 60 条确定性随机语料（种子 20261121），
以及 9 条 Chat 模板用例。

## 5. 测试命令与实际输出

### 5.1 直接运行

```bash
./Service/build/tokenizer/tokenizer_test.exe \
  Models/Qwen3 Service/tokenizer/tests/data/golden.json
```

实际输出：

```text
INFO: vocab=151669 eos=2 chat=1
PERF: json_large_keys=20000 parse=4.78 ms
PERF: load=163.81 ms, encode_1KB_avg=0.160 ms, tokens=231
all tokenizer checks passed
```

退出码：`0`。性能数值随文件缓存波动，以上为一次实测。

### 5.2 CTest

```bash
ctest --test-dir Service/build --output-on-failure
```

实际输出：

```text
Test project D:/Mobile_Net/Service/build
    Start 1: tokenizer_test
1/2 Test #1: tokenizer_test ...................   Passed    0.39 sec
    Start 2: session_test
2/2 Test #2: session_test .....................   Passed    4.62 sec

100% tests passed out of 2

Total Test time (real) =   5.14 sec
```

## 6. 用例与结论

| 用例 | 输入/动作 | 期望 | 实际 |
| --- | --- | --- | --- |
| 加载与元信息 | `Models/Qwen3` | `kOk`；vocab=151669；EOS=[151645,151643]；支持 Chat | 通过 |
| ASCII | `Hello` | ids `[9707]` | 通过 |
| 英文句子 | `The capital of France is` | `[785, 6722, 315, 9625, 374]` | 通过 |
| 中文与全角标点 | `你好，世界！` | `[108386, 3837, 99489, 6313]` | 通过 |
| Emoji | `emoji 😀🚀` | `[37523, 90316, 145836]` | 通过 |
| ZWJ Emoji | 👨‍👩‍👧‍👦 | 与 HF 一致 | 通过 |
| NFC/NFD | `café` / `cafe\u0301` | `[924, 58858, ...]` 一致 | 通过 |
| 空白回溯 | `a\n\nb   c` | `[64, 271, 65, 256, 272]` | 通过 |
| added/special token | `<|im_start|>user\nhi<|im_end|>\n` | `[151644, 872, 198, 6023, 151645, 198]` | 通过 |
| 非 special added token | `<tool_call>...</tool_call>` | 151657/151658 保留 | 通过 |
| `<think>` 文本 | `<think>reasoning</think>answer` | 151667/151668 保留 | 通过 |
| 前导标点 | `  !` | `[220, 753]` | 通过 |
| 多空格 | `a   b` | `[64, 256, 293]` | 通过 |
| contraction | `'s 'S 're 'll we're` | 与 HF 一致 | 通过 |
| 数字 | `12345` | 每个数字单独成 token | 通过 |
| 代码 | `def add(a, b):\n    return a + b\n` | 与 HF 一致 | 通过 |
| 混合空白 | `\t \r\n\n  x` | `[3677, 80823, 220, 856]` | 通过 |
| NBSP | `\u00a0` | `[4102]` | 通过 |
| 混合语言 | `你好Hello123 世界` | `[108386, 9707, 16, 17, 18, 220, 99489]` | 通过 |
| 长时间字符串 | ISO 时间、长英文段落 | 与 HF 一致 | 通过 |
| 60 条随机语料 | 固定种子随机混排 | 与 HF 完全一致 | 通过 |
| Decode | 上述 ids | 与 HF `decode(skip_special_tokens=true)` 一致 | 通过 |
| Decode keep-special | 上述 ids | 与 HF `skip_special_tokens=false` 一致 | 通过 |
| 流式切分 | 每个用例逐 token Push | 拼接结果与整段 Decode 逐字节一致 | 通过 |
| 流式非法字节 | 单字节 token id 94 | 输出 U+FFFD，不中断 | 通过 |
| Chat 单轮 | user `Hello` | 与 HF 模板一致 | 通过 |
| Chat system+user | system + user | 与 HF 模板一致 | 通过 |
| Chat 多轮 | system/user/assistant/user | 与 HF 模板一致 | 通过 |
| Chat thinking 三态 | default/false/true | 与 HF 模板逐字节一致 | 通过 |
| Chat 历史 reasoning | assistant 含 `</think>` | 与 HF 模板一致 | 通过 |
| Chat 末尾 assistant | user/assistant，无生成提示 | 与 HF thinking 分支一致 | 通过 |
| Chat 末尾 reasoning | 末尾 assistant 含 `</think>` | 与 HF 模板一致 | 通过 |
| 非首条 system | user/system/user | 角色名保持 `system`，与 HF 一致 | 通过 |
| 非法 UTF-8 | `"\xFF"` | `kInvalidArgument` | 通过 |
| 未知 id | 200000 | 默认跳过；关闭 skip 后报错 | 通过 |
| 空 Chat 消息 | `{}` | `kInvalidArgument` | 通过 |
| tool 角色 | `{"role":"tool"}` | `kUnsupportedOp` | 通过 |
| 并发 | 4 线程 × 50 次 Encode/Decode | 结果一致，无崩溃 | 通过 |
| 性能 | 1 KB 文本 50 次编码 | 记录基线 | 0.160 ms/次 |
| 加载性能 | 加载 Qwen3 Tokenizer | 记录基线 | 163.81 ms（热缓存） |

## 7. Runtime JSON 解析器回归

背景：原 `Runtime/common/json.cc` 对对象重复键做线性扫描，151k 词表的
`tokenizer.json` 加载耗时约 44 s。已改为哈希集合检测重复键，语义不变。

回归结果：

| 检查 | 期望 | 实际 |
| --- | --- | --- |
| 重复键 `{"a":1,"a":2}` | 报错 | 通过 |
| 20,000 键对象 | 解析成功 | 通过；15.76 ms |
| 300 层嵌套 | 报错（深度上限 256） | 通过 |
| 尾随数据 `1 2` | 报错 | 通过 |
| Runtime 快速测试（7 项） | 全部通过 | 7/7 通过，28.48 s |

Runtime 快速测试命令：

```bash
ctest --test-dir Runtime/build --output-on-failure \
  -R "backend_header_check|backend_contract_test|op_env_test|ocl_sha256_test|ocl_contract_test|ocl_vs_mock_test|ocl_kv_mode_mismatch"
```

实际结果：

```text
100% tests passed out of 7
Total Test time (real) =  28.48 sec
```

## 8. 未覆盖项与风险

- Android arm64 NDK 构建与真机运行未执行；`utf8proc` 为纯 C，预期可移植，
  但需在 NDK 环境补充构建验证。
- 加载峰值内存未在手机端实测；桌面端加载后释放 JSON 树，常驻内存为
  词表/map/trie，仍需真机测量。
- 非 Qwen3 Tokenizer 变体（不同 normalizer/pretokenizer/added token 标志）
  不在基础版范围，加载时会返回 `kUnsupportedOp`。
- Unicode 版本差异（utf8proc v2.10.0 与 HF tokenizers 内置版本）可能影响
  极少数字符；金标准语料未覆盖所有 Unicode 码点。

## 9. 结论

Tokenizer 模块在 Qwen3-0.6B 上达到设计目标：

- 82 条 Encode 金标准与 9 条 Chat 模板用例全部与 HF 一致；
- 流式解码与整段解码逐字节一致；
- 错误路径、并发与 JSON 解析器回归通过；
- 加载亚秒级、1 KB 编码亚毫秒级，满足基础版使用。

## 10. Review 修复记录

| 问题 | 修复 | 验证 |
| --- | --- | --- |
| 非首条 system 被写成 user 角色 | 保留 `message.role` | 新增 golden 用例 |
| 生成 token 非法 UTF-8 导致流式失败 | StreamDecoder lossy U+FFFD | id 94 用例 |
| tokenizer.json 加载 40+ 秒 | Runtime JSON 重复键哈希集合 | 20k 键 5 ms；加载亚秒级 |

