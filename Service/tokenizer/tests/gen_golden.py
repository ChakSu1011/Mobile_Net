"""模块：Tokenizer 金标准生成脚本。

职责：
  使用 HuggingFace `tokenizers` 与 `transformers` 对 Qwen3-0.6B 生成
  Encode/Decode 与 Chat 模板金标准，供 C++ 单元测试对照。

边界：
  只读取本地模型目录，不访问网络；输出 JSON 不包含模型权重。
  脚本依赖 `tokenizers`、`transformers`，仅在开发机测试使用。

主要逻辑：
  1. 固定语料覆盖 ASCII、CJK、Emoji、代码、空白、NFC/NFD、特殊 token；
  2. 用 tokenizers 生成 ids/tokens/decoded；
  3. 用 transformers apply_chat_template 生成 Chat prompt；
  4. 写入 tests/data/golden.json，并记录库版本与资产哈希。

用法：
  python Service/tokenizer/tests/gen_golden.py \
    --model-dir Models/Qwen3 \
    --output Service/tokenizer/tests/data/golden.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]

# Encode/Decode 语料。
ENCODE_CASES = [
    ("empty", ""),
    ("ascii_hello", "Hello"),
    ("capital_question", "The capital of France is"),
    ("cjk_punct", "你好，世界！"),
    ("emoji_basic", "emoji \U0001f600\U0001f680"),
    ("emoji_zwj", "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466"),
    ("nfc_accent", "café (NFC)"),
    ("nfd_accent", "café (NFD)"),
    ("newlines", "a\n\nb   c"),
    ("special_in_text", "<|im_start|>user\nhi<|im_end|>\n"),
    ("tool_special", "<tool_call>{\"name\":\"x\"}</tool_call>"),
    ("think_text", "<think>reasoning</think>answer"),
    ("leading_punct", "  !"),
    ("multi_space", "a   b"),
    ("contractions", "'s 'S 're 'll we're"),
    ("digits", "12345"),
    ("code", "def add(a, b):\n    return a + b\n"),
    ("mixed_ws", "\t \r\n\n  x"),
    ("nbsp", " "),
    ("mixed_language", "你好Hello123 世界"),
    ("iso_time", "2026-11-21T10:00:00+08:00"),
    ("long_text", ("Large language model serving on mobile devices. " * 8).strip()),
]

# 确定性随机语料：覆盖预分词、空白回溯、Unicode 与 added token 混排。
_FUZZ_POOL = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " .,!?;:'\"-_()[]{}+=*/\\\n\r\t"
    "你好世界，！。"
    "é́"
    "  　"
    "😀🚀"
    "<|im_start|><|im_end|><tool_call></tool_call><think></think>"
)
_FUZZ_RANDOM = random.Random(20261121)
FUZZ_CASES = [
    (f"fuzz_{index:03d}", "".join(
        _FUZZ_RANDOM.choice(_FUZZ_POOL)
        for _ in range(_FUZZ_RANDOM.randint(0, 60))
    ))
    for index in range(60)
]

# Chat 模板语料。
CHAT_CASES = [
    {
        "name": "single_user_default",
        "messages": [{"role": "user", "content": "Hello"}],
        "add_generation_prompt": True,
        "enable_thinking": "default",
    },
    {
        "name": "single_user_no_think",
        "messages": [{"role": "user", "content": "Hello"}],
        "add_generation_prompt": True,
        "enable_thinking": "false",
    },
    {
        "name": "system_user_think",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "add_generation_prompt": True,
        "enable_thinking": "true",
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "And 2+2?"},
        ],
        "add_generation_prompt": True,
        "enable_thinking": "false",
    },
    {
        "name": "assistant_think_history",
        "messages": [
            {"role": "user", "content": "Solve."},
            {"role": "assistant", "content": "<think>try</think>The answer is 4."},
            {"role": "user", "content": "Why?"},
        ],
        "add_generation_prompt": True,
        "enable_thinking": "false",
    },
    {
        "name": "no_generation_prompt",
        "messages": [{"role": "user", "content": "Hello"}],
        "add_generation_prompt": False,
        "enable_thinking": "default",
    },
    {
        "name": "last_assistant_no_prompt",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        "add_generation_prompt": False,
        "enable_thinking": "default",
    },
    {
        "name": "last_assistant_reasoning",
        "messages": [
            {"role": "user", "content": "Solve."},
            {"role": "assistant", "content": "<think>try</think>The answer is 4."},
        ],
        "add_generation_prompt": False,
        "enable_thinking": "default",
    },
]


def build_chat_prompt(hf_tokenizer: AutoTokenizer, case: dict) -> str:
    """生成一个 Chat 用例的模板文本。"""
    kwargs = {}
    mode = case["enable_thinking"]
    if mode != "default":
        kwargs["enable_thinking"] = mode == "true"
    return hf_tokenizer.apply_chat_template(
        case["messages"],
        tokenize=False,
        add_generation_prompt=case["add_generation_prompt"],
        **kwargs,
    )


def main() -> None:
    """生成并写出 golden.json。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "Models" / "Qwen3"),
        help="Qwen3 模型目录",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("data") / "golden.json"),
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer_json = model_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    hf_tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), use_fast=True, trust_remote_code=False
    )

    encode_cases = []
    for name, text in ENCODE_CASES + FUZZ_CASES:
        encoding = tokenizer.encode(text)
        encode_cases.append(
            {
                "name": name,
                "text": text,
                "ids": encoding.ids,
                "tokens": encoding.tokens,
                "decoded": tokenizer.decode(
                    encoding.ids, skip_special_tokens=True
                ),
                "decoded_keep_special": tokenizer.decode(
                    encoding.ids, skip_special_tokens=False
                ),
            }
        )

    chat_cases = []
    for case in CHAT_CASES:
        chat_cases.append(
            {
                "name": case["name"],
                "messages": case["messages"],
                "add_generation_prompt": case["add_generation_prompt"],
                "enable_thinking": case["enable_thinking"],
                "prompt": build_chat_prompt(hf_tokenizer, case),
            }
        )

    sha256 = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
    payload = {
        "meta": {
            "model": "Qwen3-0.6B",
            "tokenizer_json_sha256": sha256,
            "tokenizers_version": __import__("tokenizers").__version__,
            "transformers_version": __import__("transformers").__version__,
        },
        "encode_cases": encode_cases,
        "chat_cases": chat_cases,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output} ({len(encode_cases)} encode, {len(chat_cases)} chat)")


if __name__ == "__main__":
    main()
