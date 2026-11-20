"""模块：Qwen3-0.6B PyTorch 金标准生成脚本。

职责：
  使用 Transformers 加载 Models/Qwen3，按固定 prompt 执行一次完整前向，
  保存 prompt token ids、argmax token 与 top-5 logits，供 Runtime Mock
  端到端测试比较。

边界：
  不加载 Runtime Graph、不写模型包、不修改仓库其它文件；
  需要本机安装 torch 与 transformers。

用法：
  python Tests/golden/generate_qwen3_golden.py \
    --model-dir Models/Qwen3 \
    --output Tests/golden/qwen3_0_6b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ["Hello", "The capital of France is", "1+1="]


def main() -> int:
    """生成金标准 JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="Models/Qwen3")
    parser.add_argument(
        "--output", default="Tests/golden/qwen3_0_6b.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32)
    model.eval()

    output = {
        "model": "qwen3_0_6b",
        "package": "Models/Qwen3/artifacts/qwen3_0_6b",
        "dtype": "float32_reference",
        "prompts": [],
    }
    for text in PROMPTS:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
        top = torch.topk(logits, 5)
        output["prompts"].append({
            "text": text,
            "prompt_token_ids": input_ids[0].tolist(),
            "expected_token_id": int(top.indices[0]),
            "top5_token_ids": top.indices.tolist(),
            "top5_logits": [float(value) for value in top.values],
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
