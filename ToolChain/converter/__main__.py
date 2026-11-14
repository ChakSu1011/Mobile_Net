"""模块：Converter 命令行入口。

职责：
  解析参数，执行 Qwen3 转换并写出 ``model_graph.json`` 与
  ``conversion_report.json``，在终端打印图计数摘要。

边界：
  只做参数解析与结果输出，不实现转换逻辑。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ToolChain.converter.converter import (
    convert,
    write_conversion_outputs,
)


def main(argv: list[str] | None = None) -> int:
    """执行命令行转换。

    参数：
      argv：参数列表；None 表示使用 ``sys.argv[1:]``。

    返回：
      进程退出码，正常为 0。

    异常：
      BuildError：Adapter 未注册或模型图非法；
      ValueError/RuntimeError：配置或构图失败；
      OSError：文件读写失败。
    """
    parser = argparse.ArgumentParser(
        description="Convert a model to a Graph graph")
    parser.add_argument(
        "--adapter",
        default="qwen3",
        choices=["qwen3"],
        help="model adapter name",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to config.json",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="optional safetensors file or directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output directory",
    )
    args = parser.parse_args(argv)

    result = convert(
        args.adapter,
        args.config,
        weights_path=args.weights,
    )
    paths = write_conversion_outputs(result, args.output)
    summary = result.model_graph.summary()
    print(f"adapter {result.adapter}")
    print(f"graph {summary['name']}")
    print(f"inputs {summary['inputs']}")
    print(f"constants {summary['constants']}")
    print(f"compute_ops {summary['ops']}")
    print(f"outputs {summary['outputs']}")
    print(f"total_tensors {summary['total_tensors']}")
    print(f"constant_bytes {summary['constant_bytes']}")
    print(f"model_graph {paths['model_graph']}")
    print(f"report {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
