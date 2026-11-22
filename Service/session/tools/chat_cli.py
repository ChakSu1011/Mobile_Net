"""模块：Session 交互式命令行聊天客户端。

职责：
  通过 HTTP 连接本地 Session 服务，创建有状态会话；从标准输入读取文本，
  把回复打印到终端。支持流式 SSE、多轮上下文、会话重置。

边界：
  只使用 Python 标准库；不加载模型、不依赖 openai SDK。
  默认连接 http://127.0.0.1:18080，服务需先启动。

主要逻辑：
  1. 启动时 POST /v1/sessions 创建会话（可带 system prompt）；
  2. 每次输入 POST /v1/sessions/{id}/chat，流式或非流式读取回复；
  3. 服务端保存历史，因此客户端不需要拼接 messages；
  4. /reset 创建新会话，/exit 退出。

用法：
  python Service/session/tools/chat_cli.py
  python Service/session/tools/chat_cli.py --no-stream --max-tokens 64
  python Service/session/tools/chat_cli.py --system "Answer briefly."
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _configure_stdio() -> None:
    """尽量使用 UTF-8 输入输出，减少 Windows 控制台中文乱码。"""
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    """发送 JSON POST 请求并返回解析后的响应。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_error(error: urllib.error.HTTPError) -> str:
    """把 HTTP 错误转换为可读文本。"""
    raw = error.read().decode("utf-8", "replace")
    try:
        payload = json.loads(raw)
        detail = payload.get("error", {})
        message = detail.get("message", raw)
        code = detail.get("code", "")
        return f"[HTTP {error.code}] {message}" + (f" ({code})" if code else "")
    except json.JSONDecodeError:
        return f"[HTTP {error.code}] {raw}"


def create_session(base_url: str, system_prompt: str, timeout: float) -> str:
    """创建会话并返回 session_id。"""
    payload = {}
    if system_prompt:
        payload["system_prompt"] = system_prompt
    data = _post_json(base_url + "/v1/sessions", payload, timeout)
    return data["session_id"]


def chat_non_stream(base_url: str, session_id: str, message: str,
                    model: str, max_tokens: int, timeout: float) -> None:
    """发送一轮对话并打印完整回复。"""
    data = _post_json(
        f"{base_url}/v1/sessions/{session_id}/chat",
        {"model": model, "message": message, "max_tokens": max_tokens},
        timeout,
    )
    choice = data["choices"][0]
    print(choice["message"]["content"])
    usage = data.get("usage")
    if usage:
        print(
            f"[usage] prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} "
            f"finish={choice.get('finish_reason')}"
        )


def chat_stream(base_url: str, session_id: str, message: str,
                model: str, max_tokens: int, timeout: float) -> None:
    """发送一轮对话并按 SSE 增量打印回复。"""
    payload = {
        "model": model,
        "message": message,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/sessions/{session_id}/chat",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    finish_reason = None
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data_text = line[6:]
            if data_text == "[DONE]":
                break
            data = json.loads(data_text)
            if "error" in data:
                error = data["error"]
                print(f"\n[stream error] {error.get('message')}")
                break
            choice = data["choices"][0]
            piece = choice.get("delta", {}).get("content", "")
            if piece:
                print(piece, end="", flush=True)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if "usage" in data:
                usage = data["usage"]
    print()
    if usage:
        print(
            f"[usage] prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} "
            f"finish={finish_reason}"
        )


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="MobileNet Session 交互式聊天客户端")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080",
                        help="服务地址，默认 http://127.0.0.1:18080")
    parser.add_argument("--model", default="qwen3_0_6b", help="模型名")
    parser.add_argument("--system", default="You are a helpful assistant.",
                        help="system prompt；传空字符串表示不设置")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="单轮最大生成 token 数，默认 512；"
                             "可用 /max <n> 在会话中调整")
    parser.add_argument("--no-stream", action="store_true",
                        help="关闭流式输出，等待完整回复")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="单请求超时秒数，默认 180")
    parser.add_argument("--session-id", default="",
                        help="复用已有 session_id；为空则新建会话")
    args = parser.parse_args()

    _configure_stdio()
    try:
        session_id = args.session_id or create_session(
            args.base_url, args.system, args.timeout)
    except urllib.error.HTTPError as error:
        print(_format_error(error))
        return 1
    except urllib.error.URLError as error:
        print(f"无法连接服务 {args.base_url}: {error.reason}")
        print("请先在另一个终端启动 mobilenet_service.exe。")
        return 1

    stream = not args.no_stream
    max_tokens = args.max_tokens
    print("MobileNet Session 聊天客户端")
    print(f"服务: {args.base_url}  模型: {args.model}  会话: {session_id}")
    print(f"单轮生成上限: {max_tokens}")
    print("输入文字开始对话；/reset 新会话，/stream 切换流式，"
          "/max <n> 调整生成长度，/exit 退出。")
    print("=" * 60)

    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        if not text:
            continue
        if text in ("/exit", "/quit", "exit", "quit"):
            print("再见。")
            return 0
        if text == "/help":
            print("命令: /reset 新会话  /stream 切换流式  /max <n> 调整单轮"
                  "生成上限  /session 显示会话 ID  /exit 退出")
            continue
        if text == "/session":
            print(session_id)
            continue
        if text == "/stream":
            stream = not stream
            print(f"流式输出: {'开' if stream else '关'}")
            continue
        if text.startswith("/max "):
            try:
                value = int(text.split(maxsplit=1)[1])
                if value < 1 or value > 2048:
                    raise ValueError
                max_tokens = value
                print(f"单轮生成上限已设为 {max_tokens}")
            except (ValueError, IndexError):
                print("用法: /max <1-2048>")
            continue
        if text == "/reset":
            try:
                session_id = create_session(args.base_url, args.system,
                                            args.timeout)
                print(f"已创建新会话: {session_id}")
            except urllib.error.HTTPError as error:
                print(_format_error(error))
            except urllib.error.URLError as error:
                print(f"无法连接服务: {error.reason}")
            continue
        if text.startswith("/"):
            print("未知命令，输入 /help 查看可用命令。")
            continue

        try:
            if stream:
                chat_stream(args.base_url, session_id, text, args.model,
                            max_tokens, args.timeout)
            else:
                chat_non_stream(args.base_url, session_id, text, args.model,
                                max_tokens, args.timeout)
        except urllib.error.HTTPError as error:
            print(_format_error(error))
            print("如果提示 session not found，服务可能已重启；输入 /reset"
                  " 创建新会话。")
        except urllib.error.URLError as error:
            print(f"无法连接服务: {error.reason}")
        except TimeoutError:
            print("请求超时；服务可能仍在推理，稍后重试。")


if __name__ == "__main__":
    raise SystemExit(main())
