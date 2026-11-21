# Session 模块测试结果

## 1. 测试范围

本轮验证 `Service/session/`：

- 服务初始化/启动/停止与配置加载；
- HTTP/OpenAI 端点：health、models、chat/completions（非流式/SSE）；
- 原生会话：创建/查询/删除/多轮/隔离；
- max_tokens 截断、stop 字符串、错误映射、会话忙、请求超时；
- OpenAPI 文档；
- FakeEngine 集成测试与真实 OCL Qwen3-0.6B 冒烟。

未覆盖：Android 真机、TLS/鉴权、PrefixCache、采样参数生效、长上下文/
并发真实请求、长时间稳定性压测。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-21 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | zig clang wrapper（Clang 22.1.8） |
| 语言标准 | C++17 |
| Tokenizer 资产 | `Models/Qwen3/`（Qwen3-0.6B） |
| OCL 模型包 | `Models/Qwen3/artifacts/qwen3_0_6b_kv/` |
| Engine | Runtime Engine，OCL 后端，`threaded=true` |
| 测试双 | `FakeEngine`（固定 token 序列/延迟） |
| 模型加载 | `verify_weights_sha256=false`（仅冒烟；生产默认 true） |

## 3. 构建

```bash
cd d:/Mobile_Net
cmake -S Service -B Service/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=d:/Mobile_Net/Service/tools/zig_cc.cmd \
  -DCMAKE_CXX_COMPILER=d:/Mobile_Net/Graph/tools/zig_cxx.cmd
cmake --build Service/build
```

构建结果：`mobilenet_service.exe`、`session_test.exe`、
`libmobilenet_session.a` 生成成功，无编译错误。

## 4. FakeEngine 集成测试

### 4.1 命令

```bash
./Service/build/session/session_test.exe Models/Qwen3
```

实际输出：

```text
all session checks passed
```

退出码：`0`。

### 4.2 用例与结论

| 用例 | 输入/动作 | 期望 | 实际 |
| --- | --- | --- | --- |
| 初始化/启动 | FakeEngine + 端口 0 | 成功；端口分配 | 通过 |
| `/health` | GET | 200；engine_state=RUNNING；model 正确 | 通过 |
| `/v1/models` | GET | 200；data[0].id=qwen3_0_6b | 通过 |
| 非流式 chat | user `Hi` | 200；content=`Hello world`；finish_reason=stop；usage>0 | 通过 |
| max_tokens=1 | user `Hi` | finish_reason=length | 通过 |
| 流式 chat | `stream=true` | SSE 顺序正确；拼接=`Hello world`；finish_reason=stop；`[DONE]` | 通过 |
| stop 字符串 | stop=`["world"]` | content=`Hello ` | 通过 |
| 创建会话 | system_prompt | 200；session_id 非空 | 通过 |
| 会话单轮 | message `Hi` | 200；content=`Hello world`；session_id 回显 | 通过 |
| 会话消息计数 | 单轮后 GET | message_count=3（system+user+assistant） | 通过 |
| 会话两轮 | 第二轮 message `Again` | message_count=5 | 通过 |
| 有状态流式 | 会话内 `stream=true` | SSE content=`Hello world`；session_id 回显；message_count=7 | 通过 |
| 会话隔离 | 新建第二会话 | message_count=0 | 通过 |
| 删除会话 | DELETE | 200；再次 GET 返回 404 | 通过 |
| 非法 JSON | `not-json` | 400；error 信封 | 通过 |
| 模型不匹配 | model=`other` | 404 model_not_found | 通过 |
| tools 参数 | `tools: []` | 400 unsupported_parameter | 通过 |
| n=2 | `n: 2` | 400 unsupported_parameter | 通过 |
| 未知会话 | `/v1/sessions/missing/chat` | 404 session_not_found | 通过 |
| 上下文超限 | 2000 个 `a` | 400 context_length_exceeded | 通过 |
| OpenAPI | GET `/openapi.json` | 200；可解析；含 `/v1/chat/completions` | 通过 |
| 会话忙 | FakeEngine 300 ms/Token；并发两次 chat | 一个 200，一个 409 session_busy | 通过 |
| 客户端断开 | 流式请求 1 ms 读超时后关闭连接 | 服务取消请求；随后 health 仍 200 | 通过 |
| 请求超时 | FakeEngine 400 ms；超时 50 ms | 504；随后 health 仍 200 | 通过 |

### 4.3 CTest

```bash
ctest --test-dir Service/build --output-on-failure
```

实际输出：

```text
Test project D:/Mobile_Net/Service/build
    Start 1: tokenizer_test
1/2 Test #1: tokenizer_test ...................   Passed    0.27 sec
    Start 2: session_test
2/2 Test #2: session_test .....................   Passed    2.95 sec

100% tests passed out of 2

Total Test time (real) =   3.26 sec
```

## 5. OCL 真实模型冒烟

### 5.1 配置

- 后端：`ocl`
- 模型包：`Models/Qwen3/artifacts/qwen3_0_6b_kv`
- `max_num_tokens=256`，`max_concurrent_requests=1`
- `kv_cache_bytes=67108864`（64 MiB；Engine 要求显式配置）
- `default_max_tokens=1`；请求内设置 `max_tokens=4`
- `enable_thinking=false`
- `verify_weights_sha256=false`（仅冒烟）
- 端口：18080

### 5.2 命令

```bash
python TempSpace/service_validation_20261121_145431/ocl_smoke.py \
  D:/Mobile_Net/Service/build/session/mobilenet_service.exe \
  D:/Mobile_Net/TempSpace/service_validation_20261121_145431/ocl_config.json \
  D:/Mobile_Net/TempSpace/service_validation_20261121_145431/ocl_service_final.log
```

### 5.3 实际输出

```text
HEALTH {"status":"ok","engine_state":"RUNNING","model":"qwen3_0_6b","uptime_s":0,"stats":{"total_steps":27,"total_requests":0,"total_tokens":0,"running_requests":0,"waiting_requests":0,"kv_used_blocks":0,"kv_free_blocks":36}}
NONSTREAM_STATUS 200
NONSTREAM_RESPONSE {"id":"chatcmpl-a95f70f25c42861c","object":"chat.completion","created":1795244278,"model":"qwen3_0_6b","choices":[{"index":0,"message":{"role":"assistant","content":"The capital of France"},"finish_reason":"length"}],"usage":{"prompt_tokens":17,"completion_tokens":4,"total_tokens":21}}
NONSTREAM_CONTENT 'The capital of France'
STREAM_STATUS 200
STREAM_EVENTS 7
STREAM_CONTENT 'The capital of France'
STREAM_FINISH length
STREAM_USAGE {'prompt_tokens': 17, 'completion_tokens': 4, 'total_tokens': 21}
STREAM_DONE True
```

退出码：`0`。

服务日志（完整）：

```text
[14:57:57.897] [info] server initialized: model=qwen3_0_6b max_num_tokens=256 backend=ocl
[14:57:57.902] [info] listening on http://127.0.0.1:18080
[14:57:58.305] [info] GET /health handled=1 elapsed_ms=0.054700
[14:57:58.830] [info] POST /v1/chat/completions handled=1 elapsed_ms=502.183800
[14:57:59.265] [info] POST /v1/chat/completions handled=1 elapsed_ms=416.568100
```

结论：

- 真实 Qwen3-0.6B + OCL 后端可加载并运行；
- 非流式与流式返回内容逐字节一致；
- usage、finish_reason、`[DONE]` 正确；
- 单请求（17 prompt + 4 生成）端到端耗时约 0.42–0.50 s（本机 OCL）。

### 5.4 真实文字交互（单轮问答）

在上节配置基础上，使用 prompt `What is the capital of France?` 与
`max_tokens=8` 再次请求：

```text
STATUS 200
RESPONSE {"id":"chatcmpl-35a3b29766f10f1f","object":"chat.completion","created":1795244716,"model":"qwen3_0_6b","choices":[{"index":0,"message":{"role":"assistant","content":"The capital of France is **Paris**"},"finish_reason":"length"}],"usage":{"prompt_tokens":19,"completion_tokens":8,"total_tokens":27}}
CONTENT 'The capital of France is **Paris**'
FINISH length
USAGE {'prompt_tokens': 19, 'completion_tokens': 8, 'total_tokens': 27}
```

结论：Qwen3-0.6B 能根据文本 prompt 生成可读答案。

### 5.5 真实多轮会话

创建带 system prompt 的会话，连续两轮对话：

```text
SESSION c0e8ffcec569266e
FIRST {"id": "chatcmpl-29122043b25ddc55", "object": "chat.completion", "created": 1795244729, "model": "qwen3_0_6b", "session_id": "c0e8ffcec569266e", "choices": [{"index": 0, "message": {"role": "assistant", "content": "The capital of France is Paris."}, "finish_reason": "length"}], "usage": {"prompt_tokens": 27, "completion_tokens": 8, "total_tokens": 35}}
SECOND {"id": "chatcmpl-c171c1312579a34b", "object": "chat.completion", "created": 1795244730, "model": "qwen3_0_6b", "session_id": "c0e8ffcec569266e", "choices": [{"index": 0, "message": {"role": "assistant", "content": "The capital of Germany is Berlin."}, "finish_reason": "length"}], "usage": {"prompt_tokens": 47, "completion_tokens": 8, "total_tokens": 55}}
FIRST_CONTENT 'The capital of France is Paris.'
SECOND_CONTENT 'The capital of Germany is Berlin.'
```

结论：

- 第二轮 `prompt_tokens` 从 27 增至 47，证明会话历史被带入下一轮；
- 两个响应均带同一个 `session_id`；
- 真实多轮文字交互可用。

## 6. Runtime 依赖回归

Session 使用 Runtime Engine 与 JSON 解析器。本轮修改
`Runtime/common/json.cc` 的重复键检测（线性 → 哈希集合）后，
Runtime 快速测试全部通过：

```text
100% tests passed out of 7
Total Test time (real) =  28.48 sec
```

测试项：`backend_header_check`、`backend_contract_test`、`op_env_test`、
`ocl_sha256_test`、`ocl_contract_test`、`ocl_vs_mock_test`、
`ocl_kv_mode_mismatch`。

## 7. 未覆盖项与风险

- 真实 OCL 覆盖了单轮、流式与两轮会话；长上下文、并发真实请求与长时间
  稳定性未做压测。
- 当前 Engine 为 argmax，`temperature/top_p/top_k/seed` 接受但不生效；
  需要采样时需等待 Engine 阶段三。
- 无 PrefixCache：多轮重复 Prefill，延迟随历史增长。
- 无 TLS/鉴权、无 keep-alive、只支持 Content-Length 请求体；基础版面向
  本机/受信局域网。
- 单请求 token 上限 256（模型包规划值）；超限会截断旧消息或返回 400。
- Android 真机未验证；Winsock/POSIX 分支需在 NDK 环境补充构建与运行。
- FakeEngine 不覆盖真实 Engine 的调度/数值细节；OCL 冒烟作为补充。

## 8. 结论

Session 模块达到基础版目标：

- P0 端点全部可用，OpenAI 基础客户端可完成非流式与流式对话；
- 多轮会话、隔离、删除、stop、忙、超时、错误映射均通过；
- 真实 Qwen3-0.6B OCL 冒烟通过，文本交互可用；
- 四件套（设计文档、代码、测试结果、skill）齐全。
