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
| max_completion_tokens=null | 同时带 max_tokens=1 | 回退到 max_tokens，finish_reason=length | 通过 |
| 流式 chat | `stream=true` + include_usage | SSE 顺序正确；拼接=`Hello world`；finish_reason=stop；usage 存在；`[DONE]` | 通过 |
| stop 字符串 | stop=`["world"]` | content=`Hello ` | 通过 |
| 跨 token stop | stop=`["lo wo"]` | content=`Hel` | 通过 |
| EOS usage | 终态 token 为 EOS | content 不含 EOS；completion_tokens=2 | 通过 |
| 创建会话 | system_prompt | 200；session_id 非空 | 通过 |
| 会话单轮 | message `Hi` | 200；content=`Hello world`；session_id 回显 | 通过 |
| 会话消息计数 | 单轮后 GET | message_count=3（system+user+assistant） | 通过 |
| 会话两轮 | 第二轮 message `Again` | message_count=5 | 通过 |
| 有状态流式 | 会话内 `stream=true` | SSE content=`Hello world`；session_id 回显；message_count=7 | 通过 |
| 会话隔离 | 新建第二会话 | message_count=0 | 通过 |
| 历史上限 | max_history_messages=4；三轮对话 | message_count 裁剪为 4 | 通过 |
| Stop 后重启 | Stop → Init → Start → chat | 聊天 200 | 通过 |
| 端口独占 | 两个 Server 绑定同一端口 | 第二个 Start 失败 | 通过 |
| 删除会话 | DELETE | 200；再次 GET 返回 404 | 通过 |
| 非法 JSON | `not-json` | 400；error 信封 | 通过 |
| 模型不匹配 | model=`other` | 404 model_not_found | 通过 |
| tools 参数 | `tools: []` | 400 unsupported_parameter | 通过 |
| n=2 | `n: 2` | 400 unsupported_parameter | 通过 |
| 未知会话 | `/v1/sessions/missing/chat` | 404 session_not_found | 通过 |
| 上下文超限 | 2000 个 `a`（预算 256） | 400 context_length_exceeded | 通过 |
| 长上下文预算 | 预算 2048；500 token prompt | 200，正常生成 | 通过 |
| 长上下文超限 | 预算 2048；3500 token prompt | 400 context_length_exceeded | 通过 |
| 成功截断 | 5 条各 100 字符的历史消息 | 丢弃旧消息后 200；content=`Hello world` | 通过 |
| 方法不匹配 | GET `/v1/chat/completions` | 405 | 通过 |
| 请求体上限 | max_body_bytes=512；发送更大 body | 413 | 通过 |
| OpenAPI | GET `/openapi.json` | 200；可解析；含 `/v1/chat/completions` | 通过 |
| 会话忙 | FakeEngine 300 ms/Token；并发两次 chat | 一个 200，一个 409 session_busy | 通过 |
| 多连接并发 | 4 线程并发无状态 chat | 全部 200；内容一致；无串扰 | 通过 |
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
1/2 Test #1: tokenizer_test ...................   Passed    0.39 sec
    Start 2: session_test
2/2 Test #2: session_test .....................   Passed    4.62 sec

100% tests passed out of 2

Total Test time (real) =   5.14 sec
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
  D:/Mobile_Net/TempSpace/service_validation_20261121_145431/ocl_service_postfix.log
```

### 5.3 实际输出

```text
HEALTH {"status":"ok","engine_state":"RUNNING","model":"qwen3_0_6b","uptime_s":1,"stats":{"total_steps":67,"total_requests":0,"total_tokens":0,"running_requests":0,"waiting_requests":0,"kv_used_blocks":0,"kv_free_blocks":36}}
NONSTREAM_STATUS 200
NONSTREAM_RESPONSE {"id":"chatcmpl-5d4bb7de5d0ffeba","object":"chat.completion","created":1795245460,"model":"qwen3_0_6b","choices":[{"index":0,"message":{"role":"assistant","content":"The capital of France"},"finish_reason":"length"}],"usage":{"prompt_tokens":17,"completion_tokens":4,"total_tokens":21}}
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
[15:17:39.489] [info] server initialized: model=qwen3_0_6b max_num_tokens=256 backend=ocl
[15:17:39.491] [info] listening on http://127.0.0.1:18080
[15:17:40.557] [info] GET /health handled=1 elapsed_ms=0.055300
[15:17:40.929] [info] POST /v1/chat/completions handled=1 elapsed_ms=355.788100
[15:17:41.312] [info] POST /v1/chat/completions handled=1 elapsed_ms=358.507000
```

结论：

- 真实 Qwen3-0.6B + OCL 后端可加载并运行；
- 非流式与流式返回内容逐字节一致；
- usage、finish_reason、`[DONE]` 正确；
- 单请求（17 prompt + 4 生成）端到端耗时约 0.36 s（本机 OCL）。
- 最终干净构建后再次执行同一冒烟，结果一致（日志
  `TempSpace/service_validation_20261121_145431/ocl_service_finalclean.log`）。

### 5.4 真实文字交互（单轮问答）

在上节配置基础上，使用 prompt `What is the capital of France?` 与
`max_tokens=8` 再次请求：

```text
STATUS 200
RESPONSE {"id":"chatcmpl-e2417587a69e34d1","object":"chat.completion","created":1795245469,"model":"qwen3_0_6b","choices":[{"index":0,"message":{"role":"assistant","content":"The capital of France is **Paris**"},"finish_reason":"length"}],"usage":{"prompt_tokens":19,"completion_tokens":8,"total_tokens":27}}
CONTENT 'The capital of France is **Paris**'
FINISH length
USAGE {'prompt_tokens': 19, 'completion_tokens': 8, 'total_tokens': 27}
```

结论：Qwen3-0.6B 能根据文本 prompt 生成可读答案。

### 5.5 真实多轮会话

创建带 system prompt 的会话，连续两轮对话：

```text
SESSION 156bcc375c5e6294
FIRST {"id": "chatcmpl-f7c3b407d4192d56", "object": "chat.completion", "created": 1795245473, "model": "qwen3_0_6b", "session_id": "156bcc375c5e6294", "choices": [{"index": 0, "message": {"role": "assistant", "content": "The capital of France is Paris."}, "finish_reason": "length"}], "usage": {"prompt_tokens": 27, "completion_tokens": 7, "total_tokens": 34}}
SECOND {"id": "chatcmpl-583a3007900ef276", "object": "chat.completion", "created": 1795245474, "model": "qwen3_0_6b", "session_id": "156bcc375c5e6294", "choices": [{"index": 0, "message": {"role": "assistant", "content": "The capital of Germany is Berlin."}, "finish_reason": "length"}], "usage": {"prompt_tokens": 47, "completion_tokens": 7, "total_tokens": 54}}
FIRST_CONTENT 'The capital of France is Paris.'
SECOND_CONTENT 'The capital of Germany is Berlin.'
```

结论：

- 第二轮 `prompt_tokens` 从 27 增至 47，证明会话历史被带入下一轮；
- 两个响应均带同一个 `session_id`；
- `completion_tokens=7`：终态 EOS 未计入 usage（与设计契约一致）；
- 真实多轮文字交互可用。

### 5.6 2048 上下文与长 prompt 验证

为支持更长对话，本次将 `engine.max_num_tokens=256`（单步图预算）与
`generation.max_context_tokens=2048`（单请求 prompt+生成预算）解耦，
`kv_cache_bytes` 设为 256 MiB。

设备信息（`ocl_contract_test.exe`）：

```text
device.name: Intel(R) Arc(TM) 130T GPU (16GB)
device.global_bytes: 17612341248
device.max_alloc_bytes: 4294959104
```

显存估算：

| 项目 | 大小 |
| --- | --- |
| 权重 `weights.bin` | 1,192,099,840 B ≈ 1.11 GiB |
| Activation arena（单步 256） | 78,317,568 B ≈ 74.7 MiB |
| KV Cache（256 MiB） | 268,435,456 B；146 个 block × 16 token ≈ 2336 token |
| 合计 | ≈ 1.45 GiB |
| OpenCL 报告全局显存 | ≈ 16.4 GiB（集显共享内存，实际可用受系统/驱动限制） |

2048 token 的 KV 约 `2048 × 112 KiB ≈ 224 MiB`，256 MiB 预算留有
约 288 token 余量。服务启动后 `/health` 显示
`kv_free_blocks=146`。

实际请求（OCL，`max_context_tokens=2048`）：

```text
=== short: content_tokens=49 status=200 elapsed=1.00s ===
content: 'OK.'
finish: stop usage: {'prompt_tokens': 67, 'completion_tokens': 2, 'total_tokens': 69}

=== medium_800: content_tokens=817 status=200 elapsed=12.26s ===
content: 'OK.'
finish: stop usage: {'prompt_tokens': 835, 'completion_tokens': 2, 'total_tokens': 837}

=== near_1800: content_tokens=1801 status=200 elapsed=37.66s ===
content: 'OK.'
finish: stop usage: {'prompt_tokens': 1819, 'completion_tokens': 2, 'total_tokens': 1821}

=== over_2200: content_tokens=2209 status=400 elapsed=0.02s ===
{"error": {"message": "context_length_exceeded: prompt exceeds max_context_tokens=2048", "type": "invalid_request_error", "param": null, "code": "context_length_exceeded"}}
```

结论：

- 835 与 1819 token prompt 均成功，说明 chunked prefill + KV 长上下文工作；
- 2209 token 单条消息快速拒绝，未进入 BPE/Engine；
- 长 prompt 的 prefill 时间为 12 s / 38 s（本机 Intel Arc 130T），明显高于短 prompt，属预期；
- 2048 上下文下生成上限由 `max_tokens` 与剩余预算共同决定。

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
- Prepare 已增加字节级预判，但接近预判上限的对抗性长文本仍可能消耗较多
  CPU（BPE 未做预算内提前终止），未做 DoS 基准。
- 消息 content 中的特殊 token 字面串（如 `<|im_end|>`）会被编码为真实
  控制 token，存在 prompt injection 风险；基础版默认仅本机，公网场景需
  在接入层转义/拒绝。
- Android 真机未验证；Winsock/POSIX 分支需在 NDK 环境补充构建与运行。
- FakeEngine 不覆盖真实 Engine 的调度/数值细节；OCL 冒烟作为补充。
- 以下路径未做自动化用例：stop_token_ids 透传、max_connections 503、
  超长头部 413/431、流式开始后错误、Expect:100-continue 的端到端验证；
  其中部分已在代码中修复，建议后续补测。

## 8. 结论

Session 模块达到基础版目标：

- P0 端点全部可用，OpenAI 基础客户端可完成非流式与流式对话；
- 多轮会话、隔离、删除、stop、忙、超时、错误映射均通过；
- 真实 Qwen3-0.6B OCL 冒烟通过，文本交互可用；
- 四件套（设计文档、代码、测试结果、skill）齐全。

## 9. Review 修复记录

本轮独立代码 review 发现的问题及修复：

| 问题 | 修复 | 验证 |
| --- | --- | --- |
| WSAStartup/WSACleanup 计数失衡、Stop 不幂等 | 增加 stopped_/wsa_started_，Stop 幂等且只 Cleanup 一次 | 10 轮重复测试、OCL 冒烟 |
| 生成 token 非法 UTF-8 导致整条请求失败 | StreamDecoder 对非法字节输出 U+FFFD 并继续 | tokenizer_test token id 94 用例 |
| 非首条 system 角色名错误 | Chat 模板保留 `message.role` | 新增 golden 用例，与 HF 一致 |
| Stop 后无法重新 Init/Start | PendingRegistry::Reset，Server::Init 调用 | TestRestart |
| kIllegal 终态重复解码上一 token | 先处理 kIllegal，再解码 token | 代码审查，FakeEngine 无直接用例 |
| Session Prune/Delete 与 Chat 竞态 | Chat 取得 session 锁后回查注册表 | 代码审查；10 轮并发测试 |
| usage.completion_tokens 包含被跳过的 EOS | 使用实际送入 decoder 的 decoded_tokens | TestStopTokenUsage；真实 OCL completion=7 |
| 超长文本 Prepare CPU 风险 | 编码前字节预判；会话历史裁剪 | TestBodyLimit、TestHistoryCap |
| SSE 已开始后错误无终止标记 | 错误事件后发送 `[DONE]` | 代码审查 |
| max_completion_tokens:null 吞掉 max_tokens | null 时回退 max_tokens | TestBasicChat |
| 头部 16 KiB 上限存在窗口 | 找到 `

` 后校验长度 | 代码审查 |
| 不支持 Expect: 100-continue | 识别并回写 100 Continue | 代码审查 |
| prompt injection（content 含特殊 token） | 基础版默认本机；文档标注接入层需转义/拒绝 | 风险记录 |

