# Session 模块 Skill

## 适用场景

- 在 Runtime Engine + Tokenizer 之上启动一个本机 HTTP 服务。
- 需要 OpenAI 兼容的 `/v1/chat/completions`（非流式/SSE 流式）。
- 需要服务端多轮会话：创建、查询、删除、会话内对话与隔离。
- 需要 OpenAPI 3.0 描述、健康检查、统一错误格式。
- 需要把文本输入转换为文本输出，进行简单文字交互。

不适用：PrefixCache、多模型、多租户、鉴权/TLS、tools/多模态/logprobs、
生产级公网部署。

## 输入

- 服务配置 JSON：见 `Service/session/config.example.json`。
- 模型目录（Tokenizer 资产）。
- 模型包目录（`model.mgf` + `weights.bin`）。
- 构建产物：`Service/build/session/mobilenet_service.exe`。
- 聊天输入：UTF-8 文本。
- 测试可注入 `IEngine` 的 FakeEngine，避免加载 1.2 GB 模型。

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
2. 启动（OCL 真实模型）：
   ```bash
   ./Service/build/session/mobilenet_service.exe \
     --config Service/session/config.example.json
   ```
3. 健康检查：
   ```bash
   curl http://127.0.0.1:8080/health
   ```
4. 非流式聊天：
   ```bash
   curl -s http://127.0.0.1:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"qwen3_0_6b","messages":[{"role":"user","content":"Hello"}]}'
   ```
5. 流式聊天：
   ```bash
   curl -N http://127.0.0.1:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"qwen3_0_6b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
   ```
6. 原生会话：
   ```bash
   # 创建
   curl -s -X POST http://127.0.0.1:8080/v1/sessions \
     -H 'Content-Type: application/json' \
     -d '{"system_prompt":"You are helpful."}'
   # 对话
   curl -s -X POST http://127.0.0.1:8080/v1/sessions/<id>/chat \
     -H 'Content-Type: application/json' \
     -d '{"message":"Hello"}'
   # 查询/删除
   curl -s http://127.0.0.1:8080/v1/sessions/<id>
   curl -s -X DELETE http://127.0.0.1:8080/v1/sessions/<id>
   ```
7. 交互式客户端（另一个终端）：
   ```bash
   chcp 65001
   python Service/session/tools/chat_cli.py --base-url http://127.0.0.1:18080
   ```
   输入文字回车即得回复；服务端保存多轮历史。`/reset` 新会话，
   `/stream` 切换流式，`/exit` 退出；`--no-stream` 关闭流式，
   `--max-tokens` 控制单轮长度。
8. 测试：
   ```bash
   ctest --test-dir Service/build --output-on-failure
   ```
   `session_test` 使用 FakeEngine + 真实 Qwen3 Tokenizer，覆盖非流式、
   流式、多轮、隔离、stop、错误、会话忙、超时与 OpenAPI。

## 输出

- 非流式：OpenAI `chat.completion` JSON（含 `usage`）。
- 流式：SSE `data: {...}` chunk，终态 `finish_reason`，最后
  `data: [DONE]`。
- 原生会话：`session_id`、`created_at`、`last_active`、`message_count`、
  `state`。
- 错误：`{"error":{"message","type","param","code"}}`。
- `/openapi.json`：P0 端点 OpenAPI 3.0.3 文档。

## 常见陷阱

1. **Engine 回调不能阻塞/重入**：`ResponseCallback` 在推理线程触发；
   Session 只把事件入队。不要在回调里调用 `AddRequest/CancelRequest`。
2. **request_id 先注册再提交**：Session 预分配 `Request.request_id` 并
   注册 PendingRequest，避免推理线程先回调、注册后到的竞态。
3. **上下文预算**：`engine.max_num_tokens` 是单步图预算（当前 256），
   不要改大；单请求总上下文由 `generation.max_context_tokens` 控制。
   默认 256，演示配置 2048。prompt + max_tokens 超限会丢弃旧消息，仍
   无法满足则返回 `context_length_exceeded`。
   2048 token 的 KV 约 224 MiB，需同步设置 `engine.kv_cache_bytes`
   （演示配置 256 MiB；Qwen3-0.6B 约 112 KiB/token）。
4. **采样参数暂不生效**：Engine 阶段一/二为 argmax；temperature/top_p/
   top_k/seed 接受并校验，但不会改变输出。不要把它当作采样服务。
5. **多轮无 PrefixCache**：每轮重渲染完整历史并重新 Prefill，延迟随历史
   增长；PrefixCache 是后续模块。
6. **SSE 使用 chunked + Connection: close**：客户端需支持 chunked；
   服务端只支持 Content-Length 请求体，不支持 chunked 请求体。
7. **HTTP 基础版限制**：无 TLS/鉴权；默认绑定 127.0.0.1；单请求单连接；
   无 keep-alive；生产部署需自行加反向代理。
8. **会话忙**：同一会话串行；并发请求返回 409 `session_busy`。不同会话
   可并行，但 Engine 默认并发 1，实际排队。
9. **stop 字符串**：在解码文本上检测，跨 token 通过尾部缓冲处理；
   最多 4 个、每个 256 字节。stop token ids 透传 Engine。
9.1 **会话历史上限**：每轮成功后按 `max_history_messages`（默认 64，
   最小 4）裁剪，保留 system 与最近消息；prompt 预算由
   `generation.max_context_tokens` 控制（默认 256，演示 2048）。
10. **测试不要加载真实模型**：`session_test` 用 FakeEngine；真实 OCL
    冒烟测试耗时数十秒，建议单独手动执行并记录。
11. **错误映射依赖消息前缀**：`context_length_exceeded`、`session_busy`、
    `session_not_found` 等由 Status 消息前缀映射为 HTTP 错误码；修改
    这些字符串时必须同步 HTTP 映射与文档。
12. **prompt injection**：消息 content 中的 `<|im_start|>`、`<|im_end|>`
    等字面串会被 Tokenizer 当作 added token；基础版默认本机，公网或
    不受信输入必须在接入层转义/拒绝。

## 验证基线

- FakeEngine 集成测试：health/models、非流式/流式、max_tokens=length、
  stop 截断、多轮会话、会话隔离、删除、错误路径、忙、超时、OpenAPI。
- Tokenizer 真实资产：Qwen3-0.6B，82 条 Encode 金标准。
- 真实 OCL 冒烟：`qwen3_0_6b_kv` + 短 prompt；结果与耗时记录在
  `Service/session/tests/test_results.md`。
