# Session 模块设计文档

> 模块目录：`Service/session/`
> 文档版本：v1.0（2026-11-21）
> 对应架构文档：`MobileNet_design.md` §4.5.3、FR-2.3、TC-SVC-01
> 首个验证模型：Qwen3-0.6B

## 1. 背景与目标

### 1.1 背景

Runtime Engine 已完成 token 级推理：`AddRequest` 入队、推理线程连续调度、
`ResponseCallback` 逐步返回 token。Tokenizer 模块提供文本与 token 的双向
转换。阶段二需要在两者之上提供服务端能力：HTTP 接口、OpenAI 兼容协议、
会话管理、多轮对话、流式输出、取消与错误映射。

当前 `Service/session/` 为空目录。本模块基础版只做单进程、单模型、单 Engine
实例的简单文字交互：输入文本，返回文本（JSON 或 SSE 流）。

### 1.2 目标

1. 在 `Service/session/` 提供 C++17 的进程内服务端。
2. 提供 OpenAI 兼容的基础 HTTP 接口与原生 Session 接口。
3. 支持无状态聊天与有状态多轮会话，会话之间隔离。
4. 支持非流式 JSON 与流式 SSE 响应。
5. 支持取消、超时、stop 字符串、上下文预算与统一错误格式。
6. 提供 `/openapi.json`（OpenAPI 3.0.3）描述已实现接口。
7. 满足 memory.md 的编码规范、中文注释、四件套与六段式文档要求。

### 1.3 非目标

- 不做 PrefixCache（阶段二 T2-2）。
- 不做 Tokenizer/Detokenizer 与 GPU 推理的流水重叠（FR-2.4）。
- 不做多模型、多租户、鉴权、TLS、分布式部署。
- 不做 tools/function calling、多模态、logprobs、完整采样；当前 Engine
  阶段一/二为 argmax。
- 不做生产级 HTTP 服务器；基础版面向本机/局域网联调。
- 不实现 HTTP/2、WebSocket、分块请求体。

### 1.4 与整体架构的关系

```text
HTTP/OpenAI client
        │
        ▼
Service/session（HttpServer → Router → OpenAI/原生 handler）
        │
        ├── SessionManager（会话历史、隔离、截断）
        ├── ChatService（模板 → Encode → Engine → Decode）
        ├── Tokenizer（Service/tokenizer）
        └── IEngine ── RuntimeInferenceEngine ── Runtime Engine
```

`IEngine` 是 Session 与 Runtime 之间的测试接缝：生产实现包装
`runtime::Engine`，测试实现可注入确定性的 FakeEngine，避免每次协议测试都
加载 1.2 GB 模型并执行耗时的 CPU 参考推理。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求描述 | 验收要点 | 关联用例 |
| --- | --- | --- | --- |
| SVC-F-01 | 启动加载配置、Tokenizer、Engine；支持优雅停止 | 加载失败有明确错误；停止顺序正确、无泄漏 | TC-SVC-01 |
| SVC-F-02 | `GET /health`、`GET /v1/models` | 返回 Engine 状态与统计；OpenAI models 格式 | TC-SVC-02 |
| SVC-F-03 | `POST /v1/chat/completions` 非流式 | messages → 模板 → Encode → Engine → Decode；符合 OpenAI chat.completion | TC-SVC-03 |
| SVC-F-04 | `POST /v1/chat/completions` 流式 SSE | 首 chunk 角色、内容 delta、终 chunk finish_reason + usage、`[DONE]` | TC-SVC-04 |
| SVC-F-05 | 原生会话：创建/查询/删除/对话 | session_id 稳定；历史保存；会话隔离 | TC-SVC-05 |
| SVC-F-06 | 多轮对话与上下文预算 | 每轮完整历史重新模板化；超限按策略截断或 400 | TC-SVC-05、TC-SVC-08 |
| SVC-F-07 | 取消与停止 | 客户端断开/超时/stop 命中 → `CancelRequest`；服务保持可用 | TC-SVC-06 |
| SVC-F-08 | OpenAI 错误格式 | 统一 `{"error":{message,type,code}}` 与 HTTP 状态 | TC-SVC-07 |
| SVC-F-09 | `GET /openapi.json` | 合法 OpenAPI 3.0.3，覆盖 P0 端点 | TC-SVC-09 |
| SVC-F-10 | 配置与 CLI | JSON 配置 + `--config`；端口、路径、后端、上限可配 | TC-SVC-01 |
| SVC-F-11 | 请求参数校验 | 拒绝不支持参数（tools/n>1/多模态等），接受但记录无效采样参数 | TC-SVC-07 |
| SVC-F-12 | stop 字符串与 stop token | 文本级 stop 检测与截断；stop token 透传 Engine | TC-SVC-06 |

### 2.2 非功能需求

| 编号 | 类别 | 要求 | 验证方式 |
| --- | --- | --- | --- |
| SVC-NF-01 | 并发 | 默认 `max_concurrent_requests=1`（匹配 `[1,256]` 模型包），可配置；HTTP 支持多连接 | TC-SVC-10 |
| SVC-NF-02 | 回调安全 | Engine 回调只入队，不阻塞、不重入 Engine；队列有界 | TC-SVC-04、TC-SVC-10 |
| SVC-NF-03 | 资源上限 | 默认体 1 MiB、连接 32、请求超时 120 s、会话 64、历史 64 条 | TC-SVC-07 |
| SVC-NF-04 | 可移植性 | Windows（Winsock）/ Linux / Android（POSIX）；离线运行 | 构建检查 |
| SVC-NF-05 | 可观测性 | 请求日志含 request id、session id、token 数、耗时；`/health` 统计 | TC-SVC-02 |
| SVC-NF-06 | 工程规范 | Google C++ Style；模块/类/函数级中文注释 | 代码审查 |
| SVC-NF-07 | 性能 | 记录首 token 延迟与解码吞吐基线；不虚构成通过指标 | TC-SVC-11 |

### 2.3 约束条件

- Engine 的 `Response` 每步回调；`Response`/`Sequence` 为值快照，
  `logits` 仅在回调期间有效；Session 不保存 logits。
- Engine 回调在推理线程同步触发；严禁在回调中调用 Engine。
- 当前模型包图输入 `[1,256]`，单请求 prompt + 生成 token 上限为 256；
  Session 配置必须与模型包一致。
- 当前 Engine 采样为 argmax；`temperature/top_p/top_k/seed` 接受但暂不生效。
- 模型资产位于 `Models/`，不提交 Git；测试通过路径参数或环境变量定位。
- 所有 C++ 整数类型使用 `Utils/cpp/types.h` 别名；不使用异常穿越模块边界。

### 2.4 验收标准

- AC-SVC-1：P0 端点全部可用，OpenAI 基础客户端可完成一次非流式与流式对话。
- AC-SVC-2：两轮会话历史正确；两个会话互不串扰。
- AC-SVC-3：客户端断开、超时、stop 命中均能取消 Engine 请求，服务继续可用。
- AC-SVC-4：错误响应符合 OpenAI 错误格式。
- AC-SVC-5：FakeEngine 集成测试全部通过；OCL 真实模型冒烟测试（可选）通过。
- AC-SVC-6：四件套齐全。

## 3. 概要设计

### 3.1 分层与数据流

```text
Client
  │ HTTP/1.1
  ▼
HttpServer
  │ 线程/连接、解析、限流、超时
  ▼
HttpRouter
  ├── /health、/v1/models、/openapi.json
  ├── /v1/chat/completions（无状态）
  └── /v1/sessions*（有状态）
        │
        ▼
OpenAI API 层（请求解析、响应/错误/SSE 序列化）
        │
        ▼
ChatService（prompt 构造、预算、Engine 请求、解码、stop）
        │                         │
        ▼                         ▼
SessionManager             PendingRegistry
（历史/隔离/截断）          （request_id → 事件队列）
                                  │
                                  ▼
                             IEngine
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          RuntimeInferenceEngine           FakeEngine（测试）
                    │
                    ▼
              runtime::Engine（threaded=true）
```

### 3.2 模块划分

| 文件 | 职责 | 边界 |
| --- | --- | --- |
| `server.h/.cc` | 组装配置、Tokenizer、Engine、HTTP 服务；生命周期 | 不实现协议细节 |
| `server_config.h/.cc` | JSON 配置读取与校验 | 不加载模型 |
| `http_server.h/.cc` | socket 抽象、HTTP/1.1 解析、响应、SSE chunked | 不理解业务 JSON |
| `http_router.h/.cc` | 方法 + 路径匹配、路径参数 | 不解析业务体 |
| `json_writer.h/.cc` | JSON 序列化与转义 | 不解析 |
| `openai_api.h/.cc` | OpenAI 请求解析、响应/错误/SSE 序列化 | 不调用 Engine |
| `chat_service.h/.cc` | 模板、编码、预算、请求、解码、stop、usage | 不管理会话生命周期 |
| `session.h/.cc` | 会话消息历史、状态、默认采样 | 不直接调用 Engine |
| `session_manager.h/.cc` | 会话注册表、加锁、隔离、截断入口 | 不实现 HTTP |
| `pending_registry.h/.cc` | request_id 映射、回调入队、取消状态 | 不解析 token |
| `engine_client.h/.cc` | `IEngine` 与 `RuntimeInferenceEngine` | 不接触文本 |
| `openapi_spec.h/.cc` | OpenAPI 3.0.3 静态描述 | 不动态反射 |
| `main.cc` | CLI、信号处理 | 不实现业务 |

### 3.3 接口范围

P0（基础版必做）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 存活、Engine 状态、累计统计 |
| GET | `/v1/models` | OpenAI 模型列表 |
| POST | `/v1/chat/completions` | OpenAI 聊天，支持 `stream` |
| POST | `/v1/sessions` | 创建会话 |
| GET | `/v1/sessions/{id}` | 会话信息 |
| DELETE | `/v1/sessions/{id}` | 删除会话 |
| POST | `/v1/sessions/{id}/chat` | 有状态多轮对话，支持 `stream` |
| GET | `/openapi.json` | OpenAPI 3.0.3 |

P1（可选，本期不实现）：

- `POST /v1/completions`
- `POST /v1/tokenize`、`POST /v1/detokenize`
- `POST /v1/sessions/{id}/cancel`
- `GET /v1/sessions`

### 3.4 线程模型

| 线程 | 数量 | 职责 |
| --- | --- | --- |
| HTTP accept 线程 | 1 | `accept` 新连接，交给连接线程 |
| HTTP 连接线程 | ≤ `max_connections` | 读取请求、路由、调用 handler、写响应/SSE |
| Engine 推理线程 | 1（`threaded=true`） | Step 循环，触发 `ResponseCallback` |
| 客户端业务线程 | 即连接线程 | 等待 PendingRegistry 事件、解码、写回 |

同步规则：

- `PendingRegistry` 用互斥锁保护 map；每个请求状态有独立互斥锁与条件变量。
- Engine 回调只做短临界区入队；不得等待连接线程。
- 同一会话串行：会话互斥锁 + busy 状态；不同会话可并行提交。
- HTTP 服务器停止时：停止 accept、取消在途请求、关闭连接、join 线程。

### 3.5 技术选型与取舍

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| HTTP | 自研最小 HTTP/1.1 | 零第三方；只支持 Content-Length 请求与 chunked/close 响应 |
| JSON 输出 | 模块内 `JsonWriter` | 现有 Runtime JSON 只解析 |
| 会话历史 | 保存 messages 并每轮重渲染 | 模板语义正确；PrefixCache 后续再做 |
| Engine 接入 | `IEngine` 抽象 + Runtime 适配器 | 协议测试可用 FakeEngine，避免加载大模型 |
| 流式 | SSE + chunked | 与 OpenAI 客户端兼容 |
| 并发默认 | 1 | 当前模型包 `[1,256]`、`max_batch_size=1` |
| 思考模式 | 默认关闭 | 简单文字交互更快；请求可覆盖 |

## 4. 详细设计

### 4.1 配置

```cpp
struct ServerConfig {
  // HTTP。
  std::string host = "127.0.0.1";
  U16 port = 8080;
  U32 max_body_bytes = 1u << 20;
  U32 max_connections = 32;
  U32 request_timeout_ms = 120000;
  U32 read_timeout_ms = 30000;

  // 模型与 Tokenizer。
  std::string model_name = "qwen3_0_6b";
  std::string model_dir;       // tokenizer 资产目录
  std::string package_dir;     // model.mgf 模型包目录

  // Engine。
  std::string backend = "ocl";
  I32 device_id = 0;
  U64 kv_cache_bytes = 0;  // OCL 需要显式配置，如 64 MiB
  I32 max_num_tokens = 256;
  I32 max_concurrent_requests = 1;
  Bool verify_weights_sha256 = true;
  Bool recompute_attention = false;

  // 生成与会话。
  I32 default_max_tokens = 128;
  ThinkingMode thinking = ThinkingMode::kDisabled;
  U32 max_sessions = 64;
  U32 session_idle_timeout_s = 3600;
  U32 max_history_messages = 64;  // 最小 4
  std::string log_level = "info";
};
```

配置文件为 JSON，字段与结构体一一对应；相对路径按进程当前目录解析，
测试中优先使用绝对路径。加载时校验：

- `port` 允许 0（测试用临时端口），否则 1–65535；
- `model_dir`、`package_dir` 非空；
- `max_num_tokens > 0`、`default_max_tokens > 0`；
- `max_connections`、`max_body_bytes`、`request_timeout_ms` 非零；
- 未知字段警告但不失败。

### 4.2 HTTP 服务器

#### 4.2.1 请求解析

- 只支持 `GET`、`POST`、`DELETE`。
- 请求行：`METHOD SP PATH SP HTTP/1.1`；路径不含查询串时 `query` 为空。
- 头部大小上限 16 KiB；头部名按小写匹配。
- 请求体只支持 `Content-Length`；缺失视为空体；超过 `max_body_bytes`
  返回 413；`Transfer-Encoding: chunked` 返回 400。
- 支持 `Expect: 100-continue`：先回写 `HTTP/1.1 100 Continue` 再读 body。
- 不支持 keep-alive：响应统一 `Connection: close`，一个连接处理一个请求。
- 读取/写入超时通过 socket 选项设置；超时返回 408/关闭连接。

#### 4.2.2 响应

普通响应：

```text
HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
Content-Length: <N>
Connection: close

<JSON>
```

SSE 响应：

```text
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: close
Transfer-Encoding: chunked

<chunk：data: {...}\n\n>
...
<chunk：data: [DONE]\n\n>
0\r\n\r\n
```

- 每个 SSE 事件单独一个 chunk；`send` 失败视为客户端断开。
- 连接线程在 SSE 期间阻塞；受 `max_connections` 限制。
- POSIX 忽略 `SIGPIPE` 或使用 `MSG_NOSIGNAL`；Windows 使用 `send` 错误码。

#### 4.2.3 路由

- 精确匹配与路径参数：`/v1/sessions/:id`、`/v1/sessions/:id/chat`。
- 路径参数按 `/` 分段；不实现通配符与正则路由。
- 未匹配返回 404；方法不匹配返回 405。

### 4.3 JSON 序列化

`JsonWriter` 提供流式构建：

```cpp
JsonWriter writer;
writer.BeginObject();
writer.Key("id"); writer.String(id);
writer.Key("choices"); writer.BeginArray();
...
writer.EndArray(); writer.EndObject();
```

字符串转义：

- `"` → `\"`，`\` → `\\`；
- `\b \f \n \r \t` 使用短转义；
- 其他 U+0000–U+001F 使用 `\u00XX`；
- 非 ASCII UTF-8 原样输出（输入必须为合法 UTF-8）。

浮点数使用 `%.6g` 并校验有限值；整数使用 `I64`；布尔/空值直接输出。
序列化结果必须是合法 JSON，由单元测试用 `ParseJson` 反解析验证。

### 4.4 Engine 抽象

```cpp
class IEngine {
 public:
  virtual ~IEngine() = default;
  virtual Status Load(const RuntimeEngineConfig& config) = 0;
  virtual Status Start() = 0;
  virtual Status AddRequest(const runtime::Request& request,
                            U64* request_id) = 0;
  virtual Status CancelRequest(U64 request_id) = 0;
  virtual void SetResponseCallback(runtime::ResponseCallback callback) = 0;
  virtual runtime::EngineStats stats() const = 0;
  virtual runtime::EngineState state() const = 0;
  virtual Status Destroy() = 0;
};

class RuntimeInferenceEngine final : public IEngine {
  // 持有 runtime::Engine；Load 时设置 runtime.threaded=true。
};
```

- 生产路径：`Server` 默认创建 `RuntimeInferenceEngine`。
- 测试路径：`Server::Init(config, std::unique_ptr<IEngine>)` 注入 FakeEngine。
- `RuntimeInferenceEngine::Load` 使用 `package_dir` 与 `RuntimeConfig`；
  `runtime.threaded=true`；`max_num_tokens`、并发数、后端来自配置。
- `SetResponseCallback` 必须在 `Start` 前调用；Server 在 Load 前后均可设置。

### 4.5 PendingRegistry

```cpp
struct EngineEvent {
  I64 token_id = -1;
  runtime::RequestStatus status = runtime::RequestStatus::kInvalid;
  I32 generated_count = 0;
  I32 prompt_count = 0;
};

class PendingRegistry {
 public:
  U64 AllocateRequestId();
  Status Register(U64 request_id, std::shared_ptr<PendingRequest>* state);
  void Remove(U64 request_id);
  void OnResponse(const runtime::Response& response);
  void FailAll();
 private:
  std::mutex mutex_;
  std::unordered_map<U64, std::shared_ptr<PendingRequest>> pending_;
  std::atomic<U64> next_id_{1};
};

struct PendingRequest {
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<EngineEvent> events;
  Bool overflow = false;
  Bool terminal = false;
  static constexpr Usize kMaxEvents = 4096;
};
```

- 请求 ID 由 Session 分配并先注册，再 `AddRequest`，避免推理线程先回调的竞态。
- `OnResponse`：
  - 查找 `request_id`；不存在则忽略（可能是已取消请求的迟到回调）。
  - 拷贝 `current_token.token_id`、`status`、`generated_count`、`prompt_count`；
  - 入队；若队列达到上限，置 `overflow=true`，丢弃事件；
  - 终态置 `terminal=true`；
  - `cv.notify_one()`。
- 业务线程检查 `overflow` 后调用 `CancelRequest` 并返回 503/500。
- `FailAll` 在停止时通知所有等待者并清理。

### 4.6 ChatService

#### 4.6.1 Prompt 构造与预算

输入：`messages`、请求参数、会话默认采样。

1. 校验角色与内容（OpenAI 层已完成大部分）；按消息总字节数做快速
   上限检查（`max_num_tokens * 64`），明显超限时直接返回
   `context_length_exceeded`，避免对超长文本执行完整 BPE。接近上限的
   文本仍会完整编码；预算内提前终止是后续优化项。
2. `Tokenizer::ApplyChatTemplate` 生成 prompt 文本。
3. `Tokenizer::Encode` 生成 `prompt_ids`。
4. 计算 `max_tokens`：
   - `requested = request.max_tokens > 0 ? request.max_tokens
                                        : config.default_max_tokens`；
   - `budget = config.max_num_tokens - prompt_ids.size()`；
   - 若 `budget <= 0` → `context_length_exceeded`；
   - `max_tokens = min(requested, budget)`。
5. 截断策略（无状态与有状态共用）：
   - 保留首条 system（若有）与最后一条 user；
   - 从旧到新丢弃其余消息，直到 prompt 预算满足 `max_tokens >= 1`；
   - 若仍不满足 → 400 `context_length_exceeded`。
6. `sample_param`：
   - `is_valid=true`；
   - `max_tokens`；
   - `min_tokens=0`；
   - `stop_token_ids` 来自请求扩展；
   - `ignore_eos=false`；
   - `temperature/top_p/top_k/seed` 仅记录日志，不写入阶段一/二有效字段。

#### 4.6.2 请求与事件循环

```text
register(request_id)
  → AddRequest
  → loop:
      wait_for(event, min(deadline, tick))
      if overflow → cancel → error
      if token_id >= 0:
         if terminal 且 token_id ∈ EOS/stop ids → 不送入解码
         else StreamDecoder.Push → stop 检测 → sink
      if terminal:
         Finish decoder → finish_reason
         usage = prompt_count / decoded_count
         返回
      if deadline 到期 → CancelRequest → timeout
```

- 终态 `kStopped`：`generated_count >= max_tokens` → `length`，
  否则 `stop`。
- 终态 `kIllegal`：返回 500 / `engine_error`。
- 客户端断开：sink 返回 false → `CancelRequest` → 结束，不再写响应。
- stop 字符串：见 §4.7。
- 非流式：相同循环，但不写 sink，把增量文本累积到 `ChatOutcome.text`。

#### 4.6.3 finish_reason 与 usage

| 条件 | finish_reason |
| --- | --- |
| EOS 或自定义 stop token | `stop` |
| stop 字符串命中 | `stop` |
| 达到 `max_tokens` | `length` |
| 客户端取消/断开 | 不返回（或 `stop`） |
| Engine 失败 | 返回错误，不返回正常 choice |
| 超时 | HTTP 504，不返回正常 choice |

usage：

- `prompt_tokens = prompt_ids.size()`；
- `completion_tokens = 实际送入解码器的 token 数`（不含被跳过的
  EOS/stop token）；
- `total_tokens = prompt_tokens + completion_tokens`。

### 4.7 stop 处理

- stop token ids：透传到 `SampleParams.stop_token_ids`；终态 token 不送入
  解码器。
- stop 字符串：
  - 在解码后的文本上匹配；
  - 非流式：取所有 stop 中最早出现的位置，截断；
  - 流式：维护“未发送尾部”缓冲，长度上限为最长 stop 字节数 - 1；
    - 每次追加增量文本后先搜索 stop；
    - 命中：发送 stop 之前的文本，丢弃 stop 及其后，取消请求；
    - 未命中：发送尾部缓冲之前的安全前缀（按 UTF-8 码点边界），保留尾部。
  - stop 字符串数量上限 4，单个长度上限 256 字节。
- 命中 stop 后调用 `CancelRequest`；即使 Engine 已自然结束，调用返回
  `kRequestNotFound` 也视为成功。

### 4.8 会话管理

```cpp
class Session {
 public:
  std::string id;
  std::vector<ChatMessage> messages;
  std::chrono::system_clock::time_point created_at;
  std::chrono::steady_clock::time_point last_active;
  SessionState state;  // kIdle / kBusy / kClosed
  std::mutex mutex;
};

class SessionManager {
 public:
  Status Create(const CreateSessionOptions&, std::string* id);
  std::shared_ptr<Session> Find(const std::string& id);
  Status Delete(const std::string& id);
  Status Chat(const std::string& id, const ChatRequest&,
              StreamSink* sink, ChatOutcome* outcome);
};
```

- 会话 ID：32 位十六进制随机串；创建时保证唯一。
- 创建参数：可选 `system_prompt`、默认采样参数。
- 会话状态：
  - `kIdle` → `kBusy`（进入 Chat）→ `kIdle`（成功/失败）；
  - 删除时置 `kClosed` 并从注册表移除；在途请求取消。
- 同一会话并发 Chat：busy 时返回 409 `session_busy`。
- 会话隔离：每个会话独立 `messages`；共享同一个 Engine。
- 历史更新：
  - 请求前追加 user 消息；
  - 成功后追加 assistant 消息；
  - 失败/取消则回滚 user 消息，保持历史一致。
- 空闲淘汰：后台清扫线程（或每次访问时惰性清理）删除超过
  `session_idle_timeout_s` 且非 busy 的会话；基础版采用惰性清理。
- 历史上限：每轮成功后按 `max_history_messages` 裁剪，保留首条 system
  与最近消息，避免长会话内存无界增长；详细 prompt 截断仍由 §4.6.1 负责。
- 上下文预算：每轮构造 prompt 时按 §4.6.1 截断；成功后按
  `max_history_messages` 裁剪存储历史。`GET /v1/sessions/{id}` 返回当前
  保留的消息条数。

### 4.9 OpenAI 协议映射

#### 4.9.1 Chat Completions 请求

| 字段 | 类型 | 基础版行为 |
| --- | --- | --- |
| `model` | string | 必须等于配置模型名；否则 404 `model_not_found` |
| `messages` | array | 必填；role ∈ system/user/assistant；content 为 string |
| `stream` | bool | 默认 false |
| `stream_options.include_usage` | bool | true 时在终 chunk 附加 usage |
| `max_tokens` / `max_completion_tokens` | int | 映射；上限受上下文预算 |
| `temperature` | number | 校验 [0,2]；当前不生效，记录日志 |
| `top_p` | number | 校验 [0,1]；当前不生效 |
| `top_k` | int | 扩展；当前不生效 |
| `seed` | int | 记录；当前不生效 |
| `frequency_penalty` / `presence_penalty` | number | 校验范围；当前不生效 |
| `stop` | string/array | 最多 4 个；文本级检测 |
| `n` | int | 仅 1；其他返回 400 |
| `user` | string | 忽略 |
| `tools` / `tool_choice` / `response_format` / `logprobs` / `logit_bias` | - | 400 `unsupported_parameter` |
| `chat_template_kwargs.enable_thinking` | bool | 覆盖默认思考模式 |
| `stop_token_ids` | array | 扩展；透传 Engine |

#### 4.9.2 非流式响应

```json
{
  "id": "chatcmpl-<hex>",
  "object": "chat.completion",
  "created": 1763700000,
  "model": "qwen3_0_6b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
}
```

#### 4.9.3 流式响应

每个事件为 `data: <json>\n\n`：

1. 首 chunk：`delta: {"role":"assistant"}`，`finish_reason: null`。
2. 内容 chunk：`delta: {"content":"<增量>"}`。
3. 终 chunk：`delta: {}`，`finish_reason` 为 `stop`/`length`；
   若请求 `include_usage=true`，附加 `usage`。
4. `data: [DONE]`。

错误在流开始前以普通 JSON 返回；流开始后发生错误，发送
`data: {"error":{...}}`、`data: [DONE]`，然后结束 SSE。

#### 4.9.4 原生会话接口

- `POST /v1/sessions`
  ```json
  {"session_id": "<可选>", "system_prompt": "<可选>", "defaults": {...}}
  ```
  响应：`{"session_id":"...","created_at":...,"object":"session"}`
- `GET /v1/sessions/{id}`
  响应：`{"session_id":"...","created_at":...,"last_active":...,
  "message_count":N,"state":"idle|busy","object":"session"}`
- `DELETE /v1/sessions/{id}`
  响应：`{"session_id":"...","deleted":true}`
- `POST /v1/sessions/{id}/chat`
  ```json
  {"message":"...","stream":false,"max_tokens":64,"stop":["..."]}
  ```
  响应：OpenAI chat.completion 结构 + `session_id` 字段；
  流式与 `/v1/chat/completions` 相同的 SSE 结构，chunk 含 `session_id`。

### 4.10 错误映射

统一错误体：

```json
{"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
```

| 场景 | HTTP | type | code |
| --- | --- | --- | --- |
| JSON 非法/字段类型错误 | 400 | `invalid_request_error` | `invalid_json` / `invalid_parameter` |
| 不支持参数/取值 | 400 | `invalid_request_error` | `unsupported_parameter` |
| 上下文超限 | 400 | `invalid_request_error` | `context_length_exceeded` |
| 模型名不匹配 | 404 | `invalid_request_error` | `model_not_found` |
| 会话不存在 | 404 | `invalid_request_error` | `session_not_found` |
| 会话忙 | 409 | `invalid_request_error` | `session_busy` |
| 会话已存在 | 409 | `invalid_request_error` | `session_exists` |
| 请求体过大 | 413 | `invalid_request_error` | `body_too_large` |
| 路由/方法不存在 | 404/405 | `invalid_request_error` | `not_found` / `method_not_allowed` |
| Engine 未运行/fatal | 503 | `server_error` | `engine_unavailable` |
| 请求超时 | 504 | `server_error` | `request_timeout` |
| 内部错误 | 500 | `server_error` | `internal_error` |

Engine 错误码映射：

- `kStateError`/`kDeviceError`/`kKernelError`/`kBufferError` →
  503 `engine_unavailable`；
- `kRequestNotFound` → 409/500（取决于上下文）；
- `kInvalidArgument` → 400；
- `kTimeout` → 504。

### 4.11 生命周期与资源释放

```text
Server::Init(config, engine?)
  → Tokenizer::Load
  → IEngine::SetResponseCallback(PendingRegistry::OnResponse)
  → IEngine::Load
  → IEngine::Start
  → HttpServer::Start
Server::Stop
  → HttpServer::Stop（停止 accept、关闭监听、取消在途）
  → SessionManager::Shutdown（取消所有 busy 会话）
  → PendingRegistry::FailAll
  → IEngine::Destroy
  → Tokenizer 释放
```

- `Stop` 幂等；析构时若未 Stop 自动执行。
- HTTP 连接线程持有 `Server`/`ChatService` 指针；`Stop` 先 join HTTP 线程，
  再销毁 Engine，避免悬空指针。
- `main` 捕获 `SIGINT`/`SIGTERM`（Windows Ctrl+C）后调用 `Stop`。
- 退出码：正常 0；配置/加载失败 1。

### 4.12 日志与可观测性

- 每请求日志：
  `[info] http method path status elapsed_ms request_id session_id prompt_tokens completion_tokens`。
- 启动日志：配置摘要、监听地址、模型名、后端。
- `/health`：

```json
{
  "status": "ok",
  "engine_state": "RUNNING",
  "model": "qwen3_0_6b",
  "uptime_s": 123,
  "stats": {"total_steps":..., "total_requests":..., "total_tokens":...,
            "running_requests":..., "waiting_requests":...}
}
```

- 错误日志包含错误码与上下文；不输出 token 内容（避免隐私泄漏）。

### 4.13 安全与限制

- 默认只绑定 `127.0.0.1`；绑定 `0.0.0.0` 时启动日志告警。
- 无鉴权、无 TLS；基础版仅用于本机/受信局域网。
- 对话内容中的 `<|im_start|>`、`<|im_end|>` 等字面串会被 Tokenizer
  识别为 added token，与 HF 行为一致；这些串可能被用于 prompt injection。
  公网或不受信输入场景必须在接入层转义/拒绝。
- 请求体、头部、连接数、会话数、stop 数量均有上限。
- 不访问网络；不执行用户提供的模板/代码。

## 5. 实施步骤

| 阶段 | 任务 | 交付 | 依赖 | 风险 |
| --- | --- | --- | --- | --- |
| S1 | 设计文档确认与落盘 | 本文档 | 用户确认 | 无 |
| S2 | CMake 骨架、配置、JsonWriter、socket 抽象 | `CMakeLists.txt`、`server_config.*`、`json_writer.*` | S1 | Windows/POSIX 差异 |
| S3 | HttpServer + Router + `/health` | `http_server.*`、`http_router.*` | S2 | HTTP 边界 |
| S4 | IEngine + Runtime 适配器 + `/v1/models` | `engine_client.*` | S3、Tokenizer T2–T5 | Engine 线程/回调 |
| S5 | ChatService 非流式 + OpenAI 协议 | `chat_service.*`、`openai_api.*` | S4 | 预算/解码 |
| S6 | SessionManager 多轮/隔离/截断 | `session.*`、`session_manager.*` | S5 | 并发状态 |
| S7 | SSE 流式、取消、stop 截断 | `http_server` SSE、`chat_service` | S6 | 背压/断开 |
| S8 | 错误映射、OpenAPI、CLI | `openapi_spec.*`、`main.cc` | S7 | 兼容性 |
| S9 | FakeEngine/OCL 测试、skill、测试结果 | `tests/`、`.skill/SKILL.md` | S8 | 大模型加载耗时 |

## 6. 测试与验证设计

### 6.1 测试策略

- 协议与状态机测试使用 FakeEngine，快速、确定性。
- Tokenizer 使用真实 Qwen3 资产。
- 真实 Engine 冒烟测试使用 OCL 模型包；无设备时跳过并记录。
- HTTP 客户端用测试内最小 socket 客户端或 Python 标准库。
- OpenAPI 用 JSON 解析器校验合法性。

### 6.2 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-21 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | zig clang wrapper（`Graph/tools/zig_cxx.cmd`） |
| C++ 标准 | C++17 |
| Python | 3.13.3（Tokenizer 金标准） |
| 模型资产 | `Models/Qwen3`（Tokenizer）、`Models/Qwen3/artifacts/qwen3_0_6b_kv`（OCL） |

### 6.3 用例设计

| 编号 | 用例 | 输入/动作 | 期望 |
| --- | --- | --- | --- |
| TC-SVC-01 | 启动/停止 | FakeEngine + 临时端口 | 启动成功；停止幂等；端口释放 |
| TC-SVC-02 | health/models | HTTP GET | 200；字段与配置一致 |
| TC-SVC-03 | 非流式 chat | system+user 消息 | 200；OpenAI 结构；usage 合理 |
| TC-SVC-04 | 流式 chat | `stream=true` | SSE 顺序正确；拼接内容等于非流式；`[DONE]` |
| TC-SVC-05 | 多轮会话 | 创建 + 两轮 chat | 历史增长；两会话隔离；消息回滚正确 |
| TC-SVC-06 | 取消/stop | 客户端断开；stop 字符串；stop token | Engine 请求被取消；输出截断；服务可用 |
| TC-SVC-07 | 错误路径 | 非法 JSON、未知会话、busy、超长、模型不匹配、不支持参数 | 对应 HTTP/错误码，不崩溃 |
| TC-SVC-08 | 上下文预算 | prompt 接近 256 | 按策略截断；无法满足时 400 |
| TC-SVC-09 | OpenAPI | GET /openapi.json | 合法 JSON；含 P0 路径 |
| TC-SVC-10 | 并发 | 多连接/多会话；FakeEngine 延迟 | 无串扰；busy 409；回调不阻塞 |
| TC-SVC-11 | 性能基线 | FakeEngine；OCL 可选 | 记录首 token 延迟、吞吐 |
| TC-SVC-12 | OCL 冒烟 | `qwen3_0_6b_kv` + 短 prompt | 返回可读文本；与 Tokenizer ids 一致；无设备时跳过 |

### 6.4 通过标准

- TC-SVC-01 至 TC-SVC-11 全部通过。
- TC-SVC-12 在无 OCL 设备时标记 SKIP；有设备时记录实际输出与耗时。
- 流式内容与非流式内容必须逐字节一致（无 stop 时）。
- 失败用例不得通过放宽断言规避。

### 6.5 测试命令

```bash
cd d:/Mobile_Net
cmake -S Service -B Service/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=d:/Mobile_Net/Graph/tools/zig_cxx.cmd
cmake --build Service/build
ctest --test-dir Service/build --output-on-failure
```

实际输出、耗时与结论写入 `Service/session/tests/test_results.md`。

### 6.6 实现备注（2026-11-21）

- HttpServer::Start 增加 `max_body_bytes` 参数，请求体超限返回 413。
- ChatService::Stream 支持可选 ChatOutcome，供 SessionManager 保存流式
  助手文本。
- FakeEngine 集成测试覆盖非流式/流式/多轮/隔离/stop/错误/忙/超时/
  多连接并发/客户端断开/OpenAPI；真实 OCL 单轮、流式与两轮会话冒烟结果
  见 `tests/test_results.md`。

### 6.7 未覆盖项与风险

- 无 TLS/鉴权：不适用于公网。
- Engine argmax：采样参数不生效；文档与日志明确标注。
- 单请求 256 token：多轮对话需要截断；提升上限需重新规划模型包。
- 无 PrefixCache：多轮重复 Prefill，延迟随历史增长。
- 无 Keep-Alive：每请求新建连接，性能有限；基础版可接受。
- Windows/Android socket 差异：需在 Android NDK 环境补充构建/运行验证。
- FakeEngine 不覆盖真实 Engine 的调度/数值细节；OCL 冒烟测试作为补充。

## 附录 A：参考资料

| 资料 | 用途 |
| --- | --- |
| `Reference/vLLM/vllm/vllm/entrypoints/openai/chat_completion/protocol.py` | OpenAI Chat Completions 字段与响应结构 |
| `Reference/SGLang/sglang/python/sglang/srt/entrypoints/openai/protocol.py` | 请求字段、finish_reason、SSE 结构 |
| `Reference/SGLang/sglang/python/sglang/srt/session/session_controller.py` | 会话与多轮 token 管理参考 |
| `Reference/SGLang/sglang/python/sglang/srt/entrypoints/http_server.py` | session 端点与 tokenize 端点参考 |
| `Runtime/.doc/Runtime_design.md` §3.5–§3.7 | 请求/响应、线程模型与错误处理 |
| `MobileNet_design.md` §4.5 | Service 层职责与接口 |

## 附录 B：需求追溯

| 需求 | 设计章节 | 测试用例 |
| --- | --- | --- |
| SVC-F-01 | §4.1、§4.11 | TC-SVC-01 |
| SVC-F-02 | §4.12 | TC-SVC-02 |
| SVC-F-03 | §4.6、§4.9 | TC-SVC-03 |
| SVC-F-04 | §4.2.2、§4.9.3 | TC-SVC-04 |
| SVC-F-05 | §4.8、§4.9.4 | TC-SVC-05 |
| SVC-F-06 | §4.6.1、§4.8 | TC-SVC-05、TC-SVC-08 |
| SVC-F-07 | §4.6.2、§4.7 | TC-SVC-06 |
| SVC-F-08 | §4.10 | TC-SVC-07 |
| SVC-F-09 | §4.9、附录 A | TC-SVC-09 |
| SVC-F-10 | §4.1 | TC-SVC-01 |
| SVC-F-11 | §4.9.1 | TC-SVC-07 |
| SVC-F-12 | §4.7 | TC-SVC-06 |
| SVC-NF-01 | §3.4、§4.5 | TC-SVC-10 |
| SVC-NF-02 | §4.5 | TC-SVC-04、TC-SVC-10 |
