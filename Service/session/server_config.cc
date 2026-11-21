// 模块：Session 服务配置实现
//
// 职责：
//   读取 JSON 配置文件并映射到 ServerConfig，校验取值范围。
//
// 边界：
//   不加载模型；未知字段忽略；缺省字段使用默认值。

#include "Service/session/server_config.h"

#include <limits>
#include <string>
#include <vector>

#include "Runtime/common/file_util.h"
#include "Runtime/common/json.h"

namespace mobilenet {
namespace service {
namespace {

// 配置文件最大字节数。
constexpr U64 kConfigMaxBytes = 1ull * 1024ull * 1024ull;

// 读取整个文本文件。
Status ReadTextFile(const std::string& path, std::string* text) {
  U64 size = 0;
  Status status = runtime::GetFileSize(path, &size);
  if (!status.ok()) {
    return PrefixStatus(status, "LoadServerConfig: stat " + path);
  }
  if (size > kConfigMaxBytes) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "LoadServerConfig: config file too large");
  }
  std::vector<U8> buffer;
  status = runtime::ReadFileRange(path, 0, size, &buffer);
  if (!status.ok()) {
    return PrefixStatus(status, "LoadServerConfig: read " + path);
  }
  text->assign(reinterpret_cast<const C8*>(buffer.data()), buffer.size());
  return Status::Ok();
}

// 读取子对象；缺失返回 nullptr。
const runtime::JsonValue* FindObject(const runtime::JsonValue& root,
                                     const std::string& key) {
  const runtime::JsonValue* value = root.Find(key);
  if (value == nullptr || !value->IsObject()) {
    return nullptr;
  }
  return value;
}

// 读取可选字符串。
std::string ReadString(const runtime::JsonValue* object,
                       const std::string& key,
                       const std::string& default_value) {
  if (object == nullptr) {
    return default_value;
  }
  const runtime::JsonValue* value = object->Find(key);
  if (value == nullptr || !value->IsString()) {
    return default_value;
  }
  return value->AsString();
}

// 读取可选布尔。
Bool ReadBool(const runtime::JsonValue* object, const std::string& key,
              Bool default_value) {
  if (object == nullptr) {
    return default_value;
  }
  const runtime::JsonValue* value = object->Find(key);
  if (value == nullptr || !value->IsBool()) {
    return default_value;
  }
  return value->AsBool();
}

// 读取可选整数。
I64 ReadInt(const runtime::JsonValue* object, const std::string& key,
            I64 default_value) {
  if (object == nullptr) {
    return default_value;
  }
  const runtime::JsonValue* value = object->Find(key);
  if (value == nullptr || !value->IsInteger()) {
    return default_value;
  }
  return value->AsInt64();
}

// 校验整数范围并转换。
Status CheckRange(I64 value, I64 minimum, I64 maximum,
                  const std::string& name, I64* out) {
  if (value < minimum || value > maximum) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LoadServerConfig: " + name + " out of range");
  }
  *out = value;
  return Status::Ok();
}

}  // namespace

Status LoadServerConfig(const std::string& path, ServerConfig* config) {
  if (config == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LoadServerConfig: config must not be null");
  }
  if (path.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LoadServerConfig: path must not be empty");
  }
  std::string text;
  Status status = ReadTextFile(path, &text);
  if (!status.ok()) {
    return status;
  }
  runtime::JsonValue root;
  status = runtime::ParseJson(text, &root);
  if (!status.ok()) {
    return PrefixStatus(status, "LoadServerConfig: parse");
  }
  if (!root.IsObject()) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "LoadServerConfig: root must be an object");
  }

  ServerConfig parsed;
  const runtime::JsonValue* server = FindObject(root, "server");
  const runtime::JsonValue* model = FindObject(root, "model");
  const runtime::JsonValue* engine = FindObject(root, "engine");
  const runtime::JsonValue* generation = FindObject(root, "generation");
  const runtime::JsonValue* session = FindObject(root, "session");
  const runtime::JsonValue* logging = FindObject(root, "logging");

  parsed.host = ReadString(server, "host", parsed.host);
  const I64 port = ReadInt(server, "port", parsed.port);
  const I64 max_body =
      ReadInt(server, "max_body_bytes", parsed.max_body_bytes);
  const I64 max_connections =
      ReadInt(server, "max_connections", parsed.max_connections);
  const I64 request_timeout =
      ReadInt(server, "request_timeout_ms", parsed.request_timeout_ms);
  const I64 read_timeout =
      ReadInt(server, "read_timeout_ms", parsed.read_timeout_ms);
  I64 checked = 0;
  status = CheckRange(port, 0, 65535, "server.port", &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.port = static_cast<U16>(checked);
  status = CheckRange(max_body, 1024, 64 * 1024 * 1024,
                      "server.max_body_bytes", &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.max_body_bytes = static_cast<U32>(checked);
  status = CheckRange(max_connections, 1, 4096,
                      "server.max_connections", &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.max_connections = static_cast<U32>(checked);
  status = CheckRange(request_timeout, 1, 24 * 3600 * 1000,
                      "server.request_timeout_ms", &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.request_timeout_ms = static_cast<U32>(checked);
  status = CheckRange(read_timeout, 1, 24 * 3600 * 1000,
                      "server.read_timeout_ms", &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.read_timeout_ms = static_cast<U32>(checked);

  parsed.model_name = ReadString(model, "name", parsed.model_name);
  parsed.model_dir = ReadString(model, "model_dir", parsed.model_dir);
  parsed.package_dir = ReadString(model, "package_dir", parsed.package_dir);

  parsed.backend = ReadString(engine, "backend", parsed.backend);
  const I64 device_id = ReadInt(engine, "device_id", parsed.device_id);
  const I64 kv_cache = ReadInt(engine, "kv_cache_bytes",
                               static_cast<I64>(parsed.kv_cache_bytes));
  const I64 max_num_tokens =
      ReadInt(engine, "max_num_tokens", parsed.max_num_tokens);
  const I64 max_concurrent =
      ReadInt(engine, "max_concurrent_requests",
              parsed.max_concurrent_requests);
  parsed.verify_weights_sha256 =
      ReadBool(engine, "verify_weights_sha256",
               parsed.verify_weights_sha256);
  parsed.recompute_attention =
      ReadBool(engine, "recompute_attention", parsed.recompute_attention);
  status = CheckRange(device_id, 0, 1024, "engine.device_id", &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.device_id = static_cast<I32>(checked);
  status = CheckRange(kv_cache, 0, std::numeric_limits<I64>::max(),
                      "engine.kv_cache_bytes", &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.kv_cache_bytes = static_cast<U64>(checked);
  status = CheckRange(max_num_tokens, 1, 1 << 20, "engine.max_num_tokens",
                      &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.max_num_tokens = static_cast<I32>(checked);
  status = CheckRange(max_concurrent, 1, 1024,
                      "engine.max_concurrent_requests", &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.max_concurrent_requests = static_cast<I32>(checked);

  const I64 default_max_tokens =
      ReadInt(generation, "default_max_tokens", parsed.default_max_tokens);
  status = CheckRange(default_max_tokens, 1, 1 << 20,
                      "generation.default_max_tokens", &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.default_max_tokens = static_cast<I32>(checked);
  const Bool enable_thinking =
      ReadBool(generation, "enable_thinking", false);
  parsed.thinking =
      enable_thinking ? ThinkingMode::kEnabled : ThinkingMode::kDisabled;

  const I64 max_sessions =
      ReadInt(session, "max_sessions", parsed.max_sessions);
  const I64 idle_timeout = ReadInt(session, "idle_timeout_s",
                                   parsed.session_idle_timeout_s);
  status = CheckRange(max_sessions, 1, 65536, "session.max_sessions",
                      &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.max_sessions = static_cast<U32>(checked);
  status = CheckRange(idle_timeout, 0, 365 * 24 * 3600,
                      "session.idle_timeout_s", &checked);
  if (!status.ok()) {
    return status;
  }
  parsed.session_idle_timeout_s = static_cast<U32>(checked);

  parsed.log_level = ReadString(logging, "level", parsed.log_level);

  if (parsed.model_name.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LoadServerConfig: model.name must not be empty");
  }
  if (parsed.model_dir.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LoadServerConfig: model.model_dir must not be empty");
  }
  if (parsed.host.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LoadServerConfig: server.host must not be empty");
  }
  *config = std::move(parsed);
  return Status::Ok();
}

}  // namespace service
}  // namespace mobilenet
