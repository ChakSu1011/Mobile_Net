// 模块：Session 服务配置
//
// 职责：
//   定义服务启动配置，并从 JSON 文件加载与校验。
//
// 边界：
//   - 只解析配置，不加载模型/Tokenizer；
//   - 路径按进程当前目录解释，不做环境变量展开；
//   - 未知字段忽略，非法值返回 kInvalidArgument。
//
// 依赖：
//   Service/tokenizer/tokenizer.h（ThinkingMode）、Runtime/common/json.h。

#ifndef MOBILENET_SERVICE_SESSION_SERVER_CONFIG_H_
#define MOBILENET_SERVICE_SESSION_SERVER_CONFIG_H_

#include <string>

#include "Runtime/common/status.h"
#include "Service/tokenizer/tokenizer.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// 服务配置。
struct ServerConfig {
  // HTTP 监听地址；默认仅本机。
  std::string host = "127.0.0.1";

  // HTTP 端口；0 表示由系统分配（测试用）。
  U16 port = 8080;

  // 请求体上限（字节）。
  U32 max_body_bytes = 1u << 20;

  // 最大并发 HTTP 连接数。
  U32 max_connections = 32;

  // 单请求总超时（毫秒）。
  U32 request_timeout_ms = 120000;

  // 套接字读取超时（毫秒）。
  U32 read_timeout_ms = 30000;

  // 对外模型名。
  std::string model_name = "qwen3_0_6b";

  // Tokenizer 资产目录。
  std::string model_dir;

  // 模型包目录。
  std::string package_dir;

  // 后端名称。
  std::string backend = "ocl";

  // 设备序号。
  I32 device_id = 0;

  // KV 显存预算；0 表示自动。
  U64 kv_cache_bytes = 0;

  // 单步 token 预算，必须与模型包一致。
  I32 max_num_tokens = 256;

  // Engine 并发请求上限。
  I32 max_concurrent_requests = 1;

  // 是否校验权重 SHA-256。
  Bool verify_weights_sha256 = true;

  // 是否使用无 KV 的 recompute 模式（调试/Mock）。
  Bool recompute_attention = false;

  // 默认最大生成 token 数。
  I32 default_max_tokens = 128;

  // 默认思考模式。
  ThinkingMode thinking = ThinkingMode::kDisabled;

  // 最大会话数。
  U32 max_sessions = 64;

  // 会话空闲超时（秒）。
  U32 session_idle_timeout_s = 3600;

  // 单会话保留的最大消息条数；超出后丢弃最旧的非 system 消息。
  U32 max_history_messages = 64;

  // 日志级别：error/warn/info/debug。
  std::string log_level = "info";
};

// 从 JSON 文件加载配置。
//
// 参数：
//   path：配置文件路径；
//   config：输出参数，非空；成功时写入配置。
// 返回：
//   成功返回 kOk；文件/解析错误返回 kIoError/kInvalidModelFile；
//   字段值非法返回 kInvalidArgument。
Status LoadServerConfig(const std::string& path, ServerConfig* config);

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_SERVER_CONFIG_H_
