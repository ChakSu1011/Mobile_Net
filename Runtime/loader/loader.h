// 模块：Runtime Loader 统一接口
//
// 职责：
//   定义 Engine 在加载阶段调用的 ILoader：读取并校验模型包、重建 Graph、
//   校验 IR 哈希、初始化 Backend、上传权重、分配缓冲并创建 GraphEnv/OpEnv；
//   返回 Engine 可直接持有的 Graph/Backend/ModelConfig。
//
// 边界：
//   - 不做请求调度、KV 策略或设备 API 直调；
//   - 不实现 Graph 构图语义（使用 Graph 公共接口）；
//   - OpEnvFactory 为 Loader 内部工具，不暴露给 Engine；
//   - 失败时必须把已创建资源登记到 ResourceStack，不留下半初始化状态。
//
// 依赖：
//   Runtime/common/*、Graph/include/graph.h、Runtime/backend/backend.h。

#ifndef MOBILENET_RUNTIME_LOADER_LOADER_H_
#define MOBILENET_RUNTIME_LOADER_LOADER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "Graph/include/graph.h"
#include "Runtime/backend/backend.h"
#include "Runtime/common/model_config.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/runtime_config.h"
#include "Runtime/common/status.h"

namespace mobilenet {
namespace runtime {

// 加载配置。
//
// 功能：
//   指定模型包目录与运行参数；Engine 从 EngineConfig 转换后传给 ILoader。
//
// 边界：
//   - package_dir 必须包含 model.mgf，可选 weights.bin；
//   - runtime 中的预算/开关由 Loader 解释，未知字段不得静默忽略。
struct LoadConfig {
  // 模型包目录。
  std::string package_dir;

  // 运行参数。
  RuntimeConfig runtime;
};

// 加载结果。
//
// 功能：
//   Loader 产出的可执行图、后端实例与模型配置；Engine 接管 unique_ptr。
//
// 边界：
//   - 失败时 result 不修改；
//   - graph/backend 成功时非空，由 Engine 负责最终释放；
//   - weights_bytes/arena_bytes/io_bytes 仅用于统计与日志。
struct LoadResult {
  // 重建并校验通过的图。
  std::unique_ptr<runtime_graph::Graph> graph;

  // 初始化完成的后端；Engine 接管后移交 Executor 持有。
  std::unique_ptr<Backend> backend;

  // 从 manifest 解析的模型配置。
  ModelConfig model;

  // 权重常驻字节数。
  uint64_t weights_bytes = 0;

  // Activation arena 字节数。
  uint64_t arena_bytes = 0;

  // 输入/输出缓冲字节数。
  uint64_t io_bytes = 0;
};

// Loader 统一接口。
//
// 功能：
//   Engine 在 Load 阶段调用一次，完成全部加载期编排；Loader 内部创建
//   OpEnvFactory（ExecBackend 提供默认实现）并逐 op 生成 OpEnv。
//
// 边界：
//   - 只允许在创建后单线程调用；
//   - 不持有长期状态；加载完成后由 Engine 析构；
//   - 失败时返回错误码，已创建资源由 ResourceStack 逆序释放；
//   - 不向 Engine 暴露 OpEnvFactory。
class ILoader {
 public:
  virtual ~ILoader() = default;

  // 加载模型包并创建运行时资源。
  //
  // 参数：
  //   config：包目录与运行参数；
  //   resource_stack：资源登记栈，非空；Loader 创建的设备缓冲/OpEnv 等必须
  //                   登记以便失败回滚与正常销毁；
  //   result：输出参数，成功时写入 Graph/Backend/ModelConfig。
  // 返回：
  //   成功返回 kOk；包损坏/哈希不一致返回 kInvalidModelFile；后端错误返回
  //   kUnsupportedBackend/kBackendInitError/kUnsupportedOp；显存不足返回
  //   kDeviceOom；其它失败返回对应错误码。
  virtual Status Load(const LoadConfig& config, ResourceStack* resource_stack,
                      LoadResult* result) = 0;
};

// 创建默认 Loader。
//
// 返回：
//   非空 Loader；由 Engine 持有。实现方为 RuntimeCore。
std::unique_ptr<ILoader> CreateDefaultLoader();

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_LOADER_LOADER_H_
