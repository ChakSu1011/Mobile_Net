// 模块：Session 推理引擎抽象
//
// 职责：
//   为 Session/ChatService 提供最小 Engine 接口，生产实现包装
//   runtime::Engine，测试可注入确定性 FakeEngine。
//
// 边界：
//   - 不接触文本、Tokenizer 或 HTTP；
//   - 不修改 Runtime Engine 的线程/回调契约；
//   - 所有方法不抛出异常。
//
// 依赖：
//   Runtime/engine/engine.h、Runtime/common/sequence.h。

#ifndef MOBILENET_SERVICE_SESSION_ENGINE_CLIENT_H_
#define MOBILENET_SERVICE_SESSION_ENGINE_CLIENT_H_

#include <memory>

#include "Runtime/common/runtime_config.h"
#include "Runtime/common/sequence.h"
#include "Runtime/common/status.h"
#include "Runtime/engine/engine.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// 推理引擎接口。
//
// 功能：
//   Server 通过本接口加载、启动、提交、取消请求，并接收逐步响应。
//
// 边界：
//   - 实现必须保证 AddRequest/CancelRequest 线程安全；
//   - ResponseCallback 在推理线程触发，不得阻塞或重入；
//   - Load/Start/Destroy 不得并发调用。
class IEngine {
 public:
  virtual ~IEngine() = default;

  // 加载模型包并创建运行时资源。
  virtual Status Load(const runtime::EngineConfig& config) = 0;

  // 启动推理线程。
  virtual Status Start() = 0;

  // 提交请求。
  virtual Status AddRequest(const runtime::Request& request,
                            U64* request_id) = 0;

  // 取消请求。
  virtual Status CancelRequest(U64 request_id) = 0;

  // 设置响应回调。
  virtual void SetResponseCallback(runtime::ResponseCallback callback) = 0;

  // 返回累计统计。
  virtual runtime::EngineStats stats() const = 0;

  // 返回引擎状态。
  virtual runtime::EngineState state() const = 0;

  // 释放资源；幂等。
  virtual Status Destroy() = 0;
};

// Runtime Engine 适配器。
//
// 功能：
//   持有 runtime::Engine，把接口调用转发给 Engine，并强制 threaded=true。
class RuntimeInferenceEngine final : public IEngine {
 public:
  RuntimeInferenceEngine();
  ~RuntimeInferenceEngine() override;

  RuntimeInferenceEngine(const RuntimeInferenceEngine&) = delete;
  RuntimeInferenceEngine& operator=(const RuntimeInferenceEngine&) = delete;

  Status Load(const runtime::EngineConfig& config) override;
  Status Start() override;
  Status AddRequest(const runtime::Request& request,
                    U64* request_id) override;
  Status CancelRequest(U64 request_id) override;
  void SetResponseCallback(runtime::ResponseCallback callback) override;
  runtime::EngineStats stats() const override;
  runtime::EngineState state() const override;
  Status Destroy() override;

 private:
  // 实际 Engine。
  runtime::Engine engine_;
};

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_ENGINE_CLIENT_H_
