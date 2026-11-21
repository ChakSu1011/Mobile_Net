// 模块：Session 推理引擎抽象实现
//
// 职责：
//   把 IEngine 调用转发给 runtime::Engine，并强制 threaded 模式。
//
// 边界：
//   不实现调度/推理算法；线程与回调语义由 Runtime Engine 保证。

#include "Service/session/engine_client.h"

#include <utility>

namespace mobilenet {
namespace service {

RuntimeInferenceEngine::RuntimeInferenceEngine() = default;

RuntimeInferenceEngine::~RuntimeInferenceEngine() {
  // Engine 析构会执行受控释放；显式 Destroy 时已幂等。
  (void)engine_.Destroy();
}

Status RuntimeInferenceEngine::Load(const runtime::EngineConfig& config) {
  runtime::EngineConfig effective = config;
  // Service 依赖推理线程与逐步回调。
  effective.runtime.threaded = true;
  return engine_.Load(effective);
}

Status RuntimeInferenceEngine::Start() { return engine_.Start(); }

Status RuntimeInferenceEngine::AddRequest(const runtime::Request& request,
                                          U64* request_id) {
  return engine_.AddRequest(request, request_id);
}

Status RuntimeInferenceEngine::CancelRequest(U64 request_id) {
  return engine_.CancelRequest(request_id);
}

void RuntimeInferenceEngine::SetResponseCallback(
    runtime::ResponseCallback callback) {
  engine_.SetResponseCallback(std::move(callback));
}

runtime::EngineStats RuntimeInferenceEngine::stats() const {
  return engine_.stats();
}

runtime::EngineState RuntimeInferenceEngine::state() const {
  return engine_.state();
}

Status RuntimeInferenceEngine::Destroy() { return engine_.Destroy(); }

}  // namespace service
}  // namespace mobilenet
