// 模块：Runtime Destroy 默认实现
//
// 职责：
//   实现 IDestroyer：按依赖逆序同步设备、释放 OpEnv、缓冲、GraphEnv 与
//   Backend/Executor；重复调用幂等，单个阶段失败继续后续阶段并返回首个错误。
//
// 边界：
//   - 不创建资源，不直接调用设备 API；
//   - 不停止推理线程（由 Engine 在调用前 join）；
//   - 不删除 Engine 自身持有的对象，只释放它们登记的资源；
//   - scheduler 仅作为上下文信息保留，本阶段没有统一的 Shutdown 接口，
//     Destroyer 不遍历/取消请求。
//
// 依赖：
//   Graph/include/{graph,op}.h、Runtime/backend/backend.h、
//   Runtime/executor/executor.h、Runtime/common/{resource_stack,status}.h。

#include "Runtime/destroy/destroyer.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Graph/include/graph.h"
#include "Graph/include/op.h"
#include "Runtime/backend/backend.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/status.h"
#include "Runtime/executor/executor.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// 记录首个失败状态。
//
// 参数：
//   status：本阶段返回状态；
//   first_error：累计的首个错误，成功状态才会被覆盖。
// 返回：
//   无。
void RecordFirstError(const Status& status, Status* first_error) {
  if (!status.ok() && first_error->ok()) {
    *first_error = status;
  }
}

// 销毁上下文快照。
//
// 功能：
//   用指针与初始化标志标识一次 Destroy 调用；相同快照重复调用直接返回
//   上次结果，避免对已释放对象再次 Synchronize/Deinit。
//
// 边界：
//   - 只做身份比较，不解引用任何指针。
struct ContextKey {
  // 调度器指针。
  const void* scheduler = nullptr;

  // KV 管理器指针。
  const void* kv_cache = nullptr;

  // 执行器指针。
  const void* executor = nullptr;

  // 图指针。
  const void* graph = nullptr;

  // 后端指针。
  const void* backend = nullptr;

  // 资源栈指针。
  const void* resource_stack = nullptr;

  // 后端是否已初始化。
  Bool backend_initialized = false;

  // Executor 是否已初始化。
  Bool executor_initialized = false;
};

// 比较两个上下文快照是否一致。
//
// 参数：
//   left、right：待比较快照。
// 返回：
//   全部字段一致返回 true。
Bool operator==(const ContextKey& left, const ContextKey& right) {
  return left.scheduler == right.scheduler &&
         left.kv_cache == right.kv_cache &&
         left.executor == right.executor && left.graph == right.graph &&
         left.backend == right.backend &&
         left.resource_stack == right.resource_stack &&
         left.backend_initialized == right.backend_initialized &&
         left.executor_initialized == right.executor_initialized;
}

// 由 DestroyContext 生成身份快照。
//
// 参数：
//   context：销毁上下文。
// 返回：
//   仅包含指针与标志的 ContextKey。
ContextKey MakeContextKey(const DestroyContext& context) {
  ContextKey key;
  key.scheduler = context.scheduler;
  key.kv_cache = context.kv_cache;
  key.executor = context.executor;
  key.graph = context.graph;
  key.backend = context.backend;
  key.resource_stack = context.resource_stack;
  key.backend_initialized = context.backend_initialized;
  key.executor_initialized = context.executor_initialized;
  return key;
}

// 默认销毁器。
//
// 功能：
//   按固定阶段逆序释放运行时资源，并记住最近一次上下文结果用于幂等。
//
// 边界：
//   - 只在 Engine 停止推理线程后调用；
//   - 不拥有任何被释放对象；重复调用相同上下文不再触碰其成员。
class DefaultDestroyer final : public IDestroyer {
 public:
  DefaultDestroyer() = default;

  ~DefaultDestroyer() override = default;

  // 执行销毁。
  //
  // 参数与返回见 IDestroyer::Destroy。
  Status Destroy(const DestroyContext& context) override;

 private:
  // 是否已记录最近一次销毁。
  Bool has_last_context_ = false;

  // 最近一次销毁的上下文快照。
  ContextKey last_context_;

  // 最近一次销毁的返回状态。
  Status last_status_;
};

Status DefaultDestroyer::Destroy(const DestroyContext& context) {
  const ContextKey key = MakeContextKey(context);
  if (has_last_context_ && key == last_context_) {
    return last_status_;
  }

  Status first_error = Status::Ok();

  // 阶段 1：等待在途 Kernel 完成。后端未初始化时跳过。
  if (context.backend_initialized && context.backend != nullptr) {
    RecordFirstError(
        PrefixStatus(context.backend->Synchronize(nullptr),
                     "Destroy::Synchronize"),
        &first_error);
  }

  // 阶段 2：释放全部 OpEnv。输入/输出 op 可能不在拓扑序列中，单独遍历；
  // 用集合去重，避免同一个 op 被重复清理。
  if (context.graph != nullptr) {
    std::unordered_set<runtime_graph::Op*> cleared;
    auto clear_ops = [&cleared](const std::vector<runtime_graph::Op*>& ops) {
      for (runtime_graph::Op* op : ops) {
        if (op != nullptr && cleared.insert(op).second) {
          op->ClearOpEnv();
        }
      }
    };
    clear_ops(context.graph->GetOrderlyOPList());
    clear_ops(context.graph->GetAllInputOp());
    clear_ops(context.graph->GetAllOutputOp());
  }

  // 阶段 3：逆序执行 ResourceStack 中的缓冲/资源释放闭包（含 KV 缓冲）。
  if (context.resource_stack != nullptr) {
    RecordFirstError(
        PrefixStatus(context.resource_stack->Clear(),
                     "Destroy::ResourceStack"),
        &first_error);
  }

  // 阶段 4：释放 GraphEnv。必须在 OpEnv 全部清理之后执行。
  if (context.graph != nullptr) {
    context.graph->ClearGraphEnv();
  }

  // 阶段 5：反初始化执行器与后端。
  if (context.executor_initialized && context.executor != nullptr) {
    RecordFirstError(
        PrefixStatus(context.executor->Deinit(), "Destroy::Executor::Deinit"),
        &first_error);
  } else if (context.backend_initialized && context.backend != nullptr) {
    RecordFirstError(
        PrefixStatus(context.backend->Deinit(), "Destroy::Backend::Deinit"),
        &first_error);
  }

  has_last_context_ = true;
  last_context_ = key;
  last_status_ = first_error;
  return first_error;
}

}  // namespace

std::unique_ptr<IDestroyer> CreateDefaultDestroyer() {
  return std::make_unique<DefaultDestroyer>();
}

}  // namespace runtime
}  // namespace mobilenet
