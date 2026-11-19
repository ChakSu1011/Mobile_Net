// 模块：Runtime 接口编译检查
//
// 职责：
//   在“只冻结接口、不实现内部代码”的阶段，包含全部 Runtime 公共头文件并
//   编译，验证接口声明自洽、无循环依赖、无缺失前置声明。
//
// 边界：
//   - 不链接、不运行、不验证任何实现逻辑；
//   - 不替代后续各模块单元测试；
//   - 只用于接口阶段回归。
//
// 依赖：
//   Runtime 下全部公共头文件、C++17 编译器。

#include "Runtime/backend/backend.h"
#include "Runtime/backend/kernel.h"
#include "Runtime/backend/kernel_context.h"
#include "Runtime/backend/kernel_types.h"
#include "Runtime/common/data_types.h"
#include "Runtime/common/io_binder.h"
#include "Runtime/common/kv_allocator.h"
#include "Runtime/common/kv_view.h"
#include "Runtime/common/model_config.h"
#include "Runtime/common/op_env_factory.h"
#include "Runtime/common/plan.h"
#include "Runtime/common/sequence.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/runtime_config.h"
#include "Runtime/common/status.h"
#include "Runtime/destroy/destroyer.h"
#include "Runtime/engine/engine.h"
#include "Runtime/engine/engine_modules.h"
#include "Runtime/executor/executor.h"
#include "Runtime/kvcache/kv_cache_manager.h"
#include "Runtime/loader/loader.h"
#include "Runtime/scheduler/scheduler.h"
#include "Utils/cpp/types.h"

// 接口检查入口。
//
// 返回：
//   恒为 0；编译成功即表示接口声明自洽。
I32 main() { return 0; }
