// 模块：OCL 后端自注册入口
//
// 职责：
//   在静态初始化阶段把 "ocl" 后端注册到 BackendRegistry。
//
// 边界：
//   只做注册，不初始化设备、不创建 Buffer/Kernel。
//
// 依赖：
//   Runtime/backend/backend_registry.h、
//   Runtime/backend/ocl/ocl_backend.h。

#include "Runtime/backend/backend_registry.h"
#include "Runtime/backend/ocl/ocl_backend.h"

// OCL 后端注册名固定为 "ocl"；工厂只构造对象，Init 由 Loader 调用。
MOBILENET_REGISTER_BACKEND("ocl", mobilenet::ocl::CreateOclBackend);
