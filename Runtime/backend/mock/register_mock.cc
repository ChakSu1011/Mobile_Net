// 模块：Mock 后端自注册入口
//
// 职责：
//   在静态初始化阶段把 "mock" 后端注册到 BackendRegistry。
//
// 边界：
//   只做注册，不初始化设备、不创建 Buffer/Kernel。
//
// 依赖：
//   Runtime/backend/backend_registry.h、Runtime/backend/mock/mock_backend.h。

#include "Runtime/backend/backend_registry.h"
#include "Runtime/backend/mock/mock_backend.h"
#include "Utils/cpp/types.h"

// Mock 后端注册名固定为 "mock"；工厂在 Create 时仅构造对象，不调用 Init。
MOBILENET_REGISTER_BACKEND("mock", mobilenet::mock::CreateMockBackend);
