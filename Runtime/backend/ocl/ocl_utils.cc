// 模块：OCL 错误映射与设备信息读取工具实现
//
// 职责：
//   实现 CL 错误码映射与 clGetDeviceInfo 读取封装。
//
// 边界：
//   只读查询；不创建/释放 OCL 对象。
//
// 依赖：
//   Runtime/backend/ocl/ocl_utils.h、C++17 标准库。

#include "Runtime/backend/ocl/ocl_utils.h"

#include <vector>

namespace mobilenet {
namespace ocl {

const C8* ClStatusName(cl_int code) {
  switch (code) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    default:
      return "CL_UNKNOWN";
  }
}

Status CheckCl(cl_int code, const C8* stage, const std::string& detail) {
  if (code == CL_SUCCESS) {
    return Status::Ok();
  }
  ErrorCode error = ErrorCode::kDeviceError;
  switch (code) {
    case CL_OUT_OF_RESOURCES:
    case CL_OUT_OF_HOST_MEMORY:
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      error = ErrorCode::kDeviceOom;
      break;
    case CL_BUILD_PROGRAM_FAILURE:
    case CL_INVALID_BINARY:
    case CL_INVALID_BUILD_OPTIONS:
    case CL_INVALID_PROGRAM:
    case CL_INVALID_PROGRAM_EXECUTABLE:
    case CL_COMPILER_NOT_AVAILABLE:
      error = ErrorCode::kKernelError;
      break;
    case CL_INVALID_KERNEL_NAME:
    case CL_INVALID_KERNEL:
    case CL_INVALID_ARG_INDEX:
    case CL_INVALID_ARG_VALUE:
    case CL_INVALID_WORK_DIMENSION:
    case CL_INVALID_WORK_GROUP_SIZE:
    case CL_INVALID_WORK_ITEM_SIZE:
    case CL_INVALID_GLOBAL_WORK_SIZE:
      error = ErrorCode::kKernelError;
      break;
    case CL_INVALID_VALUE:
    case CL_INVALID_DEVICE_TYPE:
    case CL_INVALID_PLATFORM:
    case CL_INVALID_DEVICE:
    case CL_INVALID_CONTEXT:
    case CL_INVALID_COMMAND_QUEUE:
    case CL_INVALID_HOST_PTR:
    case CL_INVALID_MEM_OBJECT:
      error = ErrorCode::kInvalidArgument;
      break;
    case CL_INVALID_BUFFER_SIZE:
      error = ErrorCode::kBufferError;
      break;
    default:
      error = ErrorCode::kDeviceError;
      break;
  }
  return MakeStatus(
      error, std::string("[backend=ocl][stage=") +
                 (stage != nullptr ? stage : "?") + "] native=" +
                 std::to_string(static_cast<int>(code)) + "(" +
                 ClStatusName(code) + "): " + detail);
}

Status GetDeviceString(OclFunctions* functions, cl_device_id device,
                       cl_device_info param, std::string* value) {
  if (functions == nullptr || device == nullptr || value == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GetDeviceString: null argument");
  }
  Usize size = 0;
  cl_int code =
      functions->clGetDeviceInfo(device, param, 0, nullptr, &size);
  if (code != CL_SUCCESS) {
    return CheckCl(code, "GetDeviceInfo", "size query");
  }
  if (size == 0) {
    value->clear();
    return Status::Ok();
  }
  std::vector<C8> buffer(size, '\0');
  code = functions->clGetDeviceInfo(device, param, size, buffer.data(),
                                    nullptr);
  if (code != CL_SUCCESS) {
    return CheckCl(code, "GetDeviceInfo", "string query");
  }
  value->assign(buffer.data());
  return Status::Ok();
}

Status GetDeviceU32(OclFunctions* functions, cl_device_id device,
                    cl_device_info param, U32* value) {
  if (functions == nullptr || device == nullptr || value == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GetDeviceU32: null argument");
  }
  cl_uint result = 0;
  const cl_int code = functions->clGetDeviceInfo(
      device, param, sizeof(result), &result, nullptr);
  if (code != CL_SUCCESS) {
    return CheckCl(code, "GetDeviceInfo", "u32 query");
  }
  *value = static_cast<U32>(result);
  return Status::Ok();
}

Status GetDeviceU64(OclFunctions* functions, cl_device_id device,
                    cl_device_info param, U64* value) {
  if (functions == nullptr || device == nullptr || value == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GetDeviceU64: null argument");
  }
  cl_ulong result = 0;
  const cl_int code = functions->clGetDeviceInfo(
      device, param, sizeof(result), &result, nullptr);
  if (code != CL_SUCCESS) {
    return CheckCl(code, "GetDeviceInfo", "u64 query");
  }
  *value = static_cast<U64>(result);
  return Status::Ok();
}

Status GetDeviceSize(OclFunctions* functions, cl_device_id device,
                     cl_device_info param, Usize* value) {
  if (functions == nullptr || device == nullptr || value == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GetDeviceSize: null argument");
  }
  size_t result = 0;
  const cl_int code = functions->clGetDeviceInfo(
      device, param, sizeof(result), &result, nullptr);
  if (code != CL_SUCCESS) {
    return CheckCl(code, "GetDeviceInfo", "size query");
  }
  *value = static_cast<Usize>(result);
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
