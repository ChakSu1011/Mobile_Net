// 模块：OCL 错误映射与设备信息读取工具
//
// 职责：
//   把 CL 错误码映射为 Runtime ErrorCode，并提供 clGetDeviceInfo 的
//   字符串/整数/尺寸读取封装。
//
// 边界：
//   - 只做错误码与只读查询；不创建/释放 OCL 对象；
//   - 不修改设备状态；消息带 stage/native code；
//   - 不暴露到 Backend 公共头。
//
// 依赖：
//   Runtime/backend/ocl/ocl_api.h、Runtime/common/status.h、Utils/cpp/types.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_UTILS_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_UTILS_H_

#include <string>

#include "Runtime/backend/ocl/ocl_api.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace ocl {

// 返回 CL 错误码的稳定名称。
const C8* ClStatusName(cl_int code);

// 把 CL 错误码转成 Runtime Status。
//
// 参数：
//   code：CL 返回值；
//   stage：阶段名（Init/CreateBuffer/Upload/...）；
//   detail：附加描述。
// 返回：
//   CL_SUCCESS 返回 kOk；否则按错误类别返回 kDeviceOom/kKernelError/
//   kDeviceError/kInvalidArgument/kBufferError。
Status CheckCl(cl_int code, const C8* stage, const std::string& detail);

// 读取设备字符串信息。
Status GetDeviceString(OclFunctions* functions, cl_device_id device,
                       cl_device_info param, std::string* value);

// 读取 U32 设备信息。
Status GetDeviceU32(OclFunctions* functions, cl_device_id device,
                    cl_device_info param, U32* value);

// 读取 U64 设备信息。
Status GetDeviceU64(OclFunctions* functions, cl_device_id device,
                    cl_device_info param, U64* value);

// 读取 size_t 设备信息。
Status GetDeviceSize(OclFunctions* functions, cl_device_id device,
                     cl_device_info param, Usize* value);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_UTILS_H_
