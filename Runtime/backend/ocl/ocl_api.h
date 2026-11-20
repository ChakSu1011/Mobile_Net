// 模块：OCL 动态加载 API 表
//
// 职责：
//   动态加载 OpenCL 运行库（Android/Linux 的 libOpenCL.so、Windows 的
//   OpenCL.dll），解析 OCL Backend 所需的函数指针；不在链接期依赖 OpenCL。
//
// 边界：
//   - 只做加载/卸载与符号解析，不创建 Context/Queue/Buffer；
//   - 不暴露给 Backend 公共头；CL 类型只允许出现在 ocl/ 内部；
//   - 不调用任何 OCL 函数，除符号解析外无副作用。
//
// 依赖：
//   Runtime/backend/ocl/third_party/CL/cl.h、Runtime/common/status.h、
//   Utils/cpp/types.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_API_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_API_H_

#include <string>

#include "Runtime/backend/ocl/third_party/CL/cl.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace ocl {

// OpenCL 函数指针类型。
using PfnClGetPlatformIDs = cl_int(CL_API_CALL*)(cl_uint, cl_platform_id*,
                                                 cl_uint*);
using PfnClGetPlatformInfo = cl_int(CL_API_CALL*)(cl_platform_id,
                                                  cl_platform_info, size_t,
                                                  void*, size_t*);
using PfnClGetDeviceIDs = cl_int(CL_API_CALL*)(cl_platform_id, cl_device_type,
                                               cl_uint, cl_device_id*,
                                               cl_uint*);
using PfnClGetDeviceInfo = cl_int(CL_API_CALL*)(cl_device_id, cl_device_info,
                                                size_t, void*, size_t*);
using PfnClCreateContext = cl_context(CL_API_CALL*)(
    const cl_context_properties*, cl_uint, const cl_device_id*,
    void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*,
    cl_int*);
using PfnClReleaseContext = cl_int(CL_API_CALL*)(cl_context);
using PfnClCreateCommandQueue = cl_command_queue(CL_API_CALL*)(
    cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
using PfnClCreateCommandQueueWithProperties = cl_command_queue(CL_API_CALL*)(
    cl_context, cl_device_id, const cl_queue_properties*, cl_int*);
using PfnClReleaseCommandQueue = cl_int(CL_API_CALL*)(cl_command_queue);
using PfnClCreateBuffer = cl_mem(CL_API_CALL*)(cl_context, cl_mem_flags,
                                               size_t, void*, cl_int*);
using PfnClReleaseMemObject = cl_int(CL_API_CALL*)(cl_mem);
using PfnClEnqueueWriteBuffer = cl_int(CL_API_CALL*)(
    cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint,
    const cl_event*, cl_event*);
using PfnClEnqueueReadBuffer = cl_int(CL_API_CALL*)(
    cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint,
    const cl_event*, cl_event*);
using PfnClCreateProgramWithSource = cl_program(CL_API_CALL*)(
    cl_context, cl_uint, const char**, const size_t*, cl_int*);
using PfnClCreateProgramWithBinary = cl_program(CL_API_CALL*)(
    cl_context, cl_uint, const cl_device_id*, const size_t*,
    const unsigned char**, cl_int*, cl_int*);
using PfnClBuildProgram = cl_int(CL_API_CALL*)(
    cl_program, cl_uint, const cl_device_id*, const char*,
    void(CL_CALLBACK*)(cl_program, void*), void*);
using PfnClGetProgramBuildInfo = cl_int(CL_API_CALL*)(
    cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
using PfnClGetProgramInfo = cl_int(CL_API_CALL*)(cl_program, cl_program_info,
                                                 size_t, void*, size_t*);
using PfnClReleaseProgram = cl_int(CL_API_CALL*)(cl_program);
using PfnClCreateKernel = cl_kernel(CL_API_CALL*)(cl_program, const char*,
                                                  cl_int*);
using PfnClReleaseKernel = cl_int(CL_API_CALL*)(cl_kernel);
using PfnClSetKernelArg = cl_int(CL_API_CALL*)(cl_kernel, cl_uint, size_t,
                                               const void*);
using PfnClEnqueueNDRangeKernel = cl_int(CL_API_CALL*)(
    cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
    const size_t*, cl_uint, const cl_event*, cl_event*);
using PfnClFinish = cl_int(CL_API_CALL*)(cl_command_queue);
using PfnClFlush = cl_int(CL_API_CALL*)(cl_command_queue);
using PfnClWaitForEvents = cl_int(CL_API_CALL*)(cl_uint, const cl_event*);
using PfnClReleaseEvent = cl_int(CL_API_CALL*)(cl_event);
using PfnClGetEventProfilingInfo = cl_int(CL_API_CALL*)(
    cl_event, cl_profiling_info, size_t, void*, size_t*);
using PfnClGetKernelWorkGroupInfo = cl_int(CL_API_CALL*)(
    cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*,
    size_t*);

// OpenCL 函数表。
struct OclFunctions {
  PfnClGetPlatformIDs clGetPlatformIDs = nullptr;
  PfnClGetPlatformInfo clGetPlatformInfo = nullptr;
  PfnClGetDeviceIDs clGetDeviceIDs = nullptr;
  PfnClGetDeviceInfo clGetDeviceInfo = nullptr;
  PfnClCreateContext clCreateContext = nullptr;
  PfnClReleaseContext clReleaseContext = nullptr;
  PfnClCreateCommandQueue clCreateCommandQueue = nullptr;
  PfnClCreateCommandQueueWithProperties clCreateCommandQueueWithProperties =
      nullptr;
  PfnClReleaseCommandQueue clReleaseCommandQueue = nullptr;
  PfnClCreateBuffer clCreateBuffer = nullptr;
  PfnClReleaseMemObject clReleaseMemObject = nullptr;
  PfnClEnqueueWriteBuffer clEnqueueWriteBuffer = nullptr;
  PfnClEnqueueReadBuffer clEnqueueReadBuffer = nullptr;
  PfnClCreateProgramWithSource clCreateProgramWithSource = nullptr;
  PfnClCreateProgramWithBinary clCreateProgramWithBinary = nullptr;
  PfnClBuildProgram clBuildProgram = nullptr;
  PfnClGetProgramBuildInfo clGetProgramBuildInfo = nullptr;
  PfnClGetProgramInfo clGetProgramInfo = nullptr;
  PfnClReleaseProgram clReleaseProgram = nullptr;
  PfnClCreateKernel clCreateKernel = nullptr;
  PfnClReleaseKernel clReleaseKernel = nullptr;
  PfnClSetKernelArg clSetKernelArg = nullptr;
  PfnClEnqueueNDRangeKernel clEnqueueNDRangeKernel = nullptr;
  PfnClFinish clFinish = nullptr;
  PfnClFlush clFlush = nullptr;
  PfnClWaitForEvents clWaitForEvents = nullptr;
  PfnClReleaseEvent clReleaseEvent = nullptr;
  PfnClGetEventProfilingInfo clGetEventProfilingInfo = nullptr;
  PfnClGetKernelWorkGroupInfo clGetKernelWorkGroupInfo = nullptr;
};

// 动态加载的 OpenCL API。
//
// 功能：
//   加载运行库并解析 OclFunctions；Unload 释放库句柄。
//
// 边界：
//   - 不拥有 Context/Queue/Buffer；
//   - 未 Load 时 Functions 全为空；
//   - 线程安全：只在 Backend Init/Deinit 单线程调用。
class OclApi {
 public:
  OclApi() = default;
  ~OclApi();

  OclApi(const OclApi&) = delete;
  OclApi& operator=(const OclApi&) = delete;
  OclApi(OclApi&&) = delete;
  OclApi& operator=(OclApi&&) = delete;

  // 加载 OpenCL 运行库并解析符号。
  //
  // 参数：
  //   library_path：库路径；为空时按平台默认候选查找
  //                （Windows: OpenCL.dll；Android/Linux: libOpenCL.so 等）。
  // 返回：
  //   成功返回 kOk；库或必需符号缺失返回 kBackendInitError，消息含名称。
  Status Load(const std::string& library_path);

  // 卸载运行库并清空函数表；幂等。
  void Unload();

  // 是否已成功加载。
  Bool loaded() const { return library_ != nullptr; }

  // 返回函数表；未加载时为空表。
  OclFunctions* functions() { return &functions_; }

 private:
  // 运行库句柄；Windows 为 HMODULE，Unix 为 dlopen 句柄。
  void* library_ = nullptr;

  // 已解析的函数表。
  OclFunctions functions_{};
};

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_API_H_
