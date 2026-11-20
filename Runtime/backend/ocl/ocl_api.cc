// 模块：OCL 动态加载 API 表实现
//
// 职责：
//   按平台加载 OpenCL 运行库并解析 OclFunctions 中的函数指针。
//
// 边界：
//   只做加载/卸载与符号解析；不创建任何 OCL 对象。
//
// 依赖：
//   Runtime/backend/ocl/ocl_api.h、平台动态库 API。

#include "Runtime/backend/ocl/ocl_api.h"

#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mobilenet {
namespace ocl {
namespace {

// 打开动态库；成功返回句柄，失败返回空。
void* OpenLibrary(const std::string& path) {
#if defined(_WIN32)
  return static_cast<void*>(LoadLibraryA(path.c_str()));
#else
  return dlopen(path.c_str(), RTLD_NOW);
#endif
}

// 关闭动态库句柄。
void CloseLibrary(void* library) {
  if (library == nullptr) {
    return;
  }
#if defined(_WIN32)
  FreeLibrary(static_cast<HMODULE>(library));
#else
  dlclose(library);
#endif
}

// 解析符号；失败返回空。
void* FindSymbol(void* library, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(
      GetProcAddress(static_cast<HMODULE>(library), name));
#else
  return dlsym(library, name);
#endif
}

// 解析一个函数指针。
//
// 参数：
//   library：动态库句柄；
//   name：符号名；
//   out：输出函数指针。
// 返回：
//   找到返回 true。
template <typename T>
Bool LoadSymbol(void* library, const char* name, T* out) {
  void* symbol = FindSymbol(library, name);
  if (symbol == nullptr) {
    return false;
  }
  *out = reinterpret_cast<T>(symbol);
  return true;
}

// 平台默认的 OpenCL 库候选。
std::vector<std::string> DefaultCandidates() {
#if defined(_WIN32)
  return {"OpenCL.dll"};
#else
  return {"libOpenCL.so", "libOpenCL.so.1", "/vendor/lib64/libOpenCL.so",
          "/system/vendor/lib64/libOpenCL.so"};
#endif
}

}  // namespace

OclApi::~OclApi() { Unload(); }

Status OclApi::Load(const std::string& library_path) {
  if (library_ != nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclApi::Load: OpenCL library already loaded");
  }
  std::vector<std::string> candidates;
  if (!library_path.empty()) {
    candidates.push_back(library_path);
  } else {
    candidates = DefaultCandidates();
  }

  std::string last_error;
  for (const std::string& candidate : candidates) {
    library_ = OpenLibrary(candidate);
    if (library_ != nullptr) {
      break;
    }
    last_error = candidate;
  }
  if (library_ == nullptr) {
    return MakeStatus(
        ErrorCode::kBackendInitError,
        "OclApi::Load: cannot load OpenCL library (last tried: " +
            last_error + ")");
  }

#define MOBILENET_OCL_LOAD_REQUIRED(name)                              \
  if (!LoadSymbol(library_, #name, &functions_.name)) {                \
    Unload();                                                          \
    return MakeStatus(ErrorCode::kBackendInitError,                    \
                      "OclApi::Load: missing symbol: " #name);         \
  }

  MOBILENET_OCL_LOAD_REQUIRED(clGetPlatformIDs);
  MOBILENET_OCL_LOAD_REQUIRED(clGetPlatformInfo);
  MOBILENET_OCL_LOAD_REQUIRED(clGetDeviceIDs);
  MOBILENET_OCL_LOAD_REQUIRED(clGetDeviceInfo);
  MOBILENET_OCL_LOAD_REQUIRED(clCreateContext);
  MOBILENET_OCL_LOAD_REQUIRED(clReleaseContext);
  MOBILENET_OCL_LOAD_REQUIRED(clCreateCommandQueue);
  MOBILENET_OCL_LOAD_REQUIRED(clReleaseCommandQueue);
  MOBILENET_OCL_LOAD_REQUIRED(clCreateBuffer);
  MOBILENET_OCL_LOAD_REQUIRED(clReleaseMemObject);
  MOBILENET_OCL_LOAD_REQUIRED(clEnqueueWriteBuffer);
  MOBILENET_OCL_LOAD_REQUIRED(clEnqueueReadBuffer);
  MOBILENET_OCL_LOAD_REQUIRED(clCreateProgramWithSource);
  MOBILENET_OCL_LOAD_REQUIRED(clCreateProgramWithBinary);
  MOBILENET_OCL_LOAD_REQUIRED(clBuildProgram);
  MOBILENET_OCL_LOAD_REQUIRED(clGetProgramBuildInfo);
  MOBILENET_OCL_LOAD_REQUIRED(clGetProgramInfo);
  MOBILENET_OCL_LOAD_REQUIRED(clReleaseProgram);
  MOBILENET_OCL_LOAD_REQUIRED(clCreateKernel);
  MOBILENET_OCL_LOAD_REQUIRED(clReleaseKernel);
  MOBILENET_OCL_LOAD_REQUIRED(clSetKernelArg);
  MOBILENET_OCL_LOAD_REQUIRED(clEnqueueNDRangeKernel);
  MOBILENET_OCL_LOAD_REQUIRED(clFinish);
  MOBILENET_OCL_LOAD_REQUIRED(clFlush);
  MOBILENET_OCL_LOAD_REQUIRED(clWaitForEvents);
  MOBILENET_OCL_LOAD_REQUIRED(clReleaseEvent);
  MOBILENET_OCL_LOAD_REQUIRED(clGetEventProfilingInfo);
  MOBILENET_OCL_LOAD_REQUIRED(clGetKernelWorkGroupInfo);

#undef MOBILENET_OCL_LOAD_REQUIRED

  // OpenCL 2.0+ 可选；缺失时回退 clCreateCommandQueue。
  LoadSymbol(library_, "clCreateCommandQueueWithProperties",
             &functions_.clCreateCommandQueueWithProperties);
  return Status::Ok();
}

void OclApi::Unload() {
  if (library_ != nullptr) {
    CloseLibrary(library_);
    library_ = nullptr;
  }
  functions_ = OclFunctions{};
}

}  // namespace ocl
}  // namespace mobilenet
