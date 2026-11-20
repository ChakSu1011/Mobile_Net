// 模块：OCL 设备与上下文实现
//
// 职责：
//   加载 OpenCL、选择 GPU、创建 Context/Queue、查询设备能力。
//
// 边界：
//   不创建 Buffer/Kernel；不执行传输；单设备单队列。
//
// 依赖：
//   Runtime/backend/ocl/ocl_context.h、ocl_utils.h、C++17 标准库。

#include "Runtime/backend/ocl/ocl_context.h"

#include <string>
#include <vector>

#include "Runtime/backend/ocl/ocl_utils.h"

namespace mobilenet {
namespace ocl {
namespace {

// 平台与设备的配对。
struct DeviceRecord {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
};

// 读取平台字符串。
Status GetPlatformString(OclFunctions* functions, cl_platform_id platform,
                         cl_platform_info param, std::string* value) {
  Usize size = 0;
  cl_int code = functions->clGetPlatformInfo(platform, param, 0, nullptr,
                                             &size);
  if (code != CL_SUCCESS || size == 0) {
    return CheckCl(code, "GetPlatformInfo", "size query");
  }
  std::vector<C8> buffer(size, '\0');
  code = functions->clGetPlatformInfo(platform, param, size, buffer.data(),
                                      nullptr);
  if (code != CL_SUCCESS) {
    return CheckCl(code, "GetPlatformInfo", "string query");
  }
  value->assign(buffer.data());
  return Status::Ok();
}

// 读取 max work-item sizes。
Status GetWorkItemSizes(OclFunctions* functions, cl_device_id device,
                        U32 dimensions, std::array<Usize, 3>* sizes) {
  if (dimensions == 0 || dimensions > 3) {
    return MakeStatus(ErrorCode::kDeviceError,
                      "OclContext: unsupported work-item dimensions");
  }
  std::vector<size_t> values(dimensions, 0);
  const cl_int code = functions->clGetDeviceInfo(
      device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
      values.size() * sizeof(size_t), values.data(), nullptr);
  if (code != CL_SUCCESS) {
    return CheckCl(code, "GetDeviceInfo", "work item sizes");
  }
  for (U32 i = 0; i < dimensions; ++i) {
    (*sizes)[i] = static_cast<Usize>(values[i]);
  }
  return Status::Ok();
}

}  // namespace

OclContext::~OclContext() { Deinit(); }

Status OclContext::Init(const BackendConfig& config) {
  if (initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclContext::Init: already initialized");
  }
  if (config.device_id < 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclContext::Init: device_id must be non-negative");
  }
  std::string library_path;
  const auto it = config.options.find("opencl_library");
  if (it != config.options.end()) {
    library_path = it->second;
  }

  Status status = SelectDevice(library_path, config.device_id);
  if (!status.ok()) {
    Deinit();
    return status;
  }
  status = QueryInfo();
  if (!status.ok()) {
    Deinit();
    return status;
  }

  OclFunctions* functions = api_.functions();
  cl_int error = CL_SUCCESS;
  context_ = functions->clCreateContext(nullptr, 1, &device_, nullptr,
                                        nullptr, &error);
  if (error != CL_SUCCESS || context_ == nullptr) {
    const Status result = CheckCl(error, "CreateContext", info_.device_name);
    Deinit();
    return result;
  }

  // 优先使用 2.0+ 的属性版本；缺失时回退 1.2 API。
  if (functions->clCreateCommandQueueWithProperties != nullptr) {
    const cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    error = CL_SUCCESS;
    queue_ = functions->clCreateCommandQueueWithProperties(
        context_, device_, properties, &error);
    if (error != CL_SUCCESS) {
      queue_ = nullptr;
    }
  }
  if (queue_ == nullptr) {
    error = CL_SUCCESS;
    queue_ = functions->clCreateCommandQueue(
        context_, device_, CL_QUEUE_PROFILING_ENABLE, &error);
  }
  if (error != CL_SUCCESS || queue_ == nullptr) {
    const Status result = CheckCl(error, "CreateCommandQueue",
                                  info_.device_name);
    Deinit();
    return result;
  }

  initialized_ = true;
  return Status::Ok();
}

Status OclContext::Deinit() {
  OclFunctions* functions = api_.functions();
  if (queue_ != nullptr) {
    if (functions->clReleaseCommandQueue != nullptr) {
      (void)functions->clReleaseCommandQueue(queue_);
    }
    queue_ = nullptr;
  }
  if (context_ != nullptr) {
    if (functions->clReleaseContext != nullptr) {
      (void)functions->clReleaseContext(context_);
    }
    context_ = nullptr;
  }
  api_.Unload();
  platform_ = nullptr;
  device_ = nullptr;
  info_ = OclDeviceInfo{};
  initialized_ = false;
  return Status::Ok();
}

Status OclContext::SelectDevice(const std::string& library_path,
                                I32 device_id) {
  Status status = api_.Load(library_path);
  if (!status.ok()) {
    return status;
  }
  OclFunctions* functions = api_.functions();
  cl_uint platform_count = 0;
  cl_int error = functions->clGetPlatformIDs(0, nullptr, &platform_count);
  if (error != CL_SUCCESS || platform_count == 0) {
    return CheckCl(error, "GetPlatformIDs", "no OpenCL platform");
  }
  std::vector<cl_platform_id> platforms(platform_count);
  error = functions->clGetPlatformIDs(platform_count, platforms.data(),
                                      nullptr);
  if (error != CL_SUCCESS) {
    return CheckCl(error, "GetPlatformIDs", "enumerate platforms");
  }

  std::vector<DeviceRecord> records;
  for (const cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    error = functions->clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,
                                      nullptr, &device_count);
    if (error == CL_DEVICE_NOT_FOUND || device_count == 0) {
      continue;
    }
    if (error != CL_SUCCESS) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    error = functions->clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                                      device_count, devices.data(), nullptr);
    if (error != CL_SUCCESS) {
      continue;
    }
    for (const cl_device_id device : devices) {
      records.push_back({platform, device});
    }
  }
  if (records.empty()) {
    return MakeStatus(ErrorCode::kBackendInitError,
                      "OclContext: no OpenCL GPU device found");
  }
  if (device_id < 0 || static_cast<Usize>(device_id) >= records.size()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclContext: device_id out of range");
  }
  const DeviceRecord& selected = records[static_cast<Usize>(device_id)];
  platform_ = selected.platform;
  device_ = selected.device;
  return Status::Ok();
}

Status OclContext::QueryInfo() {
  OclFunctions* functions = api_.functions();
  Status status = GetPlatformString(functions, platform_, CL_PLATFORM_NAME,
                                    &info_.platform_name);
  if (!status.ok()) {
    return status;
  }
  status = GetPlatformString(functions, platform_, CL_PLATFORM_VENDOR,
                             &info_.platform_vendor);
  if (!status.ok()) {
    return status;
  }
  status = GetPlatformString(functions, platform_, CL_PLATFORM_VERSION,
                             &info_.platform_version);
  if (!status.ok()) {
    return status;
  }

  status = GetDeviceString(functions, device_, CL_DEVICE_NAME,
                           &info_.device_name);
  if (!status.ok()) {
    return status;
  }
  status = GetDeviceString(functions, device_, CL_DEVICE_VENDOR,
                           &info_.device_vendor);
  if (!status.ok()) {
    return status;
  }
  status = GetDeviceString(functions, device_, CL_DEVICE_VERSION,
                           &info_.device_version);
  if (!status.ok()) {
    return status;
  }
  status = GetDeviceString(functions, device_, CL_DRIVER_VERSION,
                           &info_.driver_version);
  if (!status.ok()) {
    return status;
  }
  (void)GetDeviceString(functions, device_, CL_DEVICE_OPENCL_C_VERSION,
                        &info_.opencl_c_version);
  status = GetDeviceString(functions, device_, CL_DEVICE_EXTENSIONS,
                           &info_.extensions);
  if (!status.ok()) {
    return status;
  }

  (void)GetDeviceU32(functions, device_, CL_DEVICE_MAX_COMPUTE_UNITS,
                     &info_.compute_units);
  (void)GetDeviceU32(functions, device_, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                     &info_.max_clock_mhz);
  (void)GetDeviceSize(functions, device_, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                      &info_.max_work_group_size);
  (void)GetDeviceU32(functions, device_,
                     CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                     &info_.max_work_item_dimensions);
  (void)GetWorkItemSizes(functions, device_,
                        info_.max_work_item_dimensions,
                        &info_.max_work_item_sizes);
  (void)GetDeviceU32(functions, device_,
                     CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
                     &info_.preferred_vector_width_half);
  (void)GetDeviceU32(functions, device_,
                     CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                     &info_.preferred_vector_width_float);

  (void)GetDeviceU64(functions, device_, CL_DEVICE_GLOBAL_MEM_SIZE,
                     &info_.global_mem_bytes);
  (void)GetDeviceU64(functions, device_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                     &info_.max_mem_alloc_bytes);
  (void)GetDeviceU64(functions, device_, CL_DEVICE_LOCAL_MEM_SIZE,
                     &info_.local_mem_bytes);
  (void)GetDeviceU32(functions, device_, CL_DEVICE_LOCAL_MEM_TYPE,
                     &info_.local_mem_type);
  (void)GetDeviceU64(functions, device_, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                     &info_.global_cache_bytes);
  (void)GetDeviceU32(functions, device_,
                     CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                     &info_.cacheline_bytes);
  (void)GetDeviceU32(functions, device_, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                     &info_.mem_base_addr_align_bits);
  (void)GetDeviceU64(functions, device_,
                     CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                     &info_.max_constant_buffer_bytes);
  (void)GetDeviceU32(functions, device_, CL_DEVICE_MAX_CONSTANT_ARGS,
                     &info_.max_constant_args);
  (void)GetDeviceU64(functions, device_, CL_DEVICE_SVM_CAPABILITIES,
                     &info_.svm_capabilities);
  Usize profiling_resolution = 0;
  (void)GetDeviceSize(functions, device_,
                      CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                      &profiling_resolution);
  info_.profiling_timer_resolution_ns =
      static_cast<U64>(profiling_resolution);

  U32 image_support = 0;
  (void)GetDeviceU32(functions, device_, CL_DEVICE_IMAGE_SUPPORT,
                     &image_support);
  info_.image_support = image_support != 0;
  info_.fp16 =
      info_.extensions.find("cl_khr_fp16") != std::string::npos;
  info_.subgroups =
      info_.extensions.find("cl_khr_subgroups") != std::string::npos ||
      info_.extensions.find("cl_khr_subgroup") != std::string::npos;
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
