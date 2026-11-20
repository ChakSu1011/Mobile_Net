// 模块：OCL Program 缓存实现
//
// 职责：
//   实现源码编译、二进制加载/保存、build log 与内存/磁盘缓存。
//
// 边界：
//   不创建 Kernel；不拥有 Context；单设备单线程。
//
// 依赖：
//   Runtime/backend/ocl/ocl_program_cache.h、C++17 标准库。

#include "Runtime/backend/ocl/ocl_program_cache.h"

#include <cerrno>
#include <fstream>
#include <string>
#include <vector>

#include "Runtime/backend/ocl/ocl_utils.h"

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace mobilenet {
namespace ocl {

OclProgramCache::~OclProgramCache() { Deinit(); }

Status OclProgramCache::Init(OclContext* context,
                             const std::string& cache_dir) {
  if (initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclProgramCache::Init: already initialized");
  }
  if (context == nullptr || !context->initialized()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclProgramCache::Init: context is not initialized");
  }
  context_ = context;
  cache_dir_ = cache_dir;
  if (!cache_dir_.empty() && !EnsureCacheDir()) {
    return MakeStatus(ErrorCode::kIoError,
                      "OclProgramCache::Init: cannot create cache dir: " +
                          cache_dir_);
  }
  initialized_ = true;
  return Status::Ok();
}

Status OclProgramCache::Deinit() {
  if (context_ != nullptr && context_->functions() != nullptr) {
    OclFunctions* functions = context_->functions();
    if (functions->clReleaseProgram != nullptr) {
      for (auto& entry : programs_) {
        if (entry.second != nullptr) {
          (void)functions->clReleaseProgram(entry.second);
        }
      }
    }
  }
  programs_.clear();
  context_ = nullptr;
  cache_dir_.clear();
  initialized_ = false;
  return Status::Ok();
}

Status OclProgramCache::GetOrBuild(const std::string& key,
                                   const std::string& source,
                                   const std::string& options,
                                   cl_program* program) {
  if (!initialized_ || context_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclProgramCache::GetOrBuild: not initialized");
  }
  if (program == nullptr || source.empty()) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "OclProgramCache::GetOrBuild: invalid program output or empty source");
  }

  const std::string hash = MakeHash(key, source, options);
  const auto cached = programs_.find(hash);
  if (cached != programs_.end()) {
    *program = cached->second;
    return Status::Ok();
  }

  std::string binary_path;
  std::string log_path;
  if (!cache_dir_.empty()) {
    binary_path = cache_dir_ + "/" + hash + ".bin";
    log_path = cache_dir_ + "/" + hash + ".log";
    std::ifstream binary_file(binary_path, std::ios::binary);
    if (binary_file.good()) {
      binary_file.close();
      cl_program loaded = nullptr;
      if (LoadBinary(binary_path, &loaded).ok()) {
        programs_.emplace(hash, loaded);
        *program = loaded;
        return Status::Ok();
      }
    }
  }

  cl_program built = nullptr;
  Status status = BuildFromSource(source, options, log_path, &built);
  if (!status.ok()) {
    return status;
  }
  if (!binary_path.empty()) {
    (void)SaveBinary(built, binary_path);
  }
  programs_.emplace(hash, built);
  *program = built;
  return Status::Ok();
}

Status OclProgramCache::BuildFromSource(const std::string& source,
                                        const std::string& options,
                                        const std::string& log_path,
                                        cl_program* program) {
  OclFunctions* functions = context_->functions();
  const cl_device_id device = context_->device();
  const char* source_ptr = source.c_str();
  const size_t source_size = source.size();
  cl_int error = CL_SUCCESS;
  cl_program built = functions->clCreateProgramWithSource(
      context_->context(), 1, &source_ptr, &source_size, &error);
  if (error != CL_SUCCESS || built == nullptr) {
    return CheckCl(error, "CreateProgramWithSource", "create program");
  }
  error = functions->clBuildProgram(built, 1, &device, options.c_str(),
                                    nullptr, nullptr);
  if (error != CL_SUCCESS) {
    std::string log;
    (void)GetBuildLog(built, &log);
    if (!log_path.empty() && !log.empty()) {
      std::ofstream log_file(log_path, std::ios::binary);
      if (log_file.good()) {
        log_file.write(log.data(), static_cast<std::streamsize>(log.size()));
      }
    }
    (void)functions->clReleaseProgram(built);
    return MakeStatus(
        ErrorCode::kKernelError,
        "[backend=ocl][stage=BuildProgram] native=" +
            std::to_string(static_cast<int>(error)) + ": " + log);
  }
  *program = built;
  return Status::Ok();
}

Status OclProgramCache::LoadBinary(const std::string& path,
                                   cl_program* program) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    return MakeStatus(ErrorCode::kIoError,
                      "OclProgramCache::LoadBinary: cannot open " + path);
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    return MakeStatus(ErrorCode::kKernelError,
                      "OclProgramCache::LoadBinary: empty binary");
  }
  std::vector<unsigned char> data(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(data.data()), size);
  file.close();

  OclFunctions* functions = context_->functions();
  const cl_device_id device = context_->device();
  const size_t binary_size = data.size();
  const unsigned char* binary_ptr = data.data();
  cl_int binary_status = CL_SUCCESS;
  cl_int error = CL_SUCCESS;
  cl_program loaded = functions->clCreateProgramWithBinary(
      context_->context(), 1, &device, &binary_size, &binary_ptr,
      &binary_status, &error);
  if (error != CL_SUCCESS || loaded == nullptr ||
      binary_status != CL_SUCCESS) {
    if (loaded != nullptr) {
      (void)functions->clReleaseProgram(loaded);
    }
    return MakeStatus(ErrorCode::kKernelError,
                      "OclProgramCache::LoadBinary: invalid binary");
  }
  error = functions->clBuildProgram(loaded, 1, &device, "", nullptr,
                                    nullptr);
  if (error != CL_SUCCESS) {
    (void)functions->clReleaseProgram(loaded);
    return MakeStatus(ErrorCode::kKernelError,
                      "OclProgramCache::LoadBinary: build binary failed");
  }
  *program = loaded;
  return Status::Ok();
}

Status OclProgramCache::SaveBinary(cl_program program,
                                   const std::string& path) const {
  OclFunctions* functions = context_->functions();
  size_t binary_size = 0;
  cl_int error = functions->clGetProgramInfo(
      program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size), &binary_size,
      nullptr);
  if (error != CL_SUCCESS || binary_size == 0) {
    return MakeStatus(ErrorCode::kKernelError,
                      "OclProgramCache::SaveBinary: no binary size");
  }
  std::vector<unsigned char> data(binary_size, 0);
  unsigned char* data_ptr = data.data();
  error = functions->clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                                      sizeof(data_ptr), &data_ptr, nullptr);
  if (error != CL_SUCCESS) {
    return MakeStatus(ErrorCode::kKernelError,
                      "OclProgramCache::SaveBinary: get binary failed");
  }
  std::ofstream file(path, std::ios::binary);
  if (!file.good()) {
    return MakeStatus(ErrorCode::kIoError,
                      "OclProgramCache::SaveBinary: cannot open " + path);
  }
  file.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size()));
  return file.good() ? Status::Ok()
                     : MakeStatus(ErrorCode::kIoError,
                                  "OclProgramCache::SaveBinary: write failed");
}

Status OclProgramCache::GetBuildLog(cl_program program,
                                    std::string* log) const {
  if (log == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclProgramCache::GetBuildLog: null log");
  }
  OclFunctions* functions = context_->functions();
  Usize size = 0;
  cl_int error = functions->clGetProgramBuildInfo(
      program, context_->device(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
  if (error != CL_SUCCESS || size == 0) {
    log->clear();
    return Status::Ok();
  }
  std::vector<C8> buffer(size, '\0');
  error = functions->clGetProgramBuildInfo(program, context_->device(),
                                           CL_PROGRAM_BUILD_LOG, size,
                                           buffer.data(), nullptr);
  if (error != CL_SUCCESS) {
    return CheckCl(error, "GetProgramBuildInfo", "read log");
  }
  log->assign(buffer.data());
  return Status::Ok();
}

std::string OclProgramCache::MakeHash(const std::string& key,
                                      const std::string& source,
                                      const std::string& options) const {
  const OclDeviceInfo& info = context_->info();
  return Sha256Hex(key + "\n" + source + "\n" + options + "\n" +
                   info.driver_version + "\n" + info.device_name + "\n" +
                   info.device_version);
}

Bool OclProgramCache::EnsureCacheDir() const {
  if (cache_dir_.empty()) {
    return true;
  }
#if defined(_WIN32)
  const int result = _mkdir(cache_dir_.c_str());
#else
  const int result = mkdir(cache_dir_.c_str(), 0755);
#endif
  return result == 0 || errno == EEXIST;
}

}  // namespace ocl
}  // namespace mobilenet
