// 模块：OCL Program 缓存
//
// 职责：
//   按源码/编译选项/驱动/设备名生成 SHA-256 键，缓存 cl_program；
//   支持源码编译与磁盘二进制复用，失败时保存 build log。
//
// 边界：
//   - 不创建 Kernel；不绑定参数/工作尺寸；
//   - 不拥有 Context；缓存持有 cl_program 并在 Deinit 释放；
//   - 单设备、单线程使用。
//
// 依赖：
//   Runtime/backend/ocl/ocl_context.h、ocl_sha256.h、
//   Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_PROGRAM_CACHE_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_PROGRAM_CACHE_H_

#include <map>
#include <string>

#include "Runtime/backend/ocl/ocl_context.h"
#include "Runtime/backend/ocl/ocl_sha256.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace ocl {

// Program 缓存。
class OclProgramCache {
 public:
  OclProgramCache() = default;
  ~OclProgramCache();

  OclProgramCache(const OclProgramCache&) = delete;
  OclProgramCache& operator=(const OclProgramCache&) = delete;
  OclProgramCache(OclProgramCache&&) = delete;
  OclProgramCache& operator=(OclProgramCache&&) = delete;

  // 初始化缓存。
  //
  // 参数：
  //   context：非拥有 OclContext，生命周期由 Backend 保证；
  //   cache_dir：磁盘缓存目录；为空表示只做内存缓存（目录不存在时尝试创建）。
  // 返回：
  //   成功返回 kOk；参数非法/目录不可用返回 kInvalidArgument/kIoError。
  Status Init(OclContext* context, const std::string& cache_dir);

  // 释放所有 cl_program；幂等。
  Status Deinit();

  // 获取或编译 Program。
  //
  // 参数：
  //   key：逻辑键（如 kernel_key），参与哈希；
  //   source：OpenCL C 源码；
  //   options：编译选项；
  //   program：输出参数，成功时写入非空 cl_program；由缓存拥有。
  // 返回：
  //   成功返回 kOk；编译/加载失败返回 kKernelError；参数非法返回
  //   kInvalidArgument。
  Status GetOrBuild(const std::string& key, const std::string& source,
                    const std::string& options, cl_program* program);

  // 是否已初始化。
  Bool initialized() const { return initialized_; }

 private:
  // 从源码编译。
  Status BuildFromSource(const std::string& source,
                         const std::string& options,
                         const std::string& log_path, cl_program* program);

  // 从二进制加载。
  Status LoadBinary(const std::string& path, cl_program* program);

  // 保存 Program 二进制。
  Status SaveBinary(cl_program program, const std::string& path) const;

  // 读取 build log。
  Status GetBuildLog(cl_program program, std::string* log) const;

  // 生成缓存哈希。
  std::string MakeHash(const std::string& key, const std::string& source,
                       const std::string& options) const;

  // 确保缓存目录存在。
  Bool EnsureCacheDir() const;

  // 非拥有 Context。
  OclContext* context_ = nullptr;

  // 磁盘缓存目录；空表示不落盘。
  std::string cache_dir_;

  // 哈希 → cl_program。
  std::map<std::string, cl_program> programs_;

  // 是否初始化。
  Bool initialized_ = false;
};

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_PROGRAM_CACHE_H_
