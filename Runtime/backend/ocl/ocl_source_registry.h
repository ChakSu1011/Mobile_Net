// 模块：OCL Kernel 源码注册表
//
// 职责：
//   按 kernel_key 返回构建期嵌入的 OpenCL C 源码字符串。
//
// 边界：
//   - 只返回源码指针，不编译/缓存 Program；
//   - 未知 key 返回 nullptr；
//   - 源码由 CMake 从 kernels/<op>/*.cl 生成，不手写重复副本。
//
// 依赖：
//   Utils/cpp/types.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_SOURCE_REGISTRY_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_SOURCE_REGISTRY_H_

#include "Utils/cpp/types.h"

namespace mobilenet {
namespace ocl {

// 返回 kernel_key 对应的 OpenCL C 源码。
//
// 参数：
//   key：kernel_key，例如 "gather"、"add"、"sample_argmax"。
// 返回：
//   非空源码指针；未注册返回 nullptr。
const C8* GetOclSource(const C8* key);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_SOURCE_REGISTRY_H_
