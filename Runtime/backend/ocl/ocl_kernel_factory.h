// 模块：OCL Kernel 工厂
//
// 职责：
//   按 KernelDesc.kernel_key 创建对应算子的 OCL Kernel。
//
// 边界：
//   - P4.2 只实现 gather/add/sample_argmax；其余 key 返回 kUnsupportedOp；
//   - 不创建 Buffer/Program；不执行 Kernel；
//   - 不拥有 Context/ProgramCache。
//
// 依赖：
//   Runtime/backend/kernel.h、ocl_context.h、ocl_program_cache.h、
//   Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_KERNEL_FACTORY_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_KERNEL_FACTORY_H_

#include <memory>

#include "Runtime/backend/kernel.h"
#include "Runtime/backend/ocl/ocl_context.h"
#include "Runtime/backend/ocl/ocl_program_cache.h"
#include "Runtime/common/status.h"

namespace mobilenet {
namespace ocl {

// 创建 OCL Kernel。
//
// 参数：
//   desc：kernel 描述；kernel_key 决定实现；
//   context：非拥有 Context；
//   programs：非拥有 Program 缓存；
//   kernel：输出参数，成功时写入非空 Kernel。
// 返回：
//   成功返回 kOk；未实现 key 返回 kUnsupportedOp；参数非法返回
//   kInvalidArgument。
Status CreateOclKernel(const KernelDesc& desc, OclContext* context,
                       OclProgramCache* programs,
                       std::unique_ptr<Kernel>* kernel);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_KERNEL_FACTORY_H_
