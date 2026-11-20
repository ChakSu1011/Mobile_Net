// 模块：OCL Kernel 源码注册表实现
//
// 职责：
//   把 kernel_key 映射到构建期生成的源码字符串。
//
// 边界：
//   只查表；不编译、不缓存 Program。
//
// 依赖：
//   Runtime/backend/ocl/ocl_source_registry.h、
//   构建期生成的 <Op>_source.h。

#include "Runtime/backend/ocl/ocl_source_registry.h"

#include <string>

#include "Add_source.h"
#include "Gather_source.h"
#include "LinearSwiGlu_source.h"
#include "Linear_source.h"
#include "MegaAttention_source.h"
#include "PerHeadRmsNormRope_source.h"
#include "RmsNorm_source.h"
#include "Sample_source.h"

namespace mobilenet {
namespace ocl {

const C8* GetOclSource(const C8* key) {
  if (key == nullptr) {
    return nullptr;
  }
  const std::string name(key);
  if (name == "gather") {
    return kGatherSource;
  }
  if (name == "add") {
    return kAddSource;
  }
  if (name == "sample_argmax") {
    return kSampleSource;
  }
  if (name == "rms_norm") {
    return kRmsNormSource;
  }
  if (name == "linear") {
    return kLinearSource;
  }
  if (name == "linear_swi_glu") {
    return kLinearSwiGluSource;
  }
  if (name == "per_head_rmsnorm_rope") {
    return kPerHeadRmsNormRopeSource;
  }
  if (name == "mega_attention") {
    return kMegaAttentionSource;
  }
  return nullptr;
}

}  // namespace ocl
}  // namespace mobilenet
