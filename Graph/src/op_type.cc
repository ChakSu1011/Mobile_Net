// 模块：Runtime Graph OpType 实现
//
// 职责：
//   实现 OpTypeToStr()，为每种 op 类型提供稳定字符串。
//
// 边界：
//   只做枚举到字符串的映射，不解释 op 语义。
//
// 依赖：
//   op_type.h。

#include "op_type.h"

namespace mobilenet {
namespace runtime_graph {

const char* OpTypeToStr(OpType op_type) {
  switch (op_type) {
    case OpType::kUnknown:
      return "unknown";
    case OpType::kInput:
      return "input";
    case OpType::kOutput:
      return "output";
    case OpType::kMegaAttention:
      return "mega_attention";
    case OpType::kLinear:
      return "linear";
    case OpType::kLinearRmsNorm:
      return "linear_rms_norm";
    case OpType::kLinearRmsNormRope:
      return "linear_rms_norm_rope";
    case OpType::kLinearRmsNormRopeQuantPagedKVStore:
      return "linear_rms_norm_rope_quant_paged_kv_store";
    case OpType::kLinearRopeRotateQuant:
      return "linear_rope_rotate_quant";
    case OpType::kLinearMul:
      return "linear_mul";
    case OpType::kLinearApe:
      return "linear_ape";
    case OpType::kRmsNorm:
      return "rms_norm";
    case OpType::kRmsNormRopeQuantPagedKVStore:
      return "rms_norm_rope_quant_paged_kv_store";
    case OpType::kRopeQuantLinear:
      return "rope_quant_linear";
    case OpType::kMegaMoe:
      return "mega_moe";
    case OpType::kDeepseekV3Router:
      return "deepseek_v3_router";
    case OpType::kHashRouter:
      return "hash_router";
    case OpType::kScoreRouter:
      return "score_router";
    case OpType::kLinearSwiGlu:
      return "linear_swi_glu";
    case OpType::kGather:
      return "gather";
    case OpType::kExpandEmbedding:
      return "expand_embedding";
    case OpType::kAdd:
      return "add";
    case OpType::kAddRmsNorm:
      return "add_rms_norm";
    case OpType::kFfn:
      return "ffn";
    case OpType::kHcPre:
      return "hc_pre";
    case OpType::kHcPost:
      return "hc_post";
    case OpType::kHcHead:
      return "hc_head";
    case OpType::kRope:
      return "rope";
    case OpType::kSparseMla:
      return "sparse_mla";
    case OpType::kQuant:
      return "quant";
    case OpType::kRotate:
      return "rotate";
    case OpType::kKvCompress:
      return "kv_compress";
    case OpType::kIndexScore:
      return "index_score";
    case OpType::kTopK:
      return "top_k";
    case OpType::kReshape:
      return "reshape";
    case OpType::kPack:
      return "pack";
    case OpType::kUnpack:
      return "unpack";
  }
  return "unknown";
}

}  // namespace runtime_graph
}  // namespace mobilenet
