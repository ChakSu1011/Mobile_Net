// 模块：Kernel 公共枚举实现
//
// 职责：
//   实现算子类型、拷贝方向、融合阶段与 RoPE 布局的名称查询。
//
// 边界：
//   不包含 Kernel 执行逻辑。

#include "Runtime/kernel/kernel_types.h"

namespace mobilenet {

const char* OpTypeName(OpType op_type) {
  switch (op_type) {
    case OpType::kUnknown:
      return "kUnknown";
    case OpType::kInput:
      return "kInput";
    case OpType::kOutput:
      return "kOutput";
    case OpType::kEmbedding:
      return "kEmbedding";
    case OpType::kRmsNorm:
      return "kRmsNorm";
    case OpType::kQkvProjection:
      return "kQkvProjection";
    case OpType::kAttention:
      return "kAttention";
    case OpType::kResidualAdd:
      return "kResidualAdd";
    case OpType::kMlpGateUpAct:
      return "kMlpGateUpAct";
    case OpType::kMlpDownResidual:
      return "kMlpDownResidual";
    case OpType::kLmHead:
      return "kLmHead";
    case OpType::kSampling:
      return "kSampling";
    case OpType::kCast:
      return "kCast";
  }
  return "kUnknown";
}

const char* CopyKindName(CopyKind copy_kind) {
  switch (copy_kind) {
    case CopyKind::kNone:
      return "kNone";
    case CopyKind::kH2D:
      return "kH2D";
    case CopyKind::kD2H:
      return "kD2H";
  }
  return "kNone";
}

const char* BlockStageName(BlockStage stage) {
  switch (stage) {
    case BlockStage::kUnknown:
      return "kUnknown";
    case BlockStage::kStage1AttnNorm:
      return "kStage1AttnNorm";
    case BlockStage::kStage2Qkv:
      return "kStage2Qkv";
    case BlockStage::kStage3Attention:
      return "kStage3Attention";
    case BlockStage::kStage4AttnResidual:
      return "kStage4AttnResidual";
    case BlockStage::kStage5MlpNorm:
      return "kStage5MlpNorm";
    case BlockStage::kStage6MlpGateUp:
      return "kStage6MlpGateUp";
    case BlockStage::kStage7MlpDownResidual:
      return "kStage7MlpDownResidual";
    case BlockStage::kOutsideBlock:
      return "kOutsideBlock";
  }
  return "kUnknown";
}

const char* RopeLayoutName(RopeLayout layout) {
  switch (layout) {
    case RopeLayout::kNeox:
      return "kNeox";
    case RopeLayout::kGptJ:
      return "kGptJ";
  }
  return "kUnknown";
}

bool IsComputeOp(OpType op_type) {
  return op_type != OpType::kUnknown && op_type != OpType::kInput &&
         op_type != OpType::kOutput;
}

}  // namespace mobilenet
