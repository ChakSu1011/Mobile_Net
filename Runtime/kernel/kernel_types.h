// 模块：Kernel 公共枚举与分类
//
// 职责：
//   定义 Graph/RuntimeGraph 与 Kernel 共享的算子类型、I/O 拷贝方向、
//   Decoder Block 融合阶段与 RoPE 布局枚举。
//
// 边界：
//   - 本文件不定义 Kernel 接口、张量描述或后端接口；
//   - 不依赖 LibTorch。
//
// 依赖：
//   仅依赖 C++17 标准库。

#ifndef MOBILENET_RUNTIME_KERNEL_KERNEL_TYPES_H_
#define MOBILENET_RUNTIME_KERNEL_KERNEL_TYPES_H_

namespace mobilenet {

// 融合节点算子类型。
//
// 说明：
//   kInput/kOutput 是图边界节点；其余为可执行的计算节点。
//   kCast 用于纯拷贝/类型转换路径与测试，不参与模型级融合规则。
enum class OpType {
  kUnknown = 0,
  kInput = 1,
  kOutput = 2,
  kEmbedding = 3,
  kRmsNorm = 4,
  kQkvProjection = 5,
  kAttention = 6,
  kResidualAdd = 7,
  kMlpGateUpAct = 8,
  kMlpDownResidual = 9,
  kLmHead = 10,
  kSampling = 11,
  kCast = 12,
};

// I/O 边界节点的数据搬运方向。
enum class CopyKind {
  kNone = 0,
  kH2D = 1,
  kD2H = 2,
};

// Decoder Block 融合阶段。
//
// 取值与设计文档 4.5 节融合骨架一一对应；kOutsideBlock 表示 Embedding、
// 最终归一化、LM Head、Sampling、Cast 等块外算子。
enum class BlockStage {
  kUnknown = 0,
  kStage1AttnNorm = 1,
  kStage2Qkv = 2,
  kStage3Attention = 3,
  kStage4AttnResidual = 4,
  kStage5MlpNorm = 5,
  kStage6MlpGateUp = 6,
  kStage7MlpDownResidual = 7,
  kOutsideBlock = 8,
};

// RoPE 位置编码布局。
enum class RopeLayout {
  kNeox = 0,
  kGptJ = 1,
};

// 返回 OpType 的稳定名称，例如 "kRmsNorm"。
const char* OpTypeName(OpType op_type);

// 返回 CopyKind 的稳定名称，例如 "kH2D"。
const char* CopyKindName(CopyKind copy_kind);

// 返回 BlockStage 的稳定名称，例如 "kStage1AttnNorm"。
const char* BlockStageName(BlockStage stage);

// 返回 RopeLayout 的稳定名称，例如 "kNeox"。
const char* RopeLayoutName(RopeLayout layout);

// 是否为可执行计算节点。
//
// 参数：
//   op_type：算子类型。
// 返回：
//   非 kInput/kOutput/kUnknown 时返回 true。
bool IsComputeOp(OpType op_type);

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_KERNEL_KERNEL_TYPES_H_
