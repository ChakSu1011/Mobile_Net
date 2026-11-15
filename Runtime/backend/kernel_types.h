// 模块：Backend/Kernel 公共枚举
//
// 职责：
//   定义 I/O 拷贝方向、Decoder Block 融合阶段与 RoPE 布局枚举，供
//   Backend/Kernel 实现与 OpEnv 使用。
//
// 边界：
//   - 不定义 Kernel 接口、算子类型或后端接口；
//   - 算子类型统一使用 Graph 的 runtime_graph::OpType；
//   - 不依赖 LibTorch。
//
// 依赖：
//   仅依赖 C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_KERNEL_TYPES_H_
#define MOBILENET_RUNTIME_BACKEND_KERNEL_TYPES_H_

namespace mobilenet {

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

// 返回 CopyKind 的稳定名称，例如 "kH2D"。
const char* CopyKindName(CopyKind copy_kind);

// 返回 BlockStage 的稳定名称，例如 "kStage1AttnNorm"。
const char* BlockStageName(BlockStage stage);

// 返回 RopeLayout 的稳定名称，例如 "kNeox"。
const char* RopeLayoutName(RopeLayout layout);

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_KERNEL_TYPES_H_
