// 模块：Runtime Graph OpType 枚举
//
// 职责：
//   定义 graph 节点类型，供模型加载构图、Engine 初始化选择 OpEnv、日志和
//   调试输出使用；同时提供稳定的字符串名称。
//
// 边界：
//   - 只表达 op 的逻辑类别，不携带 kernel 实现、精度或 shape 信息；
//   - kInput/kOutput 是图边界 op，普通计算 op 不得使用；
//   - 新增枚举值必须同步更新 OpTypeToStr()，否则 dump_ir 会输出 unknown；
//   - 枚举顺序是 IR 契约的一部分，序列化或 Python 绑定不得随意调整。
//
// 依赖：
//   仅依赖 C++17 标准库。

#ifndef MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_OP_TYPE_H_
#define MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_OP_TYPE_H_

namespace mobilenet {
namespace runtime_graph {

// OpType 表示 graph 中 op 的类型。
enum class OpType {
  kUnknown = 0,
  kInput,
  kOutput,
  kMegaAttention,
  kLinear,
  kLinearRmsNorm,
  kLinearRmsNormRope,
  kLinearRmsNormRopeQuantPagedKVStore,
  kLinearRopeRotateQuant,
  kLinearMul,
  kLinearApe,
  kRmsNorm,
  kRmsNormRopeQuantPagedKVStore,
  kRopeQuantLinear,
  kMegaMoe,
  kDeepseekV3Router,
  kHashRouter,
  kScoreRouter,
  kLinearSwiGlu,
  kGather,
  kExpandEmbedding,
  kAdd,
  kAddRmsNorm,
  kFfn,
  kHcPre,
  kHcPost,
  kHcHead,
  kRope,
  kSparseMla,
  kQuant,
  kRotate,
  kKvCompress,
  kIndexScore,
  kTopK,
  kReshape,
  kPack,    // 按 input_start_index 把 [B,S_max,*rest] 压成 [M,*rest]。
  kUnpack,  // 按 input_start_index 把 [M,*rest] 还原成 [B,S_max,*rest]。
};

// 返回 op 类型对应的稳定字符串名称。
//
// 参数：
//   op_type：op 类型。
// 返回：
//   小写下划线形式的常量字符串，例如 "mega_attention"；未知值返回 "unknown"。
const char* OpTypeToStr(OpType op_type);

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_OP_TYPE_H_
