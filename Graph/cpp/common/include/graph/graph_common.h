// 模块：Graph 公共类型（Graph IR 与 RuntimeGraph 共享）
//
// 职责：
//   定义节点/边标识、符号维度、量化描述、融合算子描述、模型配置模板与
//   校验报告等与具体张量表示无关的公共数据结构。
//
// 边界：
//   - 本文件不包含 at::Tensor、at::ScalarType 或任何 LibTorch 头文件；
//   - Graph IR 的 at::Tensor 属性在 graph_types.h 中定义；
//   - RuntimeGraph 的 TensorDesc 在 runtime_graph_types.h 中定义；
//   - DataType/Layout/DeviceType 来自 Runtime/common/data_types.h，只用于
//     量化元数据与 RuntimeGraph，不作为 Graph IR 的张量属性类型。
//
// 依赖：
//   Runtime/common/data_types.h、Runtime/common/status.h、
//   Runtime/kernel/kernel_types.h。

#ifndef MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_GRAPH_COMMON_H_
#define MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_GRAPH_COMMON_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "Runtime/common/data_types.h"
#include "Runtime/common/status.h"
#include "Runtime/kernel/kernel_types.h"

namespace mobilenet {
namespace graph {

// 节点标识；0xFFFFFFFF 表示无效。
using NodeId = uint32_t;

// 边标识；0xFFFFFFFF 表示无效。
using EdgeId = uint32_t;

// 无效节点标识。
constexpr NodeId kInvalidNodeId = 0xFFFFFFFFu;

// 无效边标识。
constexpr EdgeId kInvalidEdgeId = 0xFFFFFFFFu;

// 节点/边/绑定名最大长度。
constexpr size_t kMaxGraphNameLength = 128;

// 属性值。
//
// 支持：bool、int64、double、string、int64 数组、double 数组、string 数组。
using AttributeValue =
    std::variant<bool, int64_t, double, std::string, std::vector<int64_t>,
                 std::vector<double>, std::vector<std::string>>;

// 符号维度。
//
// 约定：
//   - is_symbolic == false 时，value 为常量维，symbol 必须为空；
//   - is_symbolic == true 时，symbol 为符号名，value 为示例值（用于生成
//     示例 Tensor），且 value >= 0。
struct Dim {
  bool is_symbolic = false;
  int64_t value = 0;
  std::string symbol;

  // 构造常量维。
  static Dim Constant(int64_t value) { return Dim{false, value, ""}; }

  // 构造符号维。
  //
  // 参数：
  //   symbol：非空符号名；
  //   example_value：示例值，必须 >= 0。
  static Dim Symbolic(std::string symbol, int64_t example_value) {
    return Dim{true, example_value, std::move(symbol)};
  }
};

// 符号形状。
//
// 功能：
//   描述一个张量每个维度的常量/符号属性，并给出示例形状。
//
// 边界：
//   - 不携带 dtype/device/layout；
//   - 权重必须全部为常量维；激活可含符号维。
struct SymbolicShape {
  std::vector<Dim> dims;

  SymbolicShape() = default;

  // 从维度列表构造。
  explicit SymbolicShape(std::vector<Dim> values) : dims(std::move(values)) {}

  // 从初始化列表构造。
  SymbolicShape(std::initializer_list<Dim> values) : dims(values) {}

  // 秩。
  size_t Rank() const { return dims.size(); }

  // 是否所有维度均为常量维。
  bool IsFullyConcrete() const {
    for (const Dim& dim : dims) {
      if (dim.is_symbolic) {
        return false;
      }
    }
    return true;
  }

  // 返回示例形状（每个维度的 value）。
  std::vector<int64_t> ExampleShape() const {
    std::vector<int64_t> shape;
    shape.reserve(dims.size());
    for (const Dim& dim : dims) {
      shape.push_back(dim.value);
    }
    return shape;
  }
};

// 量化粒度。
enum class QuantGranularity {
  kNone = 0,
  kPerTensor = 1,
  kPerChannel = 2,
  kPerGroup = 3,
};

// 返回量化粒度稳定名称，例如 "per_tensor"。
const char* QuantGranularityName(QuantGranularity granularity);

// 量化描述。
//
// 功能：
//   作为边/权重的可选元数据；不改变 TensorDesc 的 dtype，只说明存储量化方式。
//
// 边界：
//   - 非量化图必须保持 std::nullopt；
//   - scales/zero_points 的具体布局由 Kernel 与 Serializer 约定。
struct QuantizationDesc {
  DataType storage_dtype = DataType::kUnknown;
  QuantGranularity granularity = QuantGranularity::kNone;
  std::vector<double> scales;
  std::vector<int64_t> zero_points;
  int32_t group_size = 0;
};

// 融合前的原始算子描述。
//
// 字段：
//   source_name：原始模块/参数路径，例如 "model.layers.0.self_attn.q_proj"；
//   op_kind：原始算子种类，例如 "rmsnorm"、"q_linear"；
//   attributes：原始算子属性，可为空。
struct FusedOp {
  std::string source_name;
  std::string op_kind;
  std::map<std::string, AttributeValue> attributes;
};

// 校验问题。
struct ValidationIssue {
  ErrorCode code = ErrorCode::kInvalidArgument;
  std::string message;
  std::string node_name;
  std::string edge_name;
  std::string binding_name;
  std::string rule_id;
};

// 校验报告。
//
// 功能：
//   收集一次校验的全部问题；最多保留 32 条，超出后截断并追加提示。
class ValidationReport {
 public:
  // 是否无问题。
  bool ok() const { return issues_.empty(); }

  // 问题列表。
  const std::vector<ValidationIssue>& issues() const { return issues_; }

  // 问题数量。
  size_t size() const { return issues_.size(); }

  // 添加问题。
  //
  // 参数：
  //   issue：待添加问题；超过上限时忽略并保证最后一条为截断提示。
  void AddIssue(ValidationIssue issue);

  // 清空。
  void Clear() { issues_.clear(); }

  // 生成人类可读摘要，用于日志与测试输出。
  std::string ToString() const;

 private:
  static constexpr size_t kMaxIssues = 32;
  std::vector<ValidationIssue> issues_;
};

// RoPE 配置。
struct RopeConfig {
  double theta = 10000.0;
  RopeLayout layout = RopeLayout::kNeox;
  std::string scaling_type = "none";
  double scaling_factor = 1.0;
  int32_t original_max_position_embeddings = 0;
};

// 模型配置模板。
//
// 模板参数 DTypeT：
//   - Graph IR 使用 at::ScalarType；
//   - RuntimeGraph 使用 mobilenet::DataType。
//
// 边界：
//   只描述模型结构参数，不包含权重或张量。
template <typename DTypeT>
struct ModelConfigT {
  std::string model_family;
  int32_t num_layers = 0;
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  int32_t num_attention_heads = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int32_t vocab_size = 0;
  int32_t max_context_length = 0;
  double norm_eps = 0.0;
  std::string activation;
  bool tie_word_embeddings = false;
  DTypeT dtype{};
  RopeConfig rope;
  std::map<std::string, AttributeValue> extra;
};

// 判断名称是否合法。
//
// 规则：
//   - 长度 1..128；
//   - 只允许 ASCII 字母、数字、下划线、点、连字符；
//   - 不允许以连续两个点开头等特殊语义；本函数只做字符集与长度检查。
//
// 参数：
//   name：待检查名称。
// 返回：
//   合法返回 true。
bool IsValidGraphName(std::string_view name);

// 返回 AttributeValue 的 JSON 类型名，用于诊断。
const char* AttributeTypeName(const AttributeValue& value);

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_GRAPH_COMMON_H_
