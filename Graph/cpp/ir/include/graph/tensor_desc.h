// 模块：Graph IR 到 TensorDesc 的转换边界
//
// 职责：
//   把 at::Tensor + SymbolicShape 的元数据转换为 Runtime 侧 TensorDesc；
//   提供 Graph IR 到 RuntimeGraph 的整体转换入口；提供序列化前的张量规范化
//   （仅 LibTorch 构建可用）。
//
// 边界：
//   - ToTensorDesc 只转换元数据，不执行 detach/拷贝/连续化/转置；
//   - Serializer 必须在调用前完成 detach、CPU、连续行主序与 2-D 列主序转置；
//   - RuntimeGraph 不链接 LibTorch。
//
// 依赖：
//   graph.h、runtime_graph_types.h、validation_common.h。

#ifndef MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_TENSOR_DESC_H_
#define MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_TENSOR_DESC_H_

#include <ATen/ATen.h>

#include <memory>
#include <optional>
#include <string>

#include "graph/graph.h"
#include "graph/runtime_graph_types.h"
#include "graph/validation_common.h"

namespace mobilenet {
namespace graph {

class RuntimeGraph;

// at::ScalarType -> DataType。
DataType ToDataType(at::ScalarType scalar_type);

// DataType -> at::ScalarType；未知返回 Undefined。
at::ScalarType ToScalarType(DataType data_type);

// 返回规范 dtype 名称，例如 "float16"；未知返回 "unknown"。
const char* ScalarTypeName(at::ScalarType scalar_type);

// 构造全常量符号形状（维度取 tensor.sizes()）。
SymbolicShape ConcreteSymbolicShape(const at::Tensor& tensor);

// 生成校验/JSON 使用的张量视图。
//
// 参数：
//   tensor：已定义的 at::Tensor；
//   shape：符号形状，必须与 tensor.sizes() 一致；
//   quantization：可选量化描述。
// 返回：
//   ValidationTensor；tensor 未定义时 defined=false，其余字段尽量填默认值。
ValidationTensor DescribeTensor(
    const at::Tensor& tensor, const SymbolicShape& shape,
    std::optional<QuantizationDesc> quantization = std::nullopt);

// 转换元数据并检查一致性。
//
// 参数：
//   tensor：输入张量；
//   shape：符号形状；
//   quantization：可选量化描述；
//   out：输出 TensorDesc。
// 返回：
//   Status::Ok()；失败返回 INVALID_ARGUMENT/UNSUPPORTED_DTYPE/LAYOUT。
Status ToTensorDesc(const at::Tensor& tensor, const SymbolicShape& shape,
                    std::optional<QuantizationDesc> quantization,
                    TensorDesc* out);

// 值返回版本；输入非法时返回 dtype=kUnknown 的空描述，调用方应优先使用
// Status 重载做错误处理。
TensorDesc ToTensorDesc(
    const at::Tensor& tensor, const SymbolicShape& shape,
    std::optional<QuantizationDesc> quantization = std::nullopt);

// Graph IR -> RuntimeGraph。
//
// 行为：
//   - 转换节点、边、权重与模型配置；
//   - 不分配 arena；所有 RuntimeEdge.arena_id 保持 -1，由 Loader 绑定；
//   - 不修改源 Graph。
//
// 参数：
//   graph：源图；
//   out：输出 RuntimeGraph。
// 返回：
//   Status::Ok() 或转换错误。
Status ConvertGraphToRuntimeGraph(const Graph& graph,
                                  std::unique_ptr<RuntimeGraph>* out);

#ifdef MOBILENET_GRAPH_USE_LIBTORCH

// 序列化前张量规范化（仅链接 LibTorch 的工具链构建可用）。
//
// 行为：
//   detach -> CPU -> 连续；若为 2-D 列主序权重则转置为行主序。
//
// 参数：
//   tensor：源张量。
// 返回：
//   规范化后的 CPU 连续张量。
at::Tensor NormalizeForSerialization(const at::Tensor& tensor);

#endif  // MOBILENET_GRAPH_USE_LIBTORCH

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_TENSOR_DESC_H_
