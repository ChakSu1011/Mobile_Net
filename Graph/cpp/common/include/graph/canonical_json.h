// 模块：Graph 规范 JSON 片段写入
//
// 职责：
//   为 Graph IR 与 RuntimeGraph 提供字段顺序完全一致的张量、量化、属性、
//   融合算子与模型配置 JSON 写入函数，保证两者可逐字节比较。
//
// 边界：
//   - 不负责节点/边/权重的外层容器顺序；
//   - 不解析 JSON；
//   - 不访问张量数据。
//
// 依赖：
//   json_writer.h、validation_common.h、graph_common.h。

#ifndef MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_CANONICAL_JSON_H_
#define MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_CANONICAL_JSON_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "graph/graph_common.h"
#include "graph/json_writer.h"
#include "graph/validation_common.h"

namespace mobilenet {
namespace graph {

// 写入符号维度。
inline void WriteDimJson(const Dim& dim, JsonWriter* writer) {
  writer->BeginObject();
  writer->Key("symbolic");
  writer->Bool(dim.is_symbolic);
  writer->Key("value");
  writer->Int(dim.value);
  writer->Key("symbol");
  writer->String(dim.symbol);
  writer->EndObject();
}

// 写入量化描述或 null。
inline void WriteQuantizationJson(
    const std::optional<QuantizationDesc>& quantization, JsonWriter* writer) {
  if (!quantization.has_value()) {
    writer->Null();
    return;
  }
  const QuantizationDesc& value = *quantization;
  writer->BeginObject();
  writer->Key("storage_dtype");
  writer->String(DataTypeName(value.storage_dtype));
  writer->Key("granularity");
  writer->String(QuantGranularityName(value.granularity));
  writer->Key("scales");
  writer->BeginArray();
  for (double scale : value.scales) {
    writer->Double(scale);
  }
  writer->EndArray();
  writer->Key("zero_points");
  writer->BeginArray();
  for (int64_t zero_point : value.zero_points) {
    writer->Int(zero_point);
  }
  writer->EndArray();
  writer->Key("group_size");
  writer->Int(value.group_size);
  writer->EndObject();
}

// 写入单个属性值。
inline void WriteAttributeValueJson(const AttributeValue& value,
                                    JsonWriter* writer) {
  if (const auto* boolean = std::get_if<bool>(&value)) {
    writer->Bool(*boolean);
  } else if (const auto* integer = std::get_if<int64_t>(&value)) {
    writer->Int(*integer);
  } else if (const auto* number = std::get_if<double>(&value)) {
    writer->Double(*number);
  } else if (const auto* text = std::get_if<std::string>(&value)) {
    writer->String(*text);
  } else if (const auto* integers = std::get_if<std::vector<int64_t>>(&value)) {
    writer->BeginArray();
    for (int64_t integer : *integers) {
      writer->Int(integer);
    }
    writer->EndArray();
  } else if (const auto* numbers = std::get_if<std::vector<double>>(&value)) {
    writer->BeginArray();
    for (double number : *numbers) {
      writer->Double(number);
    }
    writer->EndArray();
  } else if (const auto* strings =
                 std::get_if<std::vector<std::string>>(&value)) {
    writer->BeginArray();
    for (const std::string& text : *strings) {
      writer->String(text);
    }
    writer->EndArray();
  } else {
    writer->Null();
  }
}

// 写入属性映射（按 key 升序）。
inline void WriteAttributesJson(
    const std::map<std::string, AttributeValue>& attributes,
    JsonWriter* writer) {
  writer->BeginObject();
  for (const auto& attribute : attributes) {
    writer->Key(attribute.first);
    WriteAttributeValueJson(attribute.second, writer);
  }
  writer->EndObject();
}

// 写入融合算子序列。
inline void WriteFusedOpsJson(const std::vector<FusedOp>& fused_ops,
                              JsonWriter* writer) {
  writer->BeginArray();
  for (const FusedOp& fused_op : fused_ops) {
    writer->BeginObject();
    writer->Key("source_name");
    writer->String(fused_op.source_name);
    writer->Key("op_kind");
    writer->String(fused_op.op_kind);
    writer->Key("attributes");
    WriteAttributesJson(fused_op.attributes, writer);
    writer->EndObject();
  }
  writer->EndArray();
}

// 写入张量元数据（IR 与 RuntimeGraph 共用字段顺序）。
inline void WriteTensorJson(const ValidationTensor& tensor,
                            JsonWriter* writer) {
  writer->BeginObject();
  writer->Key("dtype");
  writer->String(tensor.dtype);
  writer->Key("shape");
  writer->BeginArray();
  for (const Dim& dim : tensor.shape) {
    WriteDimJson(dim, writer);
  }
  writer->EndArray();
  writer->Key("stride");
  writer->BeginArray();
  for (int64_t stride : tensor.stride) {
    writer->Int(stride);
  }
  writer->EndArray();
  writer->Key("layout");
  writer->String(LayoutName(tensor.layout));
  writer->Key("device");
  writer->String(DeviceTypeName(tensor.device));
  writer->Key("offset");
  writer->UInt(tensor.offset);
  writer->Key("size_bytes");
  writer->UInt(tensor.size_bytes);
  writer->Key("quantization");
  WriteQuantizationJson(tensor.quantization, writer);
  writer->EndObject();
}

// 写入 RoPE 配置。
inline void WriteRopeConfigJson(const RopeConfig& rope, JsonWriter* writer) {
  writer->BeginObject();
  writer->Key("theta");
  writer->Double(rope.theta);
  writer->Key("layout");
  writer->String(RopeLayoutName(rope.layout));
  writer->Key("scaling_type");
  writer->String(rope.scaling_type);
  writer->Key("scaling_factor");
  writer->Double(rope.scaling_factor);
  writer->Key("original_max_position_embeddings");
  writer->Int(rope.original_max_position_embeddings);
  writer->EndObject();
}

// 写入模型配置。
inline void WriteModelConfigJson(const ValidationModelConfig& config,
                                 JsonWriter* writer) {
  writer->BeginObject();
  writer->Key("model_family");
  writer->String(config.model_family);
  writer->Key("num_layers");
  writer->Int(config.num_layers);
  writer->Key("hidden_size");
  writer->Int(config.hidden_size);
  writer->Key("intermediate_size");
  writer->Int(config.intermediate_size);
  writer->Key("num_attention_heads");
  writer->Int(config.num_attention_heads);
  writer->Key("num_kv_heads");
  writer->Int(config.num_kv_heads);
  writer->Key("head_dim");
  writer->Int(config.head_dim);
  writer->Key("vocab_size");
  writer->Int(config.vocab_size);
  writer->Key("max_context_length");
  writer->Int(config.max_context_length);
  writer->Key("norm_eps");
  writer->Double(config.norm_eps);
  writer->Key("activation");
  writer->String(config.activation);
  writer->Key("tie_word_embeddings");
  writer->Bool(config.tie_word_embeddings);
  writer->Key("dtype");
  writer->String(config.dtype);
  writer->Key("rope");
  WriteRopeConfigJson(config.rope, writer);
  writer->Key("extra");
  WriteAttributesJson(config.extra, writer);
  writer->EndObject();
}

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_CANONICAL_JSON_H_
