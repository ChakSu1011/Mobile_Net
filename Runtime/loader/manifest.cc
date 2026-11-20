// 模块：Runtime 模型包 manifest 视图实现
//
// 职责：
//   实现固定头解析、manifest JSON 视图构建、引用完整性/统计交叉校验、
//   ModelConfig 提取与名称转换。
//
// 边界：
//   - 不读取 weights.bin 内容、不创建 Graph、不分配设备资源；
//   - 校验失败统一返回 kInvalidModelFile，message 指出 manifest 路径；
//   - 只接受静态整数维度；symbol 维度直接拒绝。
//
// 依赖：
//   Runtime/loader/manifest.h、Runtime/common/{file_util,json}.h、
//   Graph/include/op_type.h、C++17 标准库。

#include "Runtime/loader/manifest.h"

#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Graph/include/op_type.h"
#include "Graph/include/types.h"
#include "Runtime/common/data_types.h"
#include "Runtime/common/file_util.h"
#include "Runtime/common/json.h"
#include "Runtime/common/model_config.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// 构造 manifest 校验错误。
//
// 参数：
//   detail：错误原因与位置。
// 返回：
//   kInvalidModelFile 状态。
Status ManifestError(const std::string& detail) {
  return MakeStatus(ErrorCode::kInvalidModelFile,
                    "manifest: " + detail);
}

// 读取小端 32 位整数。
//
// 参数：
//   data：至少 4 字节。
// 返回：
//   无符号值。
U32 ReadU32Le(const U8* data) {
  return static_cast<U32>(data[0]) |
         (static_cast<U32>(data[1]) << 8) |
         (static_cast<U32>(data[2]) << 16) |
         (static_cast<U32>(data[3]) << 24);
}

// 读取小端 64 位整数。
//
// 参数：
//   data：至少 8 字节。
// 返回：
//   无符号值。
U64 ReadU64Le(const U8* data) {
  U64 value = 0;
  for (Usize index = 0; index < 8; ++index) {
    value |= static_cast<U64>(data[index]) << (index * 8);
  }
  return value;
}

// 判断字符串是否为 64 位十六进制。
//
// 参数：
//   text：待检查文本。
// 返回：
//   true 表示长度为 64 且全部字符为十六进制。
Bool IsSha256Hex(const std::string& text) {
  if (text.size() != 64) {
    return false;
  }
  for (const C8 ch : text) {
    const Bool digit = ch >= '0' && ch <= '9';
    const Bool lower = ch >= 'a' && ch <= 'f';
    const Bool upper = ch >= 'A' && ch <= 'F';
    if (!digit && !lower && !upper) {
      return false;
    }
  }
  return true;
}

// 读取必需的对象成员。
//
// 参数：
//   object：父对象；
//   key：成员名；
//   context：错误位置前缀；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；成员缺失返回 kInvalidModelFile。
Status RequireMember(const JsonValue& object, const std::string& key,
                     const std::string& context,
                     const JsonValue** out) {
  *out = object.Find(key);
  if (*out == nullptr) {
    return ManifestError(context + "." + key + " is required");
  }
  return Status::Ok();
}

// 要求值为对象。
//
// 参数：
//   value：待检查值；
//   context：错误位置。
// 返回：
//   成功返回 kOk；类型不符返回 kInvalidModelFile。
Status RequireObject(const JsonValue& value, const std::string& context) {
  if (!value.IsObject()) {
    return ManifestError(context + " must be an object");
  }
  return Status::Ok();
}

// 要求值为数组。
//
// 参数：
//   value：待检查值；
//   context：错误位置。
// 返回：
//   成功返回 kOk；类型不符返回 kInvalidModelFile。
Status RequireArray(const JsonValue& value, const std::string& context) {
  if (!value.IsArray()) {
    return ManifestError(context + " must be an array");
  }
  return Status::Ok();
}

// 读取字符串值。
//
// 参数：
//   value：待读取值；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；类型不符返回 kInvalidModelFile。
Status ReadString(const JsonValue& value, const std::string& context,
                  std::string* out) {
  if (!value.IsString()) {
    return ManifestError(context + " must be a string");
  }
  *out = value.AsString();
  return Status::Ok();
}

// 读取非空字符串值。
//
// 参数：
//   value：待读取值；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；类型不符或空串返回 kInvalidModelFile。
Status ReadNonEmptyString(const JsonValue& value,
                          const std::string& context, std::string* out) {
  Status status = ReadString(value, context, out);
  if (!status.ok()) {
    return status;
  }
  if (out->empty()) {
    return ManifestError(context + " must not be empty");
  }
  return Status::Ok();
}

// 读取整数值。
//
// 参数：
//   value：待读取值；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；非整数返回 kInvalidModelFile。
Status ReadInt(const JsonValue& value, const std::string& context,
               I64* out) {
  if (!value.IsInteger()) {
    return ManifestError(context + " must be an integer");
  }
  *out = value.AsInt64();
  return Status::Ok();
}

// 读取非负整数。
//
// 参数：
//   value：待读取值；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；非整数或负数返回 kInvalidModelFile。
Status ReadUint(const JsonValue& value, const std::string& context,
                U64* out) {
  I64 signed_value = 0;
  Status status = ReadInt(value, context, &signed_value);
  if (!status.ok()) {
    return status;
  }
  if (signed_value < 0) {
    return ManifestError(context + " must not be negative");
  }
  *out = static_cast<U64>(signed_value);
  return Status::Ok();
}

// 读取布尔值。
//
// 参数：
//   value：待读取值；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；非布尔返回 kInvalidModelFile。
Status ReadBool(const JsonValue& value, const std::string& context,
                Bool* out) {
  if (!value.IsBool()) {
    return ManifestError(context + " must be a boolean");
  }
  *out = value.AsBool();
  return Status::Ok();
}

// 读取数字为 FP64（整数自动提升）。
//
// 参数：
//   value：待读取值；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；非数字返回 kInvalidModelFile。
Status ReadNumber(const JsonValue& value, const std::string& context,
                  FP64* out) {
  if (!value.IsNumber()) {
    return ManifestError(context + " must be a number");
  }
  *out = value.AsFloat64();
  return Status::Ok();
}

// 读取整数数组。
//
// 参数：
//   value：待读取值；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；非数组或元素非整数返回 kInvalidModelFile。
Status ReadIntArray(const JsonValue& value, const std::string& context,
                    std::vector<I64>* out) {
  Status status = RequireArray(value, context);
  if (!status.ok()) {
    return status;
  }
  out->clear();
  const std::vector<JsonValue>& items = value.AsArray();
  out->reserve(items.size());
  for (Usize index = 0; index < items.size(); ++index) {
    I64 item = 0;
    status = ReadInt(items[index],
                     context + "[" + std::to_string(index) + "]", &item);
    if (!status.ok()) {
      return status;
    }
    out->push_back(item);
  }
  return Status::Ok();
}

// 读取字符串数组，可要求元素非空。
//
// 参数：
//   value：待读取值；
//   context：错误位置；
//   require_non_empty：是否要求每个元素非空；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；类型不符返回 kInvalidModelFile。
Status ReadStringArray(const JsonValue& value, const std::string& context,
                       Bool require_non_empty,
                       std::vector<std::string>* out) {
  Status status = RequireArray(value, context);
  if (!status.ok()) {
    return status;
  }
  out->clear();
  const std::vector<JsonValue>& items = value.AsArray();
  out->reserve(items.size());
  for (Usize index = 0; index < items.size(); ++index) {
    std::string item;
    const std::string item_context =
        context + "[" + std::to_string(index) + "]";
    status = require_non_empty
                 ? ReadNonEmptyString(items[index], item_context, &item)
                 : ReadString(items[index], item_context, &item);
    if (!status.ok()) {
      return status;
    }
    out->push_back(std::move(item));
  }
  return Status::Ok();
}

// 把 JSON 属性值转换为 Graph AttrValue。
//
// 参数：
//   value：属性值；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；null/对象/混合数组等不可表示的值返回 kInvalidModelFile。
Status JsonAttrToValue(const JsonValue& value, const std::string& context,
                       runtime_graph::AttrValue* out) {
  switch (value.type()) {
    case JsonValue::Type::kBool:
      *out = value.AsBool();
      return Status::Ok();
    case JsonValue::Type::kNumber:
      if (value.IsInteger()) {
        *out = value.AsInt64();
      } else {
        *out = value.AsFloat64();
      }
      return Status::Ok();
    case JsonValue::Type::kString:
      *out = value.AsString();
      return Status::Ok();
    case JsonValue::Type::kArray: {
      Bool has_int = false;
      Bool has_float = false;
      Bool has_string = false;
      const std::vector<JsonValue>& items = value.AsArray();
      for (Usize index = 0; index < items.size(); ++index) {
        const JsonValue& item = items[index];
        if (item.IsBool()) {
          has_int = true;
        } else if (item.IsString()) {
          has_string = true;
        } else if (item.IsInteger()) {
          has_int = true;
        } else if (item.IsNumber()) {
          has_float = true;
        } else {
          return ManifestError(
              context + "[" + std::to_string(index) +
              "] has unsupported array element type");
        }
      }
      if (has_string && (has_int || has_float)) {
        return ManifestError(context +
                             " must not mix strings and numbers");
      }
      if (has_string) {
        std::vector<std::string> strings;
        strings.reserve(items.size());
        for (const JsonValue& item : items) {
          strings.push_back(item.AsString());
        }
        *out = std::move(strings);
        return Status::Ok();
      }
      if (has_float) {
        std::vector<FP64> numbers;
        numbers.reserve(items.size());
        for (const JsonValue& item : items) {
          numbers.push_back(item.IsBool()
                                ? (item.AsBool() ? 1.0 : 0.0)
                                : item.AsFloat64());
        }
        *out = std::move(numbers);
        return Status::Ok();
      }
      std::vector<I64> integers;
      integers.reserve(items.size());
      for (const JsonValue& item : items) {
        integers.push_back(item.IsBool()
                               ? (item.AsBool() ? 1 : 0)
                               : item.AsInt64());
      }
      *out = std::move(integers);
      return Status::Ok();
    }
    case JsonValue::Type::kNull:
      return ManifestError(context + " must not be null");
    case JsonValue::Type::kObject:
      return ManifestError(context + " must not be an object");
  }
  return ManifestError(context + " has unsupported value type");
}

// 按名称读取张量元数据。
//
// 参数：
//   tensors：张量表；
//   name：张量名；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；不存在返回 kInvalidModelFile。
Status FindTensor(const std::map<std::string, TensorMeta>& tensors,
                  const std::string& name, const std::string& context,
                  const TensorMeta** out) {
  const auto it = tensors.find(name);
  if (it == tensors.end()) {
    return ManifestError(context + " references unknown tensor: " + name);
  }
  *out = &it->second;
  return Status::Ok();
}

// 解析张量类别名称。
//
// 参数：
//   name：类别字符串；
//   kind：输出参数，非空。
// 返回：
//   成功返回 kOk；未知类别返回 kInvalidModelFile。
Status TensorKindFromName(const std::string& name, TensorKind* kind) {
  if (name == "input") {
    *kind = TensorKind::kInput;
    return Status::Ok();
  }
  if (name == "constant") {
    *kind = TensorKind::kConstant;
    return Status::Ok();
  }
  if (name == "activation") {
    *kind = TensorKind::kActivation;
    return Status::Ok();
  }
  if (name == "output") {
    *kind = TensorKind::kOutput;
    return Status::Ok();
  }
  return ManifestError("unsupported tensor kind: " + name);
}

// 计算静态 shape 的元素个数。
//
// 参数：
//   dims：逻辑维度；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；维度为负、溢出或空维度返回 kInvalidModelFile。
Status ElementCount(const std::vector<I64>& dims,
                    const std::string& context, U64* out) {
  U64 count = 1;
  for (Usize index = 0; index < dims.size(); ++index) {
    if (dims[index] < 0) {
      return ManifestError(context + " has negative dimension");
    }
    const U64 dim = static_cast<U64>(dims[index]);
    if (dim != 0 && count > std::numeric_limits<U64>::max() / dim) {
      return ManifestError(context + " element count overflows");
    }
    count *= dim;
  }
  *out = count;
  return Status::Ok();
}

// 解析 graph.tensors 数组。
//
// 参数：
//   value：tensors 数组；
//   context：错误位置；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；字段非法或名称重复返回 kInvalidModelFile。
Status ParseTensors(const JsonValue& value, const std::string& context,
                    std::map<std::string, TensorMeta>* out) {
  Status status = RequireArray(value, context);
  if (!status.ok()) {
    return status;
  }
  out->clear();
  const std::vector<JsonValue>& items = value.AsArray();
  for (Usize index = 0; index < items.size(); ++index) {
    const JsonValue& item = items[index];
    const std::string item_context =
        context + "[" + std::to_string(index) + "]";
    status = RequireObject(item, item_context);
    if (!status.ok()) {
      return status;
    }
    TensorMeta meta;
    const JsonValue* field = nullptr;
    status = RequireMember(item, "name", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadNonEmptyString(*field, item_context + ".name", &meta.name);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "kind", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    std::string kind_name;
    status = ReadString(*field, item_context + ".kind", &kind_name);
    if (!status.ok()) {
      return status;
    }
    status = TensorKindFromName(kind_name, &meta.kind);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "shape", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadIntArray(*field, item_context + ".shape", &meta.dims);
    if (!status.ok()) {
      return status;
    }
    if (meta.dims.size() > static_cast<Usize>(runtime_graph::kMaxShapeDim)) {
      return ManifestError(item_context +
                           ".shape exceeds kMaxShapeDim");
    }
    field = item.Find("pitch");
    if (field != nullptr) {
      status = ReadIntArray(*field, item_context + ".pitch", &meta.pitch);
      if (!status.ok()) {
        return status;
      }
      // 空 pitch 表示与 shape 相同（与 ToolChain MakeStorage 语义一致）。
      if (!meta.pitch.empty() &&
          meta.pitch.size() != meta.dims.size()) {
        return ManifestError(item_context +
                             ".pitch length must match shape");
      }
    }
    status = RequireMember(item, "precision", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    std::string precision_name;
    status = ReadString(*field, item_context + ".precision",
                        &precision_name);
    if (!status.ok()) {
      return status;
    }
    status = PrecisionFromName(precision_name, &meta.precision);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "layout", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    std::string layout_name;
    status = ReadString(*field, item_context + ".layout", &layout_name);
    if (!status.ok()) {
      return status;
    }
    status = LayoutFromName(layout_name, &meta.layout);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "byte_size", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadUint(*field, item_context + ".byte_size", &meta.byte_size);
    if (!status.ok()) {
      return status;
    }
    field = item.Find("weight_available");
    if (field != nullptr) {
      status = ReadBool(*field, item_context + ".weight_available",
                        &meta.weight_available);
      if (!status.ok()) {
        return status;
      }
    }
    if (out->find(meta.name) != out->end()) {
      return ManifestError(item_context + " has duplicate tensor name: " +
                           meta.name);
    }
    const std::string name = meta.name;
    out->emplace(name, std::move(meta));
  }
  return Status::Ok();
}

// 解析 manifest 的 ops 数组。
//
// 参数：
//   value：ops 数组；
//   tensors：张量表；
//   context：错误位置；
//   ops_out：输出 op 列表；
//   producers_out：输出 activation 张量到 op 名称的映射。
// 返回：
//   成功返回 kOk；字段非法、引用缺失或重复生产返回 kInvalidModelFile。
Status ParseOps(const JsonValue& value,
                const std::map<std::string, TensorMeta>& tensors,
                const std::string& context, std::vector<OpMeta>* ops_out,
                std::map<std::string, std::string>* producers_out) {
  Status status = RequireArray(value, context);
  if (!status.ok()) {
    return status;
  }
  ops_out->clear();
  producers_out->clear();
  std::set<std::string> op_names;
  const std::vector<JsonValue>& items = value.AsArray();
  for (Usize index = 0; index < items.size(); ++index) {
    const JsonValue& item = items[index];
    const std::string item_context =
        context + "[" + std::to_string(index) + "]";
    status = RequireObject(item, item_context);
    if (!status.ok()) {
      return status;
    }
    OpMeta op;
    const JsonValue* field = nullptr;
    status = RequireMember(item, "name", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadNonEmptyString(*field, item_context + ".name", &op.name);
    if (!status.ok()) {
      return status;
    }
    if (!op_names.insert(op.name).second) {
      return ManifestError(item_context + " has duplicate op name: " +
                           op.name);
    }
    status = RequireMember(item, "op_type", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadNonEmptyString(*field, item_context + ".op_type",
                                &op.op_type);
    if (!status.ok()) {
      return status;
    }
    runtime_graph::OpType op_type = runtime_graph::OpType::kUnknown;
    status = OpTypeFromName(op.op_type, &op_type);
    if (!status.ok()) {
      return status;
    }
    if (op_type == runtime_graph::OpType::kUnknown ||
        op_type == runtime_graph::OpType::kInput ||
        op_type == runtime_graph::OpType::kOutput) {
      return ManifestError(item_context +
                           " has non-compute op_type: " + op.op_type);
    }
    status = RequireMember(item, "inputs", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadStringArray(*field, item_context + ".inputs", true,
                             &op.inputs);
    if (!status.ok()) {
      return status;
    }
    if (op.inputs.empty()) {
      return ManifestError(item_context + ".inputs must not be empty");
    }
    status = RequireMember(item, "outputs", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadStringArray(*field, item_context + ".outputs", true,
                             &op.outputs);
    if (!status.ok()) {
      return status;
    }
    if (op.outputs.empty()) {
      return ManifestError(item_context + ".outputs must not be empty");
    }
    field = item.Find("attrs");
    if (field != nullptr) {
      status = RequireObject(*field, item_context + ".attrs");
      if (!status.ok()) {
        return status;
      }
      const std::vector<JsonMember>& members = field->AsObject();
      for (const JsonMember& member : members) {
        runtime_graph::AttrValue attr;
        status = JsonAttrToValue(
            member.value,
            item_context + ".attrs." + member.key, &attr);
        if (!status.ok()) {
          return status;
        }
        op.attrs.emplace(member.key, std::move(attr));
      }
    }
    for (const std::string& name : op.inputs) {
      const TensorMeta* meta = nullptr;
      status = FindTensor(tensors, name, item_context + ".inputs", &meta);
      if (!status.ok()) {
        return status;
      }
      if (meta->kind == TensorKind::kOutput) {
        return ManifestError(item_context +
                             ".inputs references output kind tensor: " +
                             name);
      }
    }
    for (const std::string& name : op.outputs) {
      const TensorMeta* meta = nullptr;
      status = FindTensor(tensors, name, item_context + ".outputs", &meta);
      if (!status.ok()) {
        return status;
      }
      if (meta->kind != TensorKind::kActivation) {
        return ManifestError(item_context +
                             ".outputs must be activation tensor: " + name);
      }
      if (!producers_out->emplace(name, op.name).second) {
        return ManifestError(item_context +
                             " has tensor with multiple producers: " + name);
      }
    }
    ops_out->push_back(std::move(op));
  }
  return Status::Ok();
}

// 解析 optimization 段。
//
// 参数：
//   value：optimization 对象；
//   context：错误位置；
//   config_out：输出 config JsonValue；
//   plan_out：输出 memory plan。
// 返回：
//   成功返回 kOk；字段非法返回 kInvalidModelFile。
Status ParseOptimization(const JsonValue& value, const std::string& context,
                         JsonValue* config_out, MemoryPlan* plan_out) {
  Status status = RequireObject(value, context);
  if (!status.ok()) {
    return status;
  }
  const JsonValue* config = nullptr;
  status = RequireMember(value, "config", context, &config);
  if (!status.ok()) {
    return status;
  }
  status = RequireObject(*config, context + ".config");
  if (!status.ok()) {
    return status;
  }
  *config_out = *config;

  const JsonValue* plan = nullptr;
  status = RequireMember(value, "memory_plan", context, &plan);
  if (!status.ok()) {
    return status;
  }
  status = RequireObject(*plan, context + ".memory_plan");
  if (!status.ok()) {
    return status;
  }
  const std::string plan_context = context + ".memory_plan";
  const JsonValue* field = nullptr;
  status = RequireMember(*plan, "alignment", plan_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, plan_context + ".alignment",
                    &plan_out->alignment);
  if (!status.ok()) {
    return status;
  }
  if (plan_out->alignment == 0) {
    return ManifestError(plan_context + ".alignment must be positive");
  }
  status = RequireMember(*plan, "arena_bytes", plan_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, plan_context + ".arena_bytes",
                    &plan_out->arena_bytes);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(*plan, "peak_bytes", plan_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, plan_context + ".peak_bytes",
                    &plan_out->peak_bytes);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(*plan, "constants_bytes", plan_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, plan_context + ".constants_bytes",
                    &plan_out->constants_bytes);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(*plan, "entries", plan_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = RequireArray(*field, plan_context + ".entries");
  if (!status.ok()) {
    return status;
  }
  plan_out->entries.clear();
  std::set<std::string> entry_names;
  const std::vector<JsonValue>& items = field->AsArray();
  for (Usize index = 0; index < items.size(); ++index) {
    const JsonValue& item = items[index];
    const std::string item_context =
        plan_context + ".entries[" + std::to_string(index) + "]";
    status = RequireObject(item, item_context);
    if (!status.ok()) {
      return status;
    }
    MemoryPlanEntry entry;
    status = RequireMember(item, "name", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadNonEmptyString(*field, item_context + ".name", &entry.name);
    if (!status.ok()) {
      return status;
    }
    if (!entry_names.insert(entry.name).second) {
      return ManifestError(item_context +
                           " has duplicate memory entry: " + entry.name);
    }
    status = RequireMember(item, "offset", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadUint(*field, item_context + ".offset", &entry.offset);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "size", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadUint(*field, item_context + ".size", &entry.size);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "birth", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadInt(*field, item_context + ".birth", &entry.birth);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "last_use", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadInt(*field, item_context + ".last_use", &entry.last_use);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "reusable", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadBool(*field, item_context + ".reusable", &entry.reusable);
    if (!status.ok()) {
      return status;
    }
    if (entry.birth > entry.last_use) {
      return ManifestError(item_context +
                           ".birth must not exceed last_use");
    }
    if (entry.offset > plan_out->arena_bytes ||
        entry.size > plan_out->arena_bytes - entry.offset) {
      return ManifestError(item_context + " exceeds arena_bytes");
    }
    plan_out->entries.push_back(std::move(entry));
  }
  return Status::Ok();
}

// 解析 weights 段。
//
// 参数：
//   value：weights 对象；
//   context：错误位置；
//   out：输出权重表。
// 返回：
//   成功返回 kOk；字段非法或条目重复返回 kInvalidModelFile。
Status ParseWeights(const JsonValue& value, const std::string& context,
                    WeightTable* out) {
  Status status = RequireObject(value, context);
  if (!status.ok()) {
    return status;
  }
  out->file = "weights.bin";
  const JsonValue* field = value.Find("file");
  if (field != nullptr) {
    status = ReadNonEmptyString(*field, context + ".file", &out->file);
    if (!status.ok()) {
      return status;
    }
  }
  // 只允许包内简单文件名，禁止路径分隔符与父目录引用。
  if (out->file == "." || out->file == ".." ||
      out->file.find('/') != std::string::npos ||
      out->file.find('\\') != std::string::npos) {
    return ManifestError(context + ".file must be a simple file name");
  }
  status = RequireMember(value, "alignment", context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, context + ".alignment", &out->alignment);
  if (!status.ok()) {
    return status;
  }
  if (out->alignment == 0) {
    return ManifestError(context + ".alignment must be positive");
  }
  status = RequireMember(value, "total_bytes", context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, context + ".total_bytes", &out->total_bytes);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(value, "entries", context, &field);
  if (!status.ok()) {
    return status;
  }
  status = RequireArray(*field, context + ".entries");
  if (!status.ok()) {
    return status;
  }
  out->entries.clear();
  std::set<std::string> names;
  const std::vector<JsonValue>& items = field->AsArray();
  for (Usize index = 0; index < items.size(); ++index) {
    const JsonValue& item = items[index];
    const std::string item_context =
        context + ".entries[" + std::to_string(index) + "]";
    status = RequireObject(item, item_context);
    if (!status.ok()) {
      return status;
    }
    WeightEntry entry;
    status = RequireMember(item, "name", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadNonEmptyString(*field, item_context + ".name", &entry.name);
    if (!status.ok()) {
      return status;
    }
    if (!names.insert(entry.name).second) {
      return ManifestError(item_context +
                           " has duplicate weight entry: " + entry.name);
    }
    field = item.Find("source");
    if (field != nullptr && !field->IsNull()) {
      status = ReadString(*field, item_context + ".source", &entry.source);
      if (!status.ok()) {
        return status;
      }
    }
    status = RequireMember(item, "shape", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadIntArray(*field, item_context + ".shape", &entry.shape);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "dtype", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadNonEmptyString(*field, item_context + ".dtype",
                                &entry.dtype);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "byte_size", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadUint(*field, item_context + ".byte_size", &entry.byte_size);
    if (!status.ok()) {
      return status;
    }
    field = item.Find("offset");
    if (field != nullptr && !field->IsNull()) {
      status = ReadUint(*field, item_context + ".offset", &entry.offset);
      if (!status.ok()) {
        return status;
      }
      entry.has_offset = true;
    }
    field = item.Find("sha256");
    if (field != nullptr && !field->IsNull()) {
      status = ReadString(*field, item_context + ".sha256", &entry.sha256);
      if (!status.ok()) {
        return status;
      }
      if (!entry.sha256.empty() && !IsSha256Hex(entry.sha256)) {
        return ManifestError(item_context +
                             ".sha256 must be 64 hex characters");
      }
    }
    status = RequireMember(item, "available", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadBool(*field, item_context + ".available", &entry.available);
    if (!status.ok()) {
      return status;
    }
    field = item.Find("reason");
    if (field != nullptr && !field->IsNull()) {
      status = ReadString(*field, item_context + ".reason", &entry.reason);
      if (!status.ok()) {
        return status;
      }
    }
    if (entry.available) {
      if (!entry.has_offset) {
        return ManifestError(item_context +
                             ".offset is required for available weight");
      }
      if (entry.byte_size == 0) {
        return ManifestError(item_context +
                             ".byte_size must be positive");
      }
      if (entry.offset % out->alignment != 0) {
        return ManifestError(item_context +
                             ".offset is not aligned to weights.alignment");
      }
      if (entry.offset > out->total_bytes ||
          entry.byte_size > out->total_bytes - entry.offset) {
        return ManifestError(item_context + " exceeds weights.total_bytes");
      }
    }
    out->entries.push_back(std::move(entry));
  }
  return Status::Ok();
}

// 解析 statistics 段。
//
// 参数：
//   value：statistics 对象；
//   context：错误位置；
//   out：输出统计。
// 返回：
//   成功返回 kOk；字段缺失/非法返回 kInvalidModelFile。
Status ParseStatistics(const JsonValue& value, const std::string& context,
                       ManifestStatistics* out) {
  Status status = RequireObject(value, context);
  if (!status.ok()) {
    return status;
  }
  const JsonValue* field = nullptr;
  status = RequireMember(value, "tensors", context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, context + ".tensors", &out->tensors);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(value, "ops", context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, context + ".ops", &out->ops);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(value, "constants", context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, context + ".constants", &out->constants);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(value, "outputs", context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, context + ".outputs", &out->outputs);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(value, "activation_tensors", context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, context + ".activation_tensors",
                    &out->activation_tensors);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(value, "constant_bytes", context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadUint(*field, context + ".constant_bytes",
                    &out->constant_bytes);
  if (!status.ok()) {
    return status;
  }
  return Status::Ok();
}

// 校验 graph 段内部的引用完整性。
//
// 参数：
//   graph：待校验图视图。
//   context：错误位置。
// 返回：
//   成功返回 kOk；引用缺失/重复返回 kInvalidModelFile。
Status ValidateGraphReferences(const GraphManifest& graph,
                               const std::string& context) {
  std::set<std::string> names;
  for (const std::string& name : graph.inputs) {
    const TensorMeta* meta = nullptr;
    Status status = FindTensor(graph.tensors, name,
                               context + ".inputs", &meta);
    if (!status.ok()) {
      return status;
    }
    if (meta->kind != TensorKind::kInput) {
      return ManifestError(context + ".inputs tensor must be input kind: " +
                           name);
    }
    if (!names.insert(name).second) {
      return ManifestError(context + " has duplicate input name: " + name);
    }
  }
  names.clear();
  for (const std::string& name : graph.constants) {
    const TensorMeta* meta = nullptr;
    Status status = FindTensor(graph.tensors, name,
                               context + ".constants", &meta);
    if (!status.ok()) {
      return status;
    }
    if (meta->kind != TensorKind::kConstant) {
      return ManifestError(
          context + ".constants tensor must be constant kind: " + name);
    }
    if (!names.insert(name).second) {
      return ManifestError(context + " has duplicate constant name: " +
                           name);
    }
  }
  // 每个 input/constant 名称只能出现一次，且不能同时属于两类。
  std::set<std::string> all_io(graph.inputs.begin(), graph.inputs.end());
  all_io.insert(graph.constants.begin(), graph.constants.end());
  if (all_io.size() != graph.inputs.size() + graph.constants.size()) {
    return ManifestError(context +
                         " has tensor name shared by inputs/constants");
  }

  std::map<std::string, std::string> producers;
  for (const OpMeta& op : graph.ops) {
    for (const std::string& output : op.outputs) {
      producers.emplace(output, op.name);
    }
  }
  for (const auto& entry : graph.tensors) {
    const TensorMeta& meta = entry.second;
    const Bool produced = producers.find(meta.name) != producers.end();
    if (meta.kind == TensorKind::kActivation && !produced) {
      return ManifestError(context + " activation tensor has no producer: " +
                           meta.name);
    }
    if ((meta.kind == TensorKind::kInput ||
         meta.kind == TensorKind::kConstant) &&
        produced) {
      return ManifestError(context + " tensor must not be produced by op: " +
                           meta.name);
    }
  }
  std::set<std::string> output_names;
  for (const IOBinding& binding : graph.outputs) {
    if (!output_names.insert(binding.name).second) {
      return ManifestError(context +
                           " has duplicate output name: " + binding.name);
    }
    const TensorMeta* meta = nullptr;
    Status status = FindTensor(graph.tensors, binding.tensor,
                               context + ".outputs", &meta);
    if (!status.ok()) {
      return status;
    }
    if (meta->kind != TensorKind::kActivation) {
      return ManifestError(
          context + ".outputs tensor must be activation kind: " +
          binding.tensor);
    }
    if (producers.find(binding.tensor) == producers.end()) {
      return ManifestError(context +
                           ".outputs tensor has no producer: " +
                           binding.tensor);
    }
  }
  return Status::Ok();
}

// 校验权重表与常量张量、内存计划与 activation 张量的一致性。
//
// 参数：
//   graph：图视图；
//   weights：权重表；
//   plan：显存计划；
//   context：错误位置前缀。
// 返回：
//   成功返回 kOk；不一致返回 kInvalidModelFile。
Status ValidateWeightAndPlan(const GraphManifest& graph,
                             const WeightTable& weights,
                             const MemoryPlan& plan,
                             const std::string& context) {
  // 每个常量必须有且仅有一个权重条目，且 available 与张量声明一致。
  for (const std::string& name : graph.constants) {
    const WeightEntry* entry = weights.Find(name);
    if (entry == nullptr) {
      return ManifestError(context + " weights.entries is missing constant: " +
                           name);
    }
    const auto tensor_it = graph.tensors.find(name);
    if (tensor_it == graph.tensors.end()) {
      return ManifestError(context + " constant tensor is missing: " + name);
    }
    if (tensor_it->second.weight_available != entry->available) {
      return ManifestError(context + " weight availability mismatch: " +
                           name);
    }
  }
  for (const WeightEntry& entry : weights.entries) {
    const auto it = graph.tensors.find(entry.name);
    if (it == graph.tensors.end() ||
        it->second.kind != TensorKind::kConstant) {
      return ManifestError(context +
                           " weight entry does not match a constant: " +
                           entry.name);
    }
  }

  // 每个 activation 必须有且仅有一个显存计划条目。
  std::set<std::string> activation_names;
  for (const auto& item : graph.tensors) {
    if (item.second.kind == TensorKind::kActivation) {
      activation_names.insert(item.first);
    }
  }
  for (const std::string& name : activation_names) {
    if (plan.Find(name) == nullptr) {
      return ManifestError(context +
                           " memory_plan is missing activation: " + name);
    }
  }
  for (const MemoryPlanEntry& entry : plan.entries) {
    if (activation_names.find(entry.name) == activation_names.end()) {
      return ManifestError(context +
                           " memory entry does not match activation: " +
                           entry.name);
    }
  }
  return Status::Ok();
}

}  // namespace

std::string Sha256BytesToHex(const std::array<U8, 32>& digest) {
  static constexpr C8 kDigits[] = "0123456789abcdef";
  std::string hex;
  hex.resize(64);
  for (Usize index = 0; index < 32; ++index) {
    hex[index * 2] = kDigits[digest[index] >> 4];
    hex[index * 2 + 1] = kDigits[digest[index] & 0x0fU];
  }
  return hex;
}

Status ParseModelPackageHeader(const U8* data, Usize size, U64 file_size,
                               ModelPackageHeader* header) {
  if (data == nullptr || header == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ParseModelPackageHeader: data/header must not be null");
  }
  if (size < kModelPackageHeaderSize || file_size < kModelPackageHeaderSize) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "model file is too short for header");
  }
  if (file_size < static_cast<U64>(size)) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "header size exceeds file size");
  }
  if (std::memcmp(data, kModelPackageMagic, 8) != 0) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "bad model file magic");
  }
  const U32 version = ReadU32Le(data + 8);
  if (version != kModelPackageFormatVersion) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "unsupported model file version: " +
                          std::to_string(version));
  }
  const U32 flags = ReadU32Le(data + 12);
  if ((flags & ~kModelPackageFlagWeights) != 0) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "unsupported model file flags: " +
                          std::to_string(flags));
  }
  const U64 manifest_offset = ReadU64Le(data + 16);
  const U64 manifest_length = ReadU64Le(data + 24);
  if (manifest_offset < kModelPackageHeaderSize || manifest_length == 0) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "invalid manifest offset/length");
  }
  if (manifest_offset > file_size ||
      manifest_length > file_size - manifest_offset) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "manifest extends beyond model file");
  }
  const U64 weights_offset = ReadU64Le(data + 64);
  const U64 weights_length = ReadU64Le(data + 72);
  const Bool has_weights = (flags & kModelPackageFlagWeights) != 0;
  if (has_weights != (weights_length > 0)) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "weights flag and length are inconsistent");
  }
  if (has_weights && weights_offset < kModelPackageHeaderSize) {
    return MakeStatus(ErrorCode::kInvalidModelFile,
                      "invalid weights offset");
  }

  header->version = version;
  header->flags = flags;
  header->manifest_offset = manifest_offset;
  header->manifest_length = manifest_length;
  for (Usize index = 0; index < 32; ++index) {
    header->manifest_sha256[index] = data[32 + index];
    header->weights_sha256[index] = data[80 + index];
  }
  header->weights_offset = weights_offset;
  header->weights_length = weights_length;
  header->has_weights = has_weights;
  return Status::Ok();
}

Status ReadModelPackageHeader(const std::string& model_file,
                              ModelPackageHeader* header) {
  if (header == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ReadModelPackageHeader: header must not be null");
  }
  U64 file_size = 0;
  Status status = GetFileSize(model_file, &file_size);
  if (!status.ok()) {
    return status;
  }
  std::vector<U8> bytes;
  status = ReadFileRange(model_file, 0, kModelPackageHeaderSize, &bytes);
  if (!status.ok()) {
    return status;
  }
  return ParseModelPackageHeader(bytes.data(), bytes.size(), file_size,
                                 header);
}

Status ParseManifest(const std::string& text, ModelManifest* manifest) {
  if (manifest == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ParseManifest: manifest must not be null");
  }
  JsonValue root;
  Status status = ParseJson(text, &root);
  if (!status.ok()) {
    return ManifestError(std::string("JSON parse failed: ") +
                         status.message());
  }
  status = RequireObject(root, "manifest");
  if (!status.ok()) {
    return status;
  }

  const std::string format_context = "manifest.format";
  const JsonValue* field = nullptr;
  status = RequireMember(root, "format", "manifest", &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadString(*field, format_context, &manifest->format);
  if (!status.ok()) {
    return status;
  }
  if (manifest->format != "MobileNet.ModelPackage") {
    return ManifestError("unsupported format: " + manifest->format);
  }
  status = RequireMember(root, "format_version", "manifest", &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadInt(*field, "manifest.format_version",
                   &manifest->format_version);
  if (!status.ok()) {
    return status;
  }
  if (manifest->format_version !=
      static_cast<I64>(kModelPackageFormatVersion)) {
    return ManifestError("unsupported format_version: " +
                         std::to_string(manifest->format_version));
  }

  // model 段。
  const JsonValue* model = nullptr;
  status = RequireMember(root, "model", "manifest", &model);
  if (!status.ok()) {
    return status;
  }
  status = RequireObject(*model, "manifest.model");
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(*model, "name", "manifest.model", &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadNonEmptyString(*field, "manifest.model.name",
                              &manifest->model_name);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(*model, "adapter", "manifest.model", &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadNonEmptyString(*field, "manifest.model.adapter",
                              &manifest->model_adapter);
  if (!status.ok()) {
    return status;
  }
  field = model->Find("config");
  manifest->model_config =
      field != nullptr ? *field : JsonValue::MakeObject({});
  if (!manifest->model_config.IsObject()) {
    return ManifestError("manifest.model.config must be an object");
  }

  // graph 段。
  const JsonValue* graph = nullptr;
  status = RequireMember(root, "graph", "manifest", &graph);
  if (!status.ok()) {
    return status;
  }
  status = RequireObject(*graph, "manifest.graph");
  if (!status.ok()) {
    return status;
  }
  const std::string graph_context = "manifest.graph";
  status = RequireMember(*graph, "schema_version", graph_context, &field);
  if (!status.ok()) {
    return status;
  }
  I64 schema_version = 0;
  status = ReadInt(*field, graph_context + ".schema_version",
                   &schema_version);
  if (!status.ok()) {
    return status;
  }
  if (schema_version != 1) {
    return ManifestError("unsupported graph.schema_version: " +
                         std::to_string(schema_version));
  }
  status = RequireMember(*graph, "name", graph_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadNonEmptyString(*field, graph_context + ".name",
                              &manifest->graph.name);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(*graph, "adapter", graph_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadNonEmptyString(*field, graph_context + ".adapter",
                              &manifest->graph.adapter);
  if (!status.ok()) {
    return status;
  }
  field = graph->Find("metadata");
  manifest->graph_metadata =
      field != nullptr ? *field : JsonValue::MakeObject({});
  if (!manifest->graph_metadata.IsObject()) {
    return ManifestError(graph_context + ".metadata must be an object");
  }

  status = RequireMember(*graph, "inputs", graph_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadStringArray(*field, graph_context + ".inputs", true,
                           &manifest->graph.inputs);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(*graph, "constants", graph_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadStringArray(*field, graph_context + ".constants", true,
                           &manifest->graph.constants);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(*graph, "outputs", graph_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = RequireArray(*field, graph_context + ".outputs");
  if (!status.ok()) {
    return status;
  }
  manifest->graph.outputs.clear();
  const std::vector<JsonValue>& output_items = field->AsArray();
  for (Usize index = 0; index < output_items.size(); ++index) {
    const JsonValue& item = output_items[index];
    const std::string item_context =
        graph_context + ".outputs[" + std::to_string(index) + "]";
    status = RequireObject(item, item_context);
    if (!status.ok()) {
      return status;
    }
    IOBinding binding;
    status = RequireMember(item, "name", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadNonEmptyString(*field, item_context + ".name",
                                &binding.name);
    if (!status.ok()) {
      return status;
    }
    status = RequireMember(item, "tensor", item_context, &field);
    if (!status.ok()) {
      return status;
    }
    status = ReadNonEmptyString(*field, item_context + ".tensor",
                                &binding.tensor);
    if (!status.ok()) {
      return status;
    }
    manifest->graph.outputs.push_back(std::move(binding));
  }
  status = RequireMember(*graph, "tensors", graph_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ParseTensors(*field, graph_context + ".tensors",
                        &manifest->graph.tensors);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(*graph, "ops", graph_context, &field);
  if (!status.ok()) {
    return status;
  }
  std::map<std::string, std::string> producers;
  status = ParseOps(*field, manifest->graph.tensors,
                    graph_context + ".ops", &manifest->graph.ops,
                    &producers);
  if (!status.ok()) {
    return status;
  }
  status = RequireMember(*graph, "ir_sha256", graph_context, &field);
  if (!status.ok()) {
    return status;
  }
  status = ReadString(*field, graph_context + ".ir_sha256",
                      &manifest->graph.ir_sha256);
  if (!status.ok()) {
    return status;
  }
  if (!IsSha256Hex(manifest->graph.ir_sha256)) {
    return ManifestError(graph_context +
                         ".ir_sha256 must be 64 hex characters");
  }
  field = graph->Find("ir_line_count");
  if (field != nullptr) {
    status = ReadInt(*field, graph_context + ".ir_line_count",
                     &manifest->graph.ir_line_count);
    if (!status.ok()) {
      return status;
    }
    if (manifest->graph.ir_line_count < 0) {
      return ManifestError(graph_context +
                           ".ir_line_count must not be negative");
    }
  }
  status = ValidateGraphReferences(manifest->graph, graph_context);
  if (!status.ok()) {
    return status;
  }

  // optimization 段。
  const JsonValue* optimization = nullptr;
  status = RequireMember(root, "optimization", "manifest", &optimization);
  if (!status.ok()) {
    return status;
  }
  status = ParseOptimization(*optimization, "manifest.optimization",
                             &manifest->optimization_config,
                             &manifest->memory_plan);
  if (!status.ok()) {
    return status;
  }
  // optimization.config.alignment 若声明必须与 memory_plan.alignment 一致。
  const JsonValue* alignment_field =
      manifest->optimization_config.Find("alignment");
  if (alignment_field != nullptr) {
    U64 config_alignment = 0;
    status = ReadUint(*alignment_field,
                      "manifest.optimization.config.alignment",
                      &config_alignment);
    if (!status.ok()) {
      return status;
    }
    if (config_alignment != manifest->memory_plan.alignment) {
      return ManifestError(
          "manifest.optimization.config.alignment does not match "
          "memory_plan.alignment");
    }
  }

  // weights 段。
  const JsonValue* weights = nullptr;
  status = RequireMember(root, "weights", "manifest", &weights);
  if (!status.ok()) {
    return status;
  }
  status = ParseWeights(*weights, "manifest.weights", &manifest->weights);
  if (!status.ok()) {
    return status;
  }

  // statistics 段与交叉校验。
  const JsonValue* statistics = nullptr;
  status = RequireMember(root, "statistics", "manifest", &statistics);
  if (!status.ok()) {
    return status;
  }
  status = ParseStatistics(*statistics, "manifest.statistics",
                           &manifest->statistics);
  if (!status.ok()) {
    return status;
  }

  U64 activation_count = 0;
  U64 constant_bytes = 0;
  for (const auto& entry : manifest->graph.tensors) {
    const TensorMeta& meta = entry.second;
    if (meta.kind == TensorKind::kActivation) {
      ++activation_count;
    } else if (meta.kind == TensorKind::kConstant) {
      if (constant_bytes > std::numeric_limits<U64>::max() - meta.byte_size) {
        return ManifestError("constant bytes overflow");
      }
      constant_bytes += meta.byte_size;
    }
  }
  if (manifest->statistics.tensors != manifest->graph.tensors.size()) {
    return ManifestError("statistics.tensors mismatch: declared " +
                         std::to_string(manifest->statistics.tensors) +
                         ", actual " +
                         std::to_string(manifest->graph.tensors.size()));
  }
  if (manifest->statistics.ops != manifest->graph.ops.size()) {
    return ManifestError("statistics.ops mismatch: declared " +
                         std::to_string(manifest->statistics.ops) +
                         ", actual " +
                         std::to_string(manifest->graph.ops.size()));
  }
  if (manifest->statistics.constants !=
      manifest->graph.constants.size()) {
    return ManifestError("statistics.constants mismatch: declared " +
                         std::to_string(manifest->statistics.constants) +
                         ", actual " +
                         std::to_string(manifest->graph.constants.size()));
  }
  if (manifest->statistics.outputs != manifest->graph.outputs.size()) {
    return ManifestError("statistics.outputs mismatch: declared " +
                         std::to_string(manifest->statistics.outputs) +
                         ", actual " +
                         std::to_string(manifest->graph.outputs.size()));
  }
  if (manifest->statistics.activation_tensors != activation_count) {
    return ManifestError(
        "statistics.activation_tensors mismatch: declared " +
        std::to_string(manifest->statistics.activation_tensors) +
        ", actual " + std::to_string(activation_count));
  }
  if (manifest->statistics.constant_bytes != constant_bytes) {
    return ManifestError(
        "statistics.constant_bytes mismatch: declared " +
        std::to_string(manifest->statistics.constant_bytes) + ", actual " +
        std::to_string(constant_bytes));
  }
  if (manifest->memory_plan.constants_bytes != 0 &&
      manifest->memory_plan.constants_bytes != constant_bytes) {
    return ManifestError(
        "memory_plan.constants_bytes does not match constant tensors");
  }
  if (manifest->memory_plan.entries.size() != activation_count) {
    return ManifestError(
        "memory_plan entries do not cover all activation tensors");
  }

  status = ValidateWeightAndPlan(manifest->graph, manifest->weights,
                                 manifest->memory_plan, "manifest");
  if (!status.ok()) {
    return status;
  }

  // 权重条目形状与张量形状、字节数必须一致。
  for (const WeightEntry& entry : manifest->weights.entries) {
    const auto tensor_it = manifest->graph.tensors.find(entry.name);
    if (tensor_it == manifest->graph.tensors.end()) {
      return ManifestError("weight entry has no tensor: " + entry.name);
    }
    const TensorMeta& meta = tensor_it->second;
    if (entry.shape != meta.dims) {
      return ManifestError("weight shape mismatch for: " + entry.name);
    }
    DataType source_dtype = DataType::kUnknown;
    status = DataTypeFromName(entry.dtype, &source_dtype);
    if (!status.ok()) {
      return status;
    }
    const U32 element_size = DataTypeSize(source_dtype);
    if (element_size == 0) {
      return ManifestError("weight dtype has no element size: " +
                           entry.name);
    }
    U64 element_count = 0;
    status = ElementCount(entry.shape,
                          "manifest.weights entry " + entry.name,
                          &element_count);
    if (!status.ok()) {
      return status;
    }
    const U64 expected =
        element_count * static_cast<U64>(element_size);
    if (entry.byte_size != expected) {
      return ManifestError("weight byte_size does not match shape/dtype: " +
                           entry.name);
    }
  }

  // 内存计划条目大小必须与 activation 张量字节数一致。
  for (const MemoryPlanEntry& entry : manifest->memory_plan.entries) {
    const auto tensor_it = manifest->graph.tensors.find(entry.name);
    if (tensor_it == manifest->graph.tensors.end()) {
      return ManifestError("memory entry has no tensor: " + entry.name);
    }
    const TensorMeta& meta = tensor_it->second;
    if (entry.size != meta.byte_size) {
      return ManifestError("memory entry size mismatch for: " + entry.name);
    }
    if (entry.offset % manifest->memory_plan.alignment != 0) {
      return ManifestError("memory entry offset is not aligned: " +
                           entry.name);
    }
  }
  return Status::Ok();
}

Status ParseModelConfig(const ModelManifest& manifest,
                        ModelConfig* config) {
  if (config == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ParseModelConfig: config must not be null");
  }
  const JsonValue* model_config =
      manifest.graph_metadata.Find("model_config");
  if (model_config == nullptr || !model_config->IsObject()) {
    return ManifestError(
        "graph.metadata.model_config is required and must be an object");
  }
  const JsonValue* inference =
      manifest.graph_metadata.Find("inference_config");
  if (inference != nullptr && !inference->IsObject()) {
    return ManifestError(
        "graph.metadata.inference_config must be an object");
  }

  const auto read_required_int = [model_config](
                                     const C8* key, I64* out) -> Status {
    const JsonValue* field = model_config->Find(key);
    if (field == nullptr) {
      return ManifestError(std::string("model_config.") + key +
                           " is required");
    }
    return ReadInt(*field, std::string("model_config.") + key, out);
  };

  ModelConfig result;
  I64 value = 0;
  Status status = read_required_int("num_hidden_layers", &value);
  if (!status.ok()) {
    return status;
  }
  if (value <= 0 ||
      value > static_cast<I64>(std::numeric_limits<I32>::max())) {
    return ManifestError("model_config.num_hidden_layers is out of range");
  }
  result.num_layers = static_cast<I32>(value);

  status = read_required_int("num_attention_heads", &value);
  if (!status.ok()) {
    return status;
  }
  if (value <= 0 ||
      value > static_cast<I64>(std::numeric_limits<I32>::max())) {
    return ManifestError("model_config.num_attention_heads is out of range");
  }
  result.num_heads = static_cast<I32>(value);

  status = read_required_int("num_key_value_heads", &value);
  if (!status.ok()) {
    return status;
  }
  if (value <= 0 ||
      value > static_cast<I64>(std::numeric_limits<I32>::max())) {
    return ManifestError("model_config.num_key_value_heads is out of range");
  }
  result.num_kv_heads = static_cast<I32>(value);

  status = read_required_int("head_dim", &value);
  if (!status.ok()) {
    return status;
  }
  if (value <= 0 ||
      value > static_cast<I64>(std::numeric_limits<I32>::max())) {
    return ManifestError("model_config.head_dim is out of range");
  }
  result.head_dim = static_cast<I32>(value);

  status = read_required_int("hidden_size", &result.hidden_size);
  if (!status.ok()) {
    return status;
  }
  if (result.hidden_size <= 0) {
    return ManifestError("model_config.hidden_size must be positive");
  }
  status = read_required_int("intermediate_size", &result.intermediate_size);
  if (!status.ok()) {
    return status;
  }
  if (result.intermediate_size <= 0) {
    return ManifestError("model_config.intermediate_size must be positive");
  }
  status = read_required_int("vocab_size", &result.vocab_size);
  if (!status.ok()) {
    return status;
  }
  if (result.vocab_size <= 0) {
    return ManifestError("model_config.vocab_size must be positive");
  }

  const JsonValue* field = model_config->Find("rms_norm_eps");
  if (field != nullptr) {
    FP64 eps = 0.0;
    status = ReadNumber(*field, "model_config.rms_norm_eps", &eps);
    if (!status.ok()) {
      return status;
    }
    if (eps < 0.0 || eps > std::numeric_limits<FP32>::max()) {
      return ManifestError("model_config.rms_norm_eps is out of range");
    }
    result.rms_norm_eps = static_cast<FP32>(eps);
  }
  field = model_config->Find("rope_theta");
  if (field != nullptr) {
    status = ReadNumber(*field, "model_config.rope_theta",
                        &result.rope_theta);
    if (!status.ok()) {
      return status;
    }
    if (result.rope_theta <= 0.0) {
      return ManifestError("model_config.rope_theta must be positive");
    }
  }

  // context_length：inference_config 优先，其次 optimization.config，
  // 最后 model_config.max_position_embeddings。
  result.context_length = 0;
  if (inference != nullptr) {
    field = inference->Find("context_length");
    if (field != nullptr) {
      status = ReadInt(*field, "inference_config.context_length",
                       &result.context_length);
      if (!status.ok()) {
        return status;
      }
    }
  }
  if (result.context_length == 0) {
    field = manifest.optimization_config.Find("context_length");
    if (field != nullptr) {
      status = ReadInt(*field, "optimization.config.context_length",
                       &result.context_length);
      if (!status.ok()) {
        return status;
      }
    }
  }
  if (result.context_length == 0) {
    field = model_config->Find("max_position_embeddings");
    if (field != nullptr) {
      status = ReadInt(*field, "model_config.max_position_embeddings",
                       &result.context_length);
      if (!status.ok()) {
        return status;
      }
    }
  }
  if (result.context_length <= 0) {
    return ManifestError("model context_length must be positive");
  }

  // max_num_tokens：inference_config 优先，其次 optimization.config。
  result.max_num_tokens = 0;
  if (inference != nullptr) {
    field = inference->Find("max_num_tokens");
    if (field != nullptr) {
      I64 tokens = 0;
      status = ReadInt(*field, "inference_config.max_num_tokens", &tokens);
      if (!status.ok()) {
        return status;
      }
      if (tokens < 0 ||
          tokens > static_cast<I64>(std::numeric_limits<I32>::max())) {
        return ManifestError("inference_config.max_num_tokens out of range");
      }
      result.max_num_tokens = static_cast<I32>(tokens);
    }
  }
  if (result.max_num_tokens == 0) {
    field = manifest.optimization_config.Find("max_num_tokens");
    if (field != nullptr) {
      I64 tokens = 0;
      status = ReadInt(*field, "optimization.config.max_num_tokens",
                       &tokens);
      if (!status.ok()) {
        return status;
      }
      if (tokens < 0 ||
          tokens > static_cast<I64>(std::numeric_limits<I32>::max())) {
        return ManifestError("optimization.max_num_tokens out of range");
      }
      result.max_num_tokens = static_cast<I32>(tokens);
    }
  }
  if (result.max_num_tokens < 0) {
    return ManifestError("model max_num_tokens must not be negative");
  }

  // max_batch_size：inference_config 优先，其次 optimization.config，
  // 最后默认 1。
  result.max_batch_size = 1;
  Bool has_batch = false;
  if (inference != nullptr) {
    field = inference->Find("max_batch_size");
    if (field != nullptr) {
      I64 batch = 0;
      status = ReadInt(*field, "inference_config.max_batch_size", &batch);
      if (!status.ok()) {
        return status;
      }
      if (batch <= 0 ||
          batch > static_cast<I64>(std::numeric_limits<I32>::max())) {
        return ManifestError("inference_config.max_batch_size out of range");
      }
      result.max_batch_size = static_cast<I32>(batch);
      has_batch = true;
    }
  }
  if (!has_batch) {
    field = manifest.optimization_config.Find("batch_size");
    if (field != nullptr) {
      I64 batch = 0;
      status = ReadInt(*field, "optimization.config.batch_size", &batch);
      if (!status.ok()) {
        return status;
      }
      if (batch <= 0 ||
          batch > static_cast<I64>(std::numeric_limits<I32>::max())) {
        return ManifestError("optimization.config.batch_size out of range");
      }
      result.max_batch_size = static_cast<I32>(batch);
    }
  }

  result.pad_token_id = 0;
  result.bos_token_id = 0;
  result.stop_token_ids.clear();
  if (inference != nullptr) {
    field = inference->Find("pad_token_id");
    if (field != nullptr) {
      status = ReadInt(*field, "inference_config.pad_token_id",
                       &result.pad_token_id);
      if (!status.ok()) {
        return status;
      }
    }
    field = inference->Find("bos_token_id");
    if (field != nullptr) {
      status = ReadInt(*field, "inference_config.bos_token_id",
                       &result.bos_token_id);
      if (!status.ok()) {
        return status;
      }
    }
    field = inference->Find("stop_token_ids");
    if (field != nullptr) {
      status = ReadIntArray(*field, "inference_config.stop_token_ids",
                            &result.stop_token_ids);
      if (!status.ok()) {
        return status;
      }
    }
  }

  // 目标精度来自 optimization.config.target_precision，不使用官方 bf16。
  field = manifest.optimization_config.Find("target_precision");
  if (field == nullptr) {
    return ManifestError("optimization.config.target_precision is required");
  }
  std::string target_precision;
  status = ReadNonEmptyString(*field,
                              "optimization.config.target_precision",
                              &target_precision);
  if (!status.ok()) {
    return status;
  }
  status = DataTypeFromName(target_precision, &result.dtype);
  if (!status.ok()) {
    return status;
  }

  *config = result;
  return Status::Ok();
}

Status PrecisionFromName(const std::string& name,
                         runtime_graph::Precision* precision) {
  if (precision == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "PrecisionFromName: precision must not be null");
  }
  if (name == "unknown") {
    *precision = runtime_graph::Precision::kUnknown;
  } else if (name == "float64") {
    *precision = runtime_graph::Precision::kFloat64;
  } else if (name == "float32") {
    *precision = runtime_graph::Precision::kFloat32;
  } else if (name == "float16") {
    *precision = runtime_graph::Precision::kFloat16;
  } else if (name == "bfloat16") {
    *precision = runtime_graph::Precision::kBFloat16;
  } else if (name == "float8_e4m3") {
    *precision = runtime_graph::Precision::kFloat8E4M3;
  } else if (name == "float8_e5m2") {
    *precision = runtime_graph::Precision::kFloat8E5M2;
  } else if (name == "int64") {
    *precision = runtime_graph::Precision::kInt64;
  } else if (name == "int32") {
    *precision = runtime_graph::Precision::kInt32;
  } else if (name == "int16") {
    *precision = runtime_graph::Precision::kInt16;
  } else if (name == "int8") {
    *precision = runtime_graph::Precision::kInt8;
  } else if (name == "uint8") {
    *precision = runtime_graph::Precision::kUInt8;
  } else if (name == "bool") {
    *precision = runtime_graph::Precision::kBool;
  } else {
    return ManifestError("unsupported precision: " + name);
  }
  return Status::Ok();
}

Status DataTypeFromName(const std::string& name, DataType* dtype) {
  if (dtype == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "DataTypeFromName: dtype must not be null");
  }
  if (name == "float64") {
    *dtype = DataType::kFloat64;
  } else if (name == "float32") {
    *dtype = DataType::kFloat32;
  } else if (name == "float16") {
    *dtype = DataType::kFloat16;
  } else if (name == "bfloat16") {
    *dtype = DataType::kBFloat16;
  } else if (name == "int8") {
    *dtype = DataType::kInt8;
  } else if (name == "uint8") {
    *dtype = DataType::kUInt8;
  } else if (name == "int32") {
    *dtype = DataType::kInt32;
  } else if (name == "int64") {
    *dtype = DataType::kInt64;
  } else if (name == "bool") {
    *dtype = DataType::kBool;
  } else {
    return ManifestError("unsupported data type: " + name);
  }
  return Status::Ok();
}

Status LayoutFromName(const std::string& name,
                      runtime_graph::Layout* layout) {
  if (layout == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LayoutFromName: layout must not be null");
  }
  if (name == "row_major") {
    *layout = runtime_graph::Layout::kRowMajor;
  } else if (name == "column_major") {
    *layout = runtime_graph::Layout::kColumnMajor;
  } else {
    return ManifestError("unsupported layout: " + name);
  }
  return Status::Ok();
}

Status OpTypeFromName(const std::string& name,
                      runtime_graph::OpType* op_type) {
  if (op_type == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OpTypeFromName: op_type must not be null");
  }
  const I32 last = static_cast<I32>(runtime_graph::OpType::kSample);
  for (I32 value = 0; value <= last; ++value) {
    const auto candidate = static_cast<runtime_graph::OpType>(value);
    if (name == runtime_graph::OpTypeToStr(candidate)) {
      *op_type = candidate;
      return Status::Ok();
    }
  }
  return ManifestError("unsupported op type: " + name);
}

const WeightEntry* WeightTable::Find(const std::string& name) const {
  for (const WeightEntry& entry : entries) {
    if (entry.name == name) {
      return &entry;
    }
  }
  return nullptr;
}

const MemoryPlanEntry* MemoryPlan::Find(const std::string& name) const {
  for (const MemoryPlanEntry& entry : entries) {
    if (entry.name == name) {
      return &entry;
    }
  }
  return nullptr;
}

}  // namespace runtime
}  // namespace mobilenet
