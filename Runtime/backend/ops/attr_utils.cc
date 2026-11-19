// 模块：算子属性严格读取工具实现
//
// 职责：
//   实现 AttrMap 的类型严格读取与 op 级错误前缀。
//
// 边界：
//   不做默认值推断或范围校验；不修改 AttrMap。
//
// 依赖：
//   Runtime/backend/ops/attr_utils.h、Runtime/common/status.h。

#include "Runtime/backend/ops/attr_utils.h"

#include <variant>
#include "Utils/cpp/types.h"

namespace mobilenet {

Bool HasAttr(const runtime_graph::AttrMap& attrs,
             const std::string& name) {
  return attrs.find(name) != attrs.end();
}

Status GetIntAttr(const runtime_graph::AttrMap& attrs, const C8* name,
                  I64* value) {
  if (name == nullptr || value == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GetIntAttr: null name or output");
  }
  const auto it = attrs.find(name);
  if (it == attrs.end()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      std::string("attr '") + name + "' is missing");
  }
  const auto* typed = std::get_if<I64>(&it->second);
  if (typed == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      std::string("attr '") + name +
                          "' must be int64");
  }
  *value = *typed;
  return Status::Ok();
}

Status GetFloatAttr(const runtime_graph::AttrMap& attrs, const C8* name,
                    F64* value) {
  if (name == nullptr || value == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GetFloatAttr: null name or output");
  }
  const auto it = attrs.find(name);
  if (it == attrs.end()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      std::string("attr '") + name + "' is missing");
  }
  if (const auto* typed = std::get_if<F64>(&it->second)) {
    *value = *typed;
    return Status::Ok();
  }
  if (const auto* integer = std::get_if<I64>(&it->second)) {
    *value = static_cast<F64>(*integer);
    return Status::Ok();
  }
  return MakeStatus(ErrorCode::kInvalidArgument,
                    std::string("attr '") + name +
                        "' must be a number");
}

Status GetBoolAttr(const runtime_graph::AttrMap& attrs, const C8* name,
                   Bool* value) {
  if (name == nullptr || value == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GetBoolAttr: null name or output");
  }
  const auto it = attrs.find(name);
  if (it == attrs.end()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      std::string("attr '") + name + "' is missing");
  }
  const auto* typed = std::get_if<Bool>(&it->second);
  if (typed == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      std::string("attr '") + name + "' must be Bool");
  }
  *value = *typed;
  return Status::Ok();
}

Status GetStringAttr(const runtime_graph::AttrMap& attrs, const C8* name,
                     std::string* value) {
  if (name == nullptr || value == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GetStringAttr: null name or output");
  }
  const auto it = attrs.find(name);
  if (it == attrs.end()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      std::string("attr '") + name + "' is missing");
  }
  const auto* typed = std::get_if<std::string>(&it->second);
  if (typed == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      std::string("attr '") + name +
                          "' must be string");
  }
  *value = *typed;
  return Status::Ok();
}

Status GetIntAttr(const runtime_graph::Op& op, const C8* name,
                  I64* value) {
  const Status status = GetIntAttr(op.Attrs(), name, value);
  if (!status.ok()) {
    return PrefixStatus(status, "op=" + op.Name());
  }
  return status;
}

Status GetFloatAttr(const runtime_graph::Op& op, const C8* name,
                    F64* value) {
  const Status status = GetFloatAttr(op.Attrs(), name, value);
  if (!status.ok()) {
    return PrefixStatus(status, "op=" + op.Name());
  }
  return status;
}

Status GetBoolAttr(const runtime_graph::Op& op, const C8* name,
                   Bool* value) {
  const Status status = GetBoolAttr(op.Attrs(), name, value);
  if (!status.ok()) {
    return PrefixStatus(status, "op=" + op.Name());
  }
  return status;
}

Status GetStringAttr(const runtime_graph::Op& op, const C8* name,
                     std::string* value) {
  const Status status = GetStringAttr(op.Attrs(), name, value);
  if (!status.ok()) {
    return PrefixStatus(status, "op=" + op.Name());
  }
  return status;
}

}  // namespace mobilenet
