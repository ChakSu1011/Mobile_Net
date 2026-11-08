// 模块：Graph 公共类型实现
//
// 职责：
//   实现量化粒度名称、校验报告收集、名称合法性检查与属性类型名。
//
// 边界：
//   不依赖 LibTorch，也不包含图构建/执行逻辑。

#include "graph/graph_common.h"

#include <sstream>
#include <utility>

namespace mobilenet {
namespace graph {

const char* QuantGranularityName(QuantGranularity granularity) {
  switch (granularity) {
    case QuantGranularity::kNone:
      return "none";
    case QuantGranularity::kPerTensor:
      return "per_tensor";
    case QuantGranularity::kPerChannel:
      return "per_channel";
    case QuantGranularity::kPerGroup:
      return "per_group";
  }
  return "none";
}

void ValidationReport::AddIssue(ValidationIssue issue) {
  if (issues_.size() >= kMaxIssues) {
    if (!issues_.empty() &&
        issues_.back().message.find("truncated") == std::string::npos) {
      issues_.push_back(ValidationIssue{
          ErrorCode::kInternal, "validation issues truncated", "", "", "", ""});
    }
    return;
  }
  issues_.push_back(std::move(issue));
}

std::string ValidationReport::ToString() const {
  if (ok()) {
    return "OK";
  }
  std::ostringstream stream;
  stream << "validation failed with " << issues_.size() << " issue(s)";
  for (const ValidationIssue& issue : issues_) {
    stream << "\n[" << ErrorCodeName(issue.code) << "]";
    if (!issue.node_name.empty()) {
      stream << " node=" << issue.node_name;
    }
    if (!issue.edge_name.empty()) {
      stream << " edge=" << issue.edge_name;
    }
    if (!issue.binding_name.empty()) {
      stream << " binding=" << issue.binding_name;
    }
    if (!issue.rule_id.empty()) {
      stream << " rule=" << issue.rule_id;
    }
    stream << ": " << issue.message;
  }
  return stream.str();
}

bool IsValidGraphName(std::string_view name) {
  if (name.empty() || name.size() > kMaxGraphNameLength) {
    return false;
  }
  for (char character : name) {
    const bool is_lower = character >= 'a' && character <= 'z';
    const bool is_upper = character >= 'A' && character <= 'Z';
    const bool is_digit = character >= '0' && character <= '9';
    const bool is_symbol =
        character == '_' || character == '.' || character == '-';
    if (!is_lower && !is_upper && !is_digit && !is_symbol) {
      return false;
    }
  }
  return true;
}

const char* AttributeTypeName(const AttributeValue& value) {
  if (std::holds_alternative<bool>(value)) {
    return "bool";
  }
  if (std::holds_alternative<int64_t>(value)) {
    return "int64";
  }
  if (std::holds_alternative<double>(value)) {
    return "double";
  }
  if (std::holds_alternative<std::string>(value)) {
    return "string";
  }
  if (std::holds_alternative<std::vector<int64_t>>(value)) {
    return "int64_array";
  }
  if (std::holds_alternative<std::vector<double>>(value)) {
    return "double_array";
  }
  if (std::holds_alternative<std::vector<std::string>>(value)) {
    return "string_array";
  }
  return "unknown";
}

}  // namespace graph
}  // namespace mobilenet
