// 模块：Runtime Graph Tensor 实现
//
// 职责：
//   实现 Tensor 的构造、连接查询、Storage 访问和复用/constant 标记。
//
// 边界：
//   不校验 op 指针归属，不管理 op 生命周期；这些职责由 Graph 承担。
//
// 依赖：
//   tensor.h。

#include "tensor.h"

#include <string>
#include <utility>
#include <vector>

namespace mobilenet {
namespace runtime_graph {

Tensor::Tensor(const std::string& name) : name_(name) {}

const std::string& Tensor::Name() const { return name_; }

const std::vector<Op*>& Tensor::SrcOps() const { return src_ops_; }

const std::vector<Op*>& Tensor::DstOps() const { return dst_ops_; }

Storage& Tensor::GetStorage() { return storage_; }

const Storage& Tensor::GetStorage() const { return storage_; }

bool Tensor::Reusable() const { return storage_.reusable; }

void Tensor::SetReusable(bool reusable) { storage_.reusable = reusable; }

bool Tensor::IsConstant() const { return is_constant_; }

void Tensor::AddSrcOp(Op* op) {
  if (op != nullptr) {
    src_ops_.push_back(op);
  }
}

void Tensor::AddDstOp(Op* op) {
  if (op != nullptr) {
    dst_ops_.push_back(op);
  }
}

void Tensor::SetConstant(bool is_constant) { is_constant_ = is_constant; }

}  // namespace runtime_graph
}  // namespace mobilenet
