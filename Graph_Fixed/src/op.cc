// 模块：Runtime Graph Op 实现
//
// 职责：
//   实现 Op 的构造、属性访问、tensor 连接维护和 OpEnv 生命周期管理。
//
// 边界：
//   不校验 tensor 归属，不执行 op 计算；连接校验和计算分别由 Graph 与
//   OpEnv 负责。
//
// 依赖：
//   runtime_graph/op.h、Runtime/common/status.h。

#include "runtime_graph/op.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Runtime/common/status.h"
#include "runtime_graph/tensor.h"

namespace mobilenet {
namespace runtime_graph {

Op::Op(const std::string& name, OpType op_type, const AttrMap& attr)
    : name_(name), op_type_(op_type), attr_(attr) {}

const std::string& Op::Name() const { return name_; }

OpType Op::Type() const { return op_type_; }

const AttrMap& Op::Attrs() const { return attr_; }

Status Op::SetAttr(const std::string& attr_name, const AttrValue& value) {
  if (attr_name.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Op::SetAttr: attribute name must not be empty");
  }
  attr_[attr_name] = value;
  return Status::Ok();
}

const std::vector<Tensor*>& Op::SrcTensors() const { return src_tensors_; }

const std::vector<Tensor*>& Op::DstTensors() const { return dst_tensors_; }

Status Op::SetOpEnv(std::unique_ptr<OpEnv> op_env) {
  if (op_env == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Op::SetOpEnv: op_env must not be null");
  }
  op_env_ = std::move(op_env);
  return Status::Ok();
}

void Op::ClearOpEnv() { op_env_.reset(); }

OpEnv* Op::GetOpEnv() const { return op_env_.get(); }

void Op::AddSrcTensor(Tensor* tensor) {
  if (tensor != nullptr) {
    src_tensors_.push_back(tensor);
  }
}

void Op::AddDstTensor(Tensor* tensor) {
  if (tensor != nullptr) {
    dst_tensors_.push_back(tensor);
  }
}

}  // namespace runtime_graph
}  // namespace mobilenet
