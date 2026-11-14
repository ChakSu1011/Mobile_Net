// 模块：Runtime Graph 对象实现
//
// 职责：
//   实现 Graph 的构图、图边界注册、post DFS 拓扑排序、constant/输入/输出
//   查询和 GraphEnv 生命周期管理。
//
// 边界：
//   - 不执行 kernel，不推导 shape/precision/layout，不分配显存；
//   - 所有 Create* 接口先完成参数与名称校验，再提交图结构；
//   - SortOp 只维护 op_list_ 中的非拥有裸指针。
//
// 依赖：
//   graph.h、tensor.h、Runtime/common/status.h。

#include "graph.h"

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Runtime/common/status.h"
#include "tensor.h"

namespace mobilenet {
namespace runtime_graph {

Graph::Graph(const std::string& name) : name_(name) {}

Graph::~Graph() {
  // graph_env_ 是最后一个成员，默认析构时会在 op_map_ 之前释放；先清空
  // 所有 OpEnv，确保派生 OpEnv 的析构逻辑仍能安全访问 GraphEnv。
  for (auto& entry : op_map_) {
    entry.second->ClearOpEnv();
  }
}

const std::string& Graph::Name() const { return name_; }

Status Graph::CreateOp(const std::string& name,
                       const std::vector<Tensor*>& src_tensors, size_t out_cnt,
                       OpType op_type, const AttrMap& attr,
                       std::vector<Tensor*>* outputs) {
  Status status = ValidateNewOpName(name);
  if (!status.ok()) {
    return status;
  }
  if (outputs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Graph::CreateOp: outputs must not be null");
  }
  if (out_cnt == 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Graph::CreateOp: out_cnt must be greater than zero");
  }
  if (op_type == OpType::kUnknown || op_type == OpType::kInput ||
      op_type == OpType::kOutput) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Graph::CreateOp: op_type must be a compute op, got " +
            std::string(OpTypeToStr(op_type)));
  }
  for (Tensor* source : src_tensors) {
    if (!OwnsTensor(source)) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "Graph::CreateOp: source tensor is null or not owned by graph: " +
              name);
    }
  }

  // 先计算并校验全部输出 tensor 名称，避免提交后才发现命名冲突。
  std::vector<std::string> output_names;
  output_names.reserve(out_cnt);
  for (size_t index = 0; index < out_cnt; ++index) {
    const std::string tensor_name =
        out_cnt == 1 ? name + ".out"
                     : name + ".out." + std::to_string(index);
    status = ValidateNewTensorName(tensor_name);
    if (!status.ok()) {
      return status;
    }
    output_names.push_back(tensor_name);
  }

  // 校验通过后提交 op 与输出 tensor 的连接关系。
  auto op = std::make_unique<Op>(name, op_type, attr);
  Op* op_ptr = op.get();
  for (Tensor* source : src_tensors) {
    op_ptr->AddSrcTensor(source);
    source->AddDstOp(op_ptr);
  }

  std::vector<Tensor*> created_tensors;
  created_tensors.reserve(out_cnt);
  for (const std::string& tensor_name : output_names) {
    auto tensor = std::make_unique<Tensor>(tensor_name);
    Tensor* tensor_ptr = tensor.get();
    tensor_ptr->AddSrcOp(op_ptr);
    op_ptr->AddDstTensor(tensor_ptr);
    tensor_map_.emplace(tensor_name, std::move(tensor));
    created_tensors.push_back(tensor_ptr);
  }
  op_map_.emplace(name, std::move(op));

  *outputs = std::move(created_tensors);
  return Status::Ok();
}

Status Graph::CreateConstant(const std::string& name, const Storage& storage,
                             Tensor** output) {
  Status status = ValidateNewTensorName(name);
  if (!status.ok()) {
    return status;
  }
  if (output == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Graph::CreateConstant: output must not be null");
  }
  if (storage.shape.GetShapeSize() < 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Graph::CreateConstant: invalid storage shape for " +
                          name);
  }

  auto tensor = std::make_unique<Tensor>(name);
  Storage stored_storage = storage;
  stored_storage.reusable = false;
  tensor->GetStorage() = stored_storage;
  tensor->SetConstant(true);

  Tensor* tensor_ptr = tensor.get();
  constant_tensor_list_.push_back(tensor_ptr);
  constant_tensor_map_.emplace(name, std::move(tensor));
  *output = tensor_ptr;
  return Status::Ok();
}

Status Graph::CreateInput(const std::string& name, Tensor** output) {
  return CreateInputInternal(name, Storage{}, output);
}

Status Graph::CreateInput(const std::string& name, Precision precision,
                          Layout layout, Tensor** output) {
  Storage storage;
  storage.precision = precision;
  storage.layout = layout;
  return CreateInputInternal(name, storage, output);
}

Status Graph::CreateOutput(const std::string& name, Tensor* input) {
  Status status = ValidateNewOpName(name);
  if (!status.ok()) {
    return status;
  }
  if (!OwnsTensor(input)) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Graph::CreateOutput: input tensor is null or not owned by graph: " +
            name);
  }

  AttrMap attr;
  attr["storage"] = Storage{};
  auto op = std::make_unique<Op>(name, OpType::kOutput, attr);
  Op* op_ptr = op.get();
  op_ptr->AddSrcTensor(input);
  input->AddDstOp(op_ptr);

  op_map_.emplace(name, std::move(op));
  output_op_list_.push_back(op_ptr);
  return Status::Ok();
}

Status Graph::SortOp() {
  op_list_.clear();
  if (output_op_list_.empty()) {
    return MakeStatus(ErrorCode::kStateError,
                      "Graph::SortOp: graph has no output op");
  }

  std::map<const Op*, int> states;
  for (Op* output_op : output_op_list_) {
    if (output_op == nullptr) {
      op_list_.clear();
      return MakeStatus(ErrorCode::kStateError,
                        "Graph::SortOp: null output op");
    }
    const Status status = VisitOp(output_op, &states);
    if (!status.ok()) {
      op_list_.clear();
      return status;
    }
  }
  return Status::Ok();
}

const std::vector<Tensor*>& Graph::GetAllConstantTensor() {
  return constant_tensor_list_;
}

const std::vector<Op*>& Graph::GetAllInputOp() { return input_op_list_; }

const std::vector<Op*>& Graph::GetAllOutputOp() { return output_op_list_; }

const std::vector<Op*>& Graph::GetOrderlyOPList() { return op_list_; }

Status Graph::SetGraphEnv(std::unique_ptr<GraphEnv> graph_env) {
  if (graph_env == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Graph::SetGraphEnv: graph_env must not be null");
  }
  graph_env_ = std::move(graph_env);
  return Status::Ok();
}

GraphEnv* Graph::GetGraphEnv() { return graph_env_.get(); }

void Graph::ClearGraphEnv() { graph_env_.reset(); }

bool Graph::OwnsTensor(const Tensor* tensor) const {
  if (tensor == nullptr) {
    return false;
  }
  for (const auto& entry : tensor_map_) {
    if (entry.second.get() == tensor) {
      return true;
    }
  }
  for (const auto& entry : constant_tensor_map_) {
    if (entry.second.get() == tensor) {
      return true;
    }
  }
  return false;
}

Status Graph::ValidateNewOpName(const std::string& name) const {
  if (name.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Graph: op name must not be empty");
  }
  if (op_map_.find(name) != op_map_.end()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Graph: duplicate op name: " + name);
  }
  return Status::Ok();
}

Status Graph::ValidateNewTensorName(const std::string& name) const {
  if (name.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Graph: tensor name must not be empty");
  }
  if (tensor_map_.find(name) != tensor_map_.end() ||
      constant_tensor_map_.find(name) != constant_tensor_map_.end()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Graph: duplicate tensor name: " + name);
  }
  return Status::Ok();
}

Status Graph::CreateInputInternal(const std::string& name,
                                  const Storage& attr_storage,
                                  Tensor** output) {
  Status status = ValidateNewOpName(name);
  if (!status.ok()) {
    return status;
  }
  status = ValidateNewTensorName(name);
  if (!status.ok()) {
    return status;
  }
  if (output == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Graph::CreateInput: output must not be null");
  }

  AttrMap attr;
  attr["storage"] = attr_storage;
  auto op = std::make_unique<Op>(name, OpType::kInput, attr);
  Op* op_ptr = op.get();

  auto tensor = std::make_unique<Tensor>(name);
  Tensor* tensor_ptr = tensor.get();
  op_ptr->AddDstTensor(tensor_ptr);
  tensor_ptr->AddSrcOp(op_ptr);

  op_map_.emplace(name, std::move(op));
  tensor_map_.emplace(name, std::move(tensor));
  input_op_list_.push_back(op_ptr);
  *output = tensor_ptr;
  return Status::Ok();
}

Status Graph::VisitOp(Op* op, std::map<const Op*, int>* states) {
  if (op == nullptr) {
    return MakeStatus(ErrorCode::kStateError, "Graph::SortOp: null op");
  }
  int& state = (*states)[op];
  if (state == 1) {
    return MakeStatus(ErrorCode::kStateError,
                      "Graph::SortOp: dependency cycle at op: " + op->Name());
  }
  if (state == 2) {
    return Status::Ok();
  }

  state = 1;
  for (Tensor* tensor : op->SrcTensors()) {
    if (tensor == nullptr) {
      return MakeStatus(ErrorCode::kStateError,
                        "Graph::SortOp: null source tensor at op: " +
                            op->Name());
    }
    if (tensor->IsConstant()) {
      // constant tensor 是依赖叶子，不要求存在 producer op。
      continue;
    }
    const std::vector<Op*>& producers = tensor->SrcOps();
    if (producers.empty()) {
      return MakeStatus(ErrorCode::kStateError,
                        "Graph::SortOp: tensor has no producer: " +
                            tensor->Name());
    }
    for (Op* producer : producers) {
      const Status status = VisitOp(producer, states);
      if (!status.ok()) {
        return status;
      }
    }
  }

  state = 2;
  op_list_.push_back(op);
  return Status::Ok();
}

}  // namespace runtime_graph
}  // namespace mobilenet
