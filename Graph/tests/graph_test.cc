// 模块：Runtime Graph C++ 单元测试
//
// 职责：
//   验证 Shape/Storage 基础语义、Graph 构图与拓扑排序、OpEnv 生命周期、
//   GraphEnv 共享存储以及主要错误路径；测试只使用内存元数据，不加载模型。
//
// 边界：
//   - 不测试真实 kernel、显存分配或 Python 绑定；
//   - 不覆盖无法通过公开接口构造的依赖环场景；
//   - 失败时返回非 0，成功时打印 ALL TESTS PASSED。
//
// 依赖：
//   runtime_graph 静态库、C++17 标准库。

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "runtime_graph/graph.h"
#include "runtime_graph/graph_env.h"
#include "runtime_graph/op.h"
#include "runtime_graph/op_env.h"
#include "runtime_graph/op_type.h"
#include "runtime_graph/shape.h"
#include "runtime_graph/storage.h"
#include "runtime_graph/tensor.h"
#include "runtime_graph/types.h"

namespace {

using mobilenet::ErrorCode;
using mobilenet::Status;
using mobilenet::runtime_graph::AttrMap;
using mobilenet::runtime_graph::DeviceType;
using mobilenet::runtime_graph::Graph;
using mobilenet::runtime_graph::GraphEnv;
using mobilenet::runtime_graph::kMaxShapeDim;
using mobilenet::runtime_graph::Layout;
using mobilenet::runtime_graph::MegaAttentionOpEnv;
using mobilenet::runtime_graph::Op;
using mobilenet::runtime_graph::OpEnv;
using mobilenet::runtime_graph::OpType;
using mobilenet::runtime_graph::Precision;
using mobilenet::runtime_graph::Shape;
using mobilenet::runtime_graph::Storage;
using mobilenet::runtime_graph::StorageToString;
using mobilenet::runtime_graph::Tensor;

// 计数 OpEnv：用于验证 Graph 析构时会主动清理挂载的 OpEnv。
class CountingOpEnv : public OpEnv {
 public:
  // 创建计数执行环境。
  //
  // 参数：
  //   op：所属 op；
  //   destroyed_count：析构次数计数器，生命周期由调用方保证。
  CountingOpEnv(Op* op, int* destroyed_count)
      : OpEnv(op), destroyed_count_(destroyed_count) {}

  ~CountingOpEnv() override { ++(*destroyed_count_); }

  // 本测试不执行真实计算。
  Status Forward() override { return Status::Ok(); }

  // 本测试不需要临时数据。
  bool NeedTempData() const override { return false; }

 private:
  int* destroyed_count_ = nullptr;
};

// 断言辅助宏：条件失败时打印文件、行号和表达式，并从当前测试函数返回 false。
#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": "            \
                << #condition << std::endl;                                  \
      return false;                                                          \
    }                                                                        \
  } while (false)

// 测试 Shape 的默认值、正常乘积、负维度和溢出保护。
//
// 返回：
//   true 表示全部断言通过；false 表示存在失败。
bool TestShape() {
  Shape shape;
  CHECK(shape.max_dim == 0);
  CHECK(shape.GetShapeSize() == 1);
  CHECK(shape.GetByteSize() == 1);

  shape.dims[0] = 2;
  shape.dims[1] = 3;
  shape.pitch[0] = 4;
  shape.pitch[1] = 3;
  shape.max_dim = 2;
  CHECK(shape.GetShapeSize() == 6);
  CHECK(shape.GetByteSize() == 12);

  shape.max_dim = kMaxShapeDim + 1;
  CHECK(shape.GetShapeSize() == -1);
  CHECK(shape.GetByteSize() == -1);

  Shape negative;
  negative.max_dim = 1;
  negative.dims[0] = -1;
  CHECK(negative.GetShapeSize() == -1);

  Shape overflow;
  overflow.max_dim = 2;
  overflow.dims[0] = std::numeric_limits<int64_t>::max();
  overflow.dims[1] = 2;
  CHECK(overflow.GetShapeSize() == -1);
  return true;
}

// 测试 Graph 的 constant/input/compute/output 构图与拓扑排序顺序。
//
// 返回：
//   true 表示全部断言通过；false 表示存在失败。
bool TestGraphConstruction() {
  std::vector<uint16_t> weight_data(6, 0);
  Storage weight_storage;
  weight_storage.shape.max_dim = 2;
  weight_storage.shape.dims[0] = 2;
  weight_storage.shape.dims[1] = 3;
  weight_storage.shape.pitch[0] = 2;
  weight_storage.shape.pitch[1] = 3;
  weight_storage.precision = Precision::kFloat16;
  weight_storage.layout = Layout::kRowMajor;
  weight_storage.device_type = DeviceType::kHost;
  weight_storage.byte_size = 12;
  weight_storage.data_ptr = weight_data.data();
  weight_storage.reusable = true;

  Graph graph("unit_graph");
  Tensor* weight = nullptr;
  CHECK(graph.CreateConstant("weight", weight_storage, &weight).ok());
  CHECK(weight != nullptr);
  CHECK(weight->IsConstant());
  CHECK(!weight->Reusable());
  CHECK(weight->SrcOps().empty());
  CHECK(StorageToString(weight->GetStorage()).find("float16") !=
        std::string::npos);
  CHECK(StorageToString(weight->GetStorage()).find("data=set") !=
        std::string::npos);

  Tensor* input = nullptr;
  CHECK(graph.CreateInput("input", Precision::kFloat16, Layout::kRowMajor,
                          &input)
            .ok());
  CHECK(input != nullptr);
  CHECK(!input->IsConstant());
  CHECK(graph.GetAllInputOp().size() == 1);

  const Op* input_op = graph.GetAllInputOp()[0];
  const auto storage_iterator = input_op->Attrs().find("storage");
  CHECK(storage_iterator != input_op->Attrs().end());
  const Storage* input_attr_storage =
      std::get_if<Storage>(&storage_iterator->second);
  CHECK(input_attr_storage != nullptr);
  CHECK(input_attr_storage->precision == Precision::kFloat16);
  CHECK(input_attr_storage->layout == Layout::kRowMajor);

  std::vector<Tensor*> linear_sources{input, weight};
  AttrMap linear_attr;
  linear_attr["axis"] = int64_t{1};
  std::vector<Tensor*> linear_outputs;
  CHECK(graph
            .CreateOp("linear", linear_sources, 1, OpType::kLinear, linear_attr,
                      &linear_outputs)
            .ok());
  CHECK(linear_outputs.size() == 1);
  Tensor* linear_output = linear_outputs[0];
  CHECK(linear_output->SrcOps().size() == 1);
  CHECK(linear_output->SrcOps()[0]->Name() == "linear");
  CHECK(linear_output->DstOps().empty());
  CHECK(weight->DstOps().size() == 1);
  CHECK(weight->DstOps()[0]->Name() == "linear");

  std::vector<Tensor*> add_sources{linear_output};
  std::vector<Tensor*> add_outputs;
  CHECK(graph.CreateOp("add", add_sources, 1, OpType::kAdd, AttrMap{},
                       &add_outputs)
            .ok());
  CHECK(add_outputs.size() == 1);
  CHECK(linear_output->DstOps().size() == 1);
  CHECK(linear_output->DstOps()[0]->Name() == "add");

  CHECK(graph.CreateOutput("output", add_outputs[0]).ok());
  CHECK(graph.GetAllOutputOp().size() == 1);
  CHECK(graph.SortOp().ok());

  const std::vector<Op*>& ordered = graph.GetOrderlyOPList();
  CHECK(ordered.size() == 4);
  CHECK(ordered[0]->Type() == OpType::kInput);
  CHECK(ordered[1]->Name() == "linear");
  CHECK(ordered[1]->Type() == OpType::kLinear);
  CHECK(ordered[2]->Name() == "add");
  CHECK(ordered[2]->Type() == OpType::kAdd);
  CHECK(ordered[3]->Type() == OpType::kOutput);
  CHECK(graph.GetAllConstantTensor().size() == 1);

  CHECK(ordered[1]->SetAttr("eps", 1e-6).ok());
  CHECK(!ordered[1]->SetAttr("", int64_t{1}).ok());
  CHECK(!ordered[1]->SetOpEnv(nullptr).ok());

  CHECK(graph.SortOp().ok());
  CHECK(graph.GetOrderlyOPList().size() == 4);
  return true;
}

// 测试 GraphEnv 共享存储与 MegaAttentionOpEnv 生命周期。
//
// 返回：
//   true 表示全部断言通过；false 表示存在失败。
bool TestEnvironment() {
  GraphEnv graph_env;
  Storage shared_storage;
  shared_storage.shape.max_dim = 1;
  shared_storage.shape.dims[0] = 64;
  shared_storage.shape.pitch[0] = 64;
  shared_storage.precision = Precision::kFloat16;
  shared_storage.byte_size = 128;
  CHECK(graph_env.AddSharedStorage("kv_cache", shared_storage).ok());
  CHECK(!graph_env.AddSharedStorage("kv_cache", shared_storage).ok());
  CHECK(graph_env.GetSharedStorage("kv_cache") != nullptr);
  CHECK(graph_env.GetSharedStorage("missing") == nullptr);
  CHECK(graph_env.SharedStorages().size() == 1);

  auto owned_env = std::make_unique<GraphEnv>();
  CHECK(owned_env->AddSharedStorage("kv_cache", shared_storage).ok());
  GraphEnv* graph_env_ptr = owned_env.get();

  Graph graph("env_graph");
  CHECK(graph.SetGraphEnv(std::move(owned_env)).ok());
  CHECK(graph.GetGraphEnv() == graph_env_ptr);

  Tensor* input = nullptr;
  CHECK(graph.CreateInput("input", &input).ok());
  std::vector<Tensor*> attention_outputs;
  CHECK(graph
            .CreateOp("attention", {input}, 1, OpType::kMegaAttention,
                      AttrMap{}, &attention_outputs)
            .ok());
  CHECK(graph.CreateOutput("output", attention_outputs[0]).ok());
  CHECK(graph.SortOp().ok());
  CHECK(graph.GetOrderlyOPList().size() == 3);

  Op* attention_op = graph.GetOrderlyOPList()[1];
  CHECK(attention_op->Type() == OpType::kMegaAttention);
  auto attention_env = std::make_unique<MegaAttentionOpEnv>(attention_op);
  MegaAttentionOpEnv* attention_env_ptr = attention_env.get();
  CHECK(attention_op->SetOpEnv(std::move(attention_env)).ok());
  CHECK(attention_op->GetOpEnv() == attention_env_ptr);
  CHECK(attention_env_ptr->NeedTempData());
  CHECK(attention_env_ptr->Init(graph.GetGraphEnv()).ok());
  const Status forward_status = attention_env_ptr->Forward();
  CHECK(!forward_status.ok());
  CHECK(forward_status.code() == ErrorCode::kUnsupportedOp);
  attention_op->ClearOpEnv();
  CHECK(attention_op->GetOpEnv() == nullptr);

  CHECK(!graph.SetGraphEnv(nullptr).ok());
  CHECK(graph_env_ptr->GetSharedStorage("kv_cache") != nullptr);
  graph.ClearGraphEnv();
  CHECK(graph.GetGraphEnv() == nullptr);

  // Graph 析构时应主动清理挂载的 OpEnv，且不应访问已释放的 GraphEnv。
  int destroyed_count = 0;
  {
    Graph scoped_graph("scoped_graph");
    auto scoped_env = std::make_unique<GraphEnv>();
    CHECK(scoped_env->AddSharedStorage("shared", shared_storage).ok());
    CHECK(scoped_graph.SetGraphEnv(std::move(scoped_env)).ok());
    Tensor* scoped_input = nullptr;
    CHECK(scoped_graph.CreateInput("x", &scoped_input).ok());
    std::vector<Tensor*> scoped_outputs;
    CHECK(scoped_graph
              .CreateOp("op", {scoped_input}, 1, OpType::kAdd, AttrMap{},
                        &scoped_outputs)
              .ok());
    CHECK(scoped_graph.CreateOutput("y", scoped_outputs[0]).ok());
    CHECK(scoped_graph.SortOp().ok());
    Op* scoped_op = scoped_graph.GetOrderlyOPList()[1];
    CHECK(scoped_op
              ->SetOpEnv(
                  std::make_unique<CountingOpEnv>(scoped_op, &destroyed_count))
              .ok());
  }
  CHECK(destroyed_count == 1);
  return true;
}

// 测试主要错误路径：空名称、重复名称、外部 tensor、空参数和非法 shape。
//
// 返回：
//   true 表示全部断言通过；false 表示存在失败。
bool TestErrors() {
  Graph graph("errors");
  std::vector<Tensor*> outputs;
  CHECK(!graph.SortOp().ok());

  CHECK(!graph.CreateOp("", {}, 1, OpType::kAdd, AttrMap{}, &outputs).ok());
  CHECK(!graph.CreateOp("zero_out", {}, 0, OpType::kAdd, AttrMap{}, &outputs)
             .ok());
  CHECK(!graph.CreateOp("null_out", {}, 1, OpType::kAdd, AttrMap{}, nullptr)
             .ok());
  CHECK(!graph.CreateOp("input_type", {}, 1, OpType::kInput, AttrMap{},
                        &outputs)
             .ok());

  Tensor* input = nullptr;
  CHECK(graph.CreateInput("x", &input).ok());
  CHECK(graph.CreateOp("op", {input}, 1, OpType::kAdd, AttrMap{}, &outputs)
            .ok());
  CHECK(!graph.CreateOp("op", {input}, 1, OpType::kAdd, AttrMap{}, &outputs)
             .ok());

  std::vector<Tensor*> null_sources{nullptr};
  CHECK(!graph.CreateOp("null_src", null_sources, 1, OpType::kAdd, AttrMap{},
                        &outputs)
             .ok());

  Storage constant_storage;
  constant_storage.shape.max_dim = 1;
  constant_storage.shape.dims[0] = 1;
  constant_storage.shape.pitch[0] = 1;
  constant_storage.byte_size = 2;
  Tensor* constant = nullptr;
  CHECK(!graph.CreateConstant("x", constant_storage, &constant).ok());
  CHECK(!graph.CreateConstant("ok_constant", constant_storage, nullptr).ok());

  Storage bad_shape;
  bad_shape.shape.max_dim = kMaxShapeDim + 1;
  CHECK(!graph.CreateConstant("bad_shape", bad_shape, &constant).ok());

  Graph other("other");
  Tensor* other_input = nullptr;
  CHECK(other.CreateInput("other_input", &other_input).ok());
  CHECK(!graph.CreateOp("foreign_src", {other_input}, 1, OpType::kAdd,
                        AttrMap{}, &outputs)
             .ok());
  CHECK(!graph.CreateOutput("foreign_output", other_input).ok());
  CHECK(!graph.CreateOutput("", input).ok());
  return true;
}

}  // namespace

// 运行全部 C++ 单元测试。
//
// 返回：
//   0 表示全部通过；1 表示存在失败。
int main() {
  if (!TestShape()) {
    return 1;
  }
  if (!TestGraphConstruction()) {
    return 1;
  }
  if (!TestEnvironment()) {
    return 1;
  }
  if (!TestErrors()) {
    return 1;
  }
  std::cout << "graph_test: ALL TESTS PASSED" << std::endl;
  return 0;
}
