// 模块：Runtime C++ 版 dump_ir 实现
//
// 职责：
//   逐行复刻 Graph/python/graph_python.cc 的 DumpIr 输出，并提供哈希与
//   行数统计。
//
// 边界：
//   - 只读 Graph；不修改 Storage、不创建 OpEnv；
//   - 格式字符串与 Python 版本必须保持完全一致（G1/G2 契约）。
//
// 依赖：
//   Runtime/loader/ir_dump.h、Graph/include/{attr,op,op_type,storage,
//   tensor}.h、Runtime/common/sha256.h、C++17 标准库。

#include "Runtime/loader/ir_dump.h"

#include <sstream>
#include <string>
#include <vector>

#include "Graph/include/attr.h"
#include "Graph/include/op.h"
#include "Graph/include/op_type.h"
#include "Graph/include/storage.h"
#include "Graph/include/tensor.h"
#include "Runtime/common/sha256.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// 返回 tensor 名称列表的文本表示。
//
// 参数：
//   tensors：tensor 指针列表，允许包含空指针。
// 返回：
//   形如 "[a,b]" 的字符串；空指针输出 "<null>"。
std::string JoinTensorNames(const std::vector<runtime_graph::Tensor*>& tensors) {
  std::ostringstream stream;
  stream << "[";
  for (Usize index = 0; index < tensors.size(); ++index) {
    if (index != 0) {
      stream << ",";
    }
    stream << (tensors[index] == nullptr ? "<null>"
                                         : tensors[index]->Name());
  }
  stream << "]";
  return stream.str();
}

}  // namespace

std::string DumpIr(runtime_graph::Graph& graph) {
  std::ostringstream stream;
  stream << "graph " << graph.Name() << "\n";

  stream << "inputs " << graph.GetAllInputOp().size() << "\n";
  for (runtime_graph::Op* op : graph.GetAllInputOp()) {
    stream << "  " << op->Name() << " -> "
           << JoinTensorNames(op->DstTensors()) << "\n";
  }

  stream << "constants " << graph.GetAllConstantTensor().size() << "\n";
  for (runtime_graph::Tensor* tensor : graph.GetAllConstantTensor()) {
    stream << "  " << tensor->Name() << " "
           << runtime_graph::StorageToString(tensor->GetStorage()) << "\n";
  }

  stream << "ops " << graph.GetOrderlyOPList().size() << "\n";
  const std::vector<runtime_graph::Op*>& ordered_ops =
      graph.GetOrderlyOPList();
  for (Usize index = 0; index < ordered_ops.size(); ++index) {
    runtime_graph::Op* op = ordered_ops[index];
    stream << "  [" << index << "] " << op->Name() << " "
           << runtime_graph::OpTypeToStr(op->Type())
           << " src=" << JoinTensorNames(op->SrcTensors())
           << " dst=" << JoinTensorNames(op->DstTensors()) << " attrs={";
    Bool first = true;
    for (const auto& entry : op->Attrs()) {
      if (!first) {
        stream << ",";
      }
      first = false;
      stream << entry.first << "="
             << runtime_graph::AttrValueToString(entry.second);
    }
    stream << "}\n";
  }

  stream << "outputs " << graph.GetAllOutputOp().size() << "\n";
  for (runtime_graph::Op* op : graph.GetAllOutputOp()) {
    stream << "  " << op->Name() << " <- "
           << JoinTensorNames(op->SrcTensors()) << "\n";
  }
  return stream.str();
}

std::string DumpIrSha256Hex(runtime_graph::Graph& graph) {
  const std::string ir = DumpIr(graph);
  return Sha256::Hex(ir.data(), ir.size());
}

Usize CountIrLines(const std::string& text) {
  Usize lines = 0;
  for (const C8 ch : text) {
    if (ch == '\n') {
      ++lines;
    }
  }
  return lines;
}

}  // namespace runtime
}  // namespace mobilenet
