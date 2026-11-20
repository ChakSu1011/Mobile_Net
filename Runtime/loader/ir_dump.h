// 模块：Runtime C++ 版 dump_ir
//
// 职责：
//   以 C++ 生成与 `Graph/python/graph_python.cc` 中 DumpIr 逐字节一致的
//   IR 文本，并可直接计算其 SHA-256，供 Loader 在绑定任何 data_ptr 之前
//   校验 manifest 的 graph.ir_sha256。
//
// 边界：
//   - 只读取 Graph 的公共查询接口，不修改图、不访问 tensor 数据；
//   - 输出格式是格式契约（G1/G2），任何改动都必须同步 Python 绑定与
//     manifest 生成逻辑；
//   - 不输出真实指针值，常量 storage 只输出 data=set/unset。
//
// 依赖：
//   Graph/include/{graph,op,tensor}.h、Runtime/common/sha256.h、
//   C++17 标准库。

#ifndef MOBILENET_RUNTIME_LOADER_IR_DUMP_H_
#define MOBILENET_RUNTIME_LOADER_IR_DUMP_H_

#include <string>

#include "Graph/include/graph.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {

// 生成只读 IR 文本。
//
// 功能：
//   依次输出 graph/inputs/constants/ops/outputs 五段；ops 段使用
//   GetOrderlyOPList 的拓扑序，属性按键排序（AttrMap 天然有序）。
//
// 参数：
//   graph：已完成构图并调用过 SortOp 的图。
// 返回：
//   与 Python DumpIr 逐字节一致的文本；末尾带换行。
std::string DumpIr(runtime_graph::Graph& graph);

// 计算 DumpIr 的 SHA-256。
//
// 参数：
//   graph：目标图。
// 返回：
//   64 字符小写十六进制摘要。
std::string DumpIrSha256Hex(runtime_graph::Graph& graph);

// 统计 IR 文本行数。
//
// 功能：
//   按 '\n' 计数，与 Python `len(text.splitlines())` 对以换行结尾的
//   DumpIr 输出一致。
//
// 参数：
//   text：DumpIr 文本。
// 返回：
//   行数。
Usize CountIrLines(const std::string& text);

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_LOADER_IR_DUMP_H_
