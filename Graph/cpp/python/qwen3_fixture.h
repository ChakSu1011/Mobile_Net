// 模块：Qwen3-0.6B 参考图构建器（测试夹具）
//
// 职责：
//   在 C++ 侧使用 at::Tensor 与 GraphBuilder 构造 Qwen3-0.6B 的完整融合
//   算子图，用于与 Python 侧参考图做 Canonical JSON 逐字节比较。
//
// 边界：
//   - 仅用于测试/工具链，不进入移动端运行时；
//   - 张量使用 meta device（权重、设备边）或 CPU 小张量（I/O host_tensor），
//     不读取真实权重数据；
//   - 模型超参数固定为 Qwen3-0.6B 的 config.json；新增模型族应另立夹具。
//
// 依赖：
//   graph/graph.h、graph/graph_builder.h、Python/torch 桥接。

#ifndef MOBILENET_GRAPH_CPP_PYTHON_QWEN3_FIXTURE_H_
#define MOBILENET_GRAPH_CPP_PYTHON_QWEN3_FIXTURE_H_

#include <memory>

#include "Runtime/common/status.h"
#include "graph/graph.h"

namespace mobilenet {
namespace graph {

// 构造 Qwen3-0.6B 参考图。
//
// 参数：
//   status：输出状态；可为空。成功时写入 Status::Ok()，失败时写入具体错误。
// 返回：
//   成功返回非空 Graph；失败返回 nullptr。
std::unique_ptr<Graph> BuildQwen3ReferenceGraph(Status* status);

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_PYTHON_QWEN3_FIXTURE_H_
