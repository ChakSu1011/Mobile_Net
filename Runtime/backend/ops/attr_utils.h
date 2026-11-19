// 模块：算子属性严格读取工具
//
// 职责：
//   为 OpSpec 与 CPU Kernel 提供类型严格的 attr 读取：缺失、类型不符或
//   空名称统一返回带属性名的错误，避免各算子重复手写 variant 判断。
//
// 边界：
//   - 只读取 Graph AttrMap，不修改属性、不做默认值推断；
//   - 不做语义/范围校验（由各 OpSpec 负责）；
//   - 不依赖设备或 Kernel 实现。
//
// 依赖：
//   Graph/include/attr.h、graph/op.h、Runtime/common/status.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OPS_ATTR_UTILS_H_
#define MOBILENET_RUNTIME_BACKEND_OPS_ATTR_UTILS_H_

#include <cstdint>
#include <string>

#include "Graph/include/attr.h"
#include "Graph/include/op.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {

// 判断属性是否存在。
//
// 参数：
//   attrs：属性表；
//   name：属性名，非空。
// 返回：
//   存在返回 true。
Bool HasAttr(const runtime_graph::AttrMap& attrs, const std::string& name);

// 读取 int64 属性。
//
// 参数：
//   attrs：属性表；
//   name：属性名；
//   value：输出参数，成功时写入。
// 返回：
//   成功返回 kOk；缺失/类型不符返回 kInvalidArgument。
Status GetIntAttr(const runtime_graph::AttrMap& attrs, const C8* name,
                  I64* value);

// 读取浮点属性；允许 int64 自动提升为 FP64。
//
// 参数：
//   attrs：属性表；
//   name：属性名；
//   value：输出参数，成功时写入。
// 返回：
//   成功返回 kOk；缺失/类型不符返回 kInvalidArgument。
Status GetFloatAttr(const runtime_graph::AttrMap& attrs, const C8* name,
                    FP64* value);

// 读取 Bool 属性。
Status GetBoolAttr(const runtime_graph::AttrMap& attrs, const C8* name,
                   Bool* value);

// 读取 string 属性。
Status GetStringAttr(const runtime_graph::AttrMap& attrs, const C8* name,
                     std::string* value);

// 读取 op 的 int64 属性；错误消息带 op 名。
Status GetIntAttr(const runtime_graph::Op& op, const C8* name,
                  I64* value);

// 读取 op 的浮点属性；错误消息带 op 名。
Status GetFloatAttr(const runtime_graph::Op& op, const C8* name,
                    FP64* value);

// 读取 op 的 Bool 属性；错误消息带 op 名。
Status GetBoolAttr(const runtime_graph::Op& op, const C8* name,
                   Bool* value);

// 读取 op 的 string 属性；错误消息带 op 名。
Status GetStringAttr(const runtime_graph::Op& op, const C8* name,
                     std::string* value);

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OPS_ATTR_UTILS_H_
