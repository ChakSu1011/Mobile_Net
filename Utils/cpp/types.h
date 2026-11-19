// 模块：全工程 C++ Host 端类型别名
//
// 职责：
//   为 Graph、Runtime、Tests 与 Tools 的 C++ 代码提供统一的 Host 端数据
//   类型别名，避免各模块直接使用 uint64_t/int32_t/size_t/float/double 等
//   原始类型名。
//
// 边界：
//   - 只定义类型别名，不引入任何运行时逻辑或第三方依赖；
//   - 别名位于全局命名空间，便于在 mobilenet/runtime_graph 与测试 main
//     中直接使用；
//   - 其它头文件不得重复定义同名别名；
//   - 第三方/厂商头文件在其公开 API 中出现的原始类型保持原样，不在本文件
//     展开。
//
// 依赖：
//   C++17 标准库 <cstddef>、<cstdint>。

#ifndef MOBILENET_UTILS_CPP_TYPES_H_
#define MOBILENET_UTILS_CPP_TYPES_H_

#include <cstddef>
#include <cstdint>

// 无符号整数类型。
using U8 = std::uint8_t;
using U16 = std::uint16_t;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

// 有符号整数类型。
using I8 = std::int8_t;
using I16 = std::int16_t;
using I32 = std::int32_t;
using I64 = std::int64_t;

// 浮点类型。
using F32 = float;
using F64 = double;

// 布尔类型。
using Bool = bool;

// 字符/字节文本类型。
using C8 = char;

// 尺寸与指针差值类型。
using Usize = std::size_t;
using Isize = std::ptrdiff_t;

#endif  // MOBILENET_UTILS_CPP_TYPES_H_
