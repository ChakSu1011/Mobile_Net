// 模块：SHA-256 工具
//
// 职责：
//   为 OCL Program/二进制缓存生成稳定哈希键。
//
// 边界：
//   - 只提供字符串 SHA-256 十六进制；不用于安全认证；
//   - 无全局状态；线程安全。
//
// 依赖：
//   C++17 标准库、Utils/cpp/types.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_SHA256_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_SHA256_H_

#include <string>

namespace mobilenet {
namespace ocl {

// 计算字符串的 SHA-256，返回 64 位小写十六进制。
//
// 参数：
//   input：任意字节内容（可含 '\0'）。
// 返回：
//   64 字符十六进制字符串。
std::string Sha256Hex(const std::string& input);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_SHA256_H_
