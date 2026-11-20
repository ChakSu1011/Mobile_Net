// 模块：Runtime 流式 SHA-256
//
// 职责：
//   提供与 Python ``hashlib.sha256(...).hexdigest()`` 一致的流式 SHA-256
//   实现，供模型包 manifest/权重文件校验与 IR 哈希比对使用。
//
// 边界：
//   - 只实现 FIPS 180-4 定义的 SHA-256，不支持 SHA-224/384/512；
//   - 不做文件读取；调用方负责分块读取并通过 Update 喂入；
//   - 输出为小写十六进制字符串，与 Python ``bytes.hex()`` 一致；
//   - 不依赖 OpenSSL 等第三方库。
//
// 依赖：
//   C++17 标准库、Utils/cpp/types.h。

#ifndef MOBILENET_RUNTIME_COMMON_SHA256_H_
#define MOBILENET_RUNTIME_COMMON_SHA256_H_

#include <string>

#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {

// 流式 SHA-256 计算器。
//
// 功能：
//   通过多次 Update 累积任意长度字节流，Final/HexDigest 得到 32 字节摘要；
//   用于大文件（1 GB 以上）分块哈希，避免一次性读入内存。
//
// 边界：
//   - 非线程安全，一个实例只允许单线程使用；
//   - Final 之后 Update 为无操作；重复调用 Final/HexDigest 返回同一摘要；
//   - 不抛出异常，不检查输入指针空指针（size 为 0 时允许空指针）。
class Sha256 {
 public:
  // 构造空哈希上下文。
  Sha256();

  ~Sha256() = default;

  Sha256(const Sha256&) = default;
  Sha256& operator=(const Sha256&) = default;
  Sha256(Sha256&&) = default;
  Sha256& operator=(Sha256&&) = default;

  // 追加一段字节。
  //
  // 参数：
  //   data：输入字节，size 大于 0 时必须非空；
  //   size：字节数，允许为 0。
  // 返回：
  //   无；内部累积到 64 字节块后立即压缩。
  void Update(const void* data, Usize size);

  // 结束计算并输出 32 字节摘要。
  //
  // 参数：
  //   digest：输出数组，长度必须至少 32 字节，不允许为空。
  // 返回：
  //   无；首次调用执行填充与压缩，后续调用直接复制结果。
  void Final(U8 digest[32]);

  // 返回小写十六进制摘要（64 个字符）。
  std::string HexDigest();

  // 一次性计算字节串的 SHA-256 十六进制摘要。
  //
  // 参数：
  //   data：输入字节，size 大于 0 时必须非空；
  //   size：字节数。
  // 返回：
  //   64 字符小写十六进制字符串。
  static std::string Hex(const void* data, Usize size);

 private:
  // 压缩一个 64 字节块并更新内部状态。
  //
  // 参数：
  //   block：指向 64 字节块，不允许为空。
  void Transform(const U8* block);

  // 完成填充与摘要计算；幂等。
  void Finish();

  // 8 个 32 位状态字。
  U32 state_[8];

  // 未压缩的尾部输入，长度小于 64。
  U8 buffer_[64];

  // buffer_ 中有效字节数。
  Usize buffer_size_ = 0;

  // 已喂入的总字节数。
  U64 total_bytes_ = 0;

  // 是否已完成计算。
  Bool finalized_ = false;

  // 完成后的 32 字节摘要。
  U8 digest_[32] = {};
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_SHA256_H_
