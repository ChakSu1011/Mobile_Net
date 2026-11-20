// 模块：Runtime 流式 SHA-256 实现
//
// 职责：
//   实现 Sha256 的 FIPS 180-4 压缩函数、填充规则与十六进制输出。
//
// 边界：
//   - 只处理 SHA-256；不读取文件、不校验文件大小；
//   - 不分配堆内存，所有工作区位于栈或对象成员；
//   - 输入指针由调用方保证有效。
//
// 依赖：
//   Runtime/common/sha256.h、C++17 标准库。

#include "Runtime/common/sha256.h"

#include <cstring>

#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// SHA-256 轮常量，来自 FIPS 180-4 第 4.2.2 节。
constexpr U32 kRoundConstants[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

// 循环右移。
//
// 参数：
//   value：待移位数值；
//   bits：移位数，取值 [1,31]。
// 返回：
//   循环右移结果。
inline U32 RotateRight(U32 value, U32 bits) {
  return (value >> bits) | (value << (32U - bits));
}

// 读取 4 字节大端无符号整数。
//
// 参数：
//   data：至少 4 字节的输入。
// 返回：
//   大端解释的 32 位值。
inline U32 LoadBigEndian(const U8* data) {
  return (static_cast<U32>(data[0]) << 24) |
         (static_cast<U32>(data[1]) << 16) |
         (static_cast<U32>(data[2]) << 8) | static_cast<U32>(data[3]);
}

// 把 32 位整数写成 4 字节大端。
//
// 参数：
//   value：待写值；
//   data：输出缓冲，至少 4 字节。
inline void StoreBigEndian(U32 value, U8* data) {
  data[0] = static_cast<U8>((value >> 24) & 0xffU);
  data[1] = static_cast<U8>((value >> 16) & 0xffU);
  data[2] = static_cast<U8>((value >> 8) & 0xffU);
  data[3] = static_cast<U8>(value & 0xffU);
}

// 单字节转小写十六进制。
//
// 参数：
//   value：0-15 的半字节。
// 返回：
//   '0'-'9' 或 'a'-'f'。
inline C8 HexDigit(U32 value) {
  static constexpr C8 kDigits[] = "0123456789abcdef";
  return kDigits[value & 0x0fU];
}

}  // namespace

Sha256::Sha256() {
  // FIPS 180-4 第 5.3.3 节初始哈希值。
  state_[0] = 0x6a09e667U;
  state_[1] = 0xbb67ae85U;
  state_[2] = 0x3c6ef372U;
  state_[3] = 0xa54ff53aU;
  state_[4] = 0x510e527fU;
  state_[5] = 0x9b05688cU;
  state_[6] = 0x1f83d9abU;
  state_[7] = 0x5be0cd19U;
}

void Sha256::Update(const void* data, Usize size) {
  if (finalized_ || size == 0) {
    return;
  }
  const U8* bytes = static_cast<const U8*>(data);
  total_bytes_ += static_cast<U64>(size);

  // 先补满上一次未压缩的尾部。
  if (buffer_size_ != 0) {
    const Usize needed = 64 - buffer_size_;
    const Usize take = size < needed ? size : needed;
    std::memcpy(buffer_ + buffer_size_, bytes, take);
    buffer_size_ += take;
    bytes += take;
    size -= take;
    if (buffer_size_ == 64) {
      Transform(buffer_);
      buffer_size_ = 0;
    }
  }

  // 直接处理完整的 64 字节块，避免逐块 memcpy。
  while (size >= 64) {
    Transform(bytes);
    bytes += 64;
    size -= 64;
  }

  // 剩余不足 64 字节留待 Final。
  if (size != 0) {
    std::memcpy(buffer_, bytes, size);
    buffer_size_ = size;
  }
}

void Sha256::Finish() {
  if (finalized_) {
    return;
  }
  // 追加 0x80，再补零到 56 mod 64，最后写 64 位大端比特长度。
  const U64 bit_length = total_bytes_ * 8U;
  buffer_[buffer_size_++] = 0x80U;
  if (buffer_size_ > 56) {
    std::memset(buffer_ + buffer_size_, 0, 64 - buffer_size_);
    Transform(buffer_);
    buffer_size_ = 0;
  }
  std::memset(buffer_ + buffer_size_, 0, 56 - buffer_size_);
  for (Usize index = 0; index < 8; ++index) {
    buffer_[56 + index] =
        static_cast<U8>((bit_length >> ((7 - index) * 8)) & 0xffU);
  }
  Transform(buffer_);
  buffer_size_ = 0;

  for (Usize index = 0; index < 8; ++index) {
    StoreBigEndian(state_[index], digest_ + index * 4);
  }
  finalized_ = true;
}

void Sha256::Final(U8 digest[32]) {
  Finish();
  std::memcpy(digest, digest_, 32);
}

std::string Sha256::HexDigest() {
  Finish();
  std::string hex;
  hex.resize(64);
  for (Usize index = 0; index < 32; ++index) {
    hex[index * 2] = HexDigit(static_cast<U32>(digest_[index] >> 4));
    hex[index * 2 + 1] = HexDigit(static_cast<U32>(digest_[index]));
  }
  return hex;
}

std::string Sha256::Hex(const void* data, Usize size) {
  Sha256 sha;
  sha.Update(data, size);
  return sha.HexDigest();
}

void Sha256::Transform(const U8* block) {
  // 前 16 个字直接读取，其余按 SHA-256 消息扩展递推。
  U32 schedule[64];
  for (Usize index = 0; index < 16; ++index) {
    schedule[index] = LoadBigEndian(block + index * 4);
  }
  for (Usize index = 16; index < 64; ++index) {
    const U32 s0 = RotateRight(schedule[index - 15], 7) ^
                   RotateRight(schedule[index - 15], 18) ^
                   (schedule[index - 15] >> 3);
    const U32 s1 = RotateRight(schedule[index - 2], 17) ^
                   RotateRight(schedule[index - 2], 19) ^
                   (schedule[index - 2] >> 10);
    schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
  }

  U32 a = state_[0];
  U32 b = state_[1];
  U32 c = state_[2];
  U32 d = state_[3];
  U32 e = state_[4];
  U32 f = state_[5];
  U32 g = state_[6];
  U32 h = state_[7];

  for (Usize index = 0; index < 64; ++index) {
    const U32 s1 =
        RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
    const U32 choose = (e & f) ^ ((~e) & g);
    const U32 temp1 =
        h + s1 + choose + kRoundConstants[index] + schedule[index];
    const U32 s0 =
        RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
    const U32 majority = (a & b) ^ (a & c) ^ (b & c);
    const U32 temp2 = s0 + majority;

    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

}  // namespace runtime
}  // namespace mobilenet
