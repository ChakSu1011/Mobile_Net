// 模块：SHA-256 工具实现
//
// 职责：
//   实现 FIPS 180-4 SHA-256 的字符串哈希与十六进制输出。
//
// 边界：
//   只处理内存字符串；不读写文件；无全局状态。
//
// 依赖：
//   Runtime/backend/ocl/ocl_sha256.h、C++17 标准库。

#include "Runtime/backend/ocl/ocl_sha256.h"

#include <array>
#include <cstdint>
#include <vector>

#include "Utils/cpp/types.h"

namespace mobilenet {
namespace ocl {
namespace {

// 循环左移。
U32 RotateRight(U32 value, U32 bits) {
  return (value >> bits) | (value << (32u - bits));
}

// SHA-256 轮常量。
constexpr std::array<U32, 64> kRoundConstants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

// 初始哈希值。
constexpr std::array<U32, 8> kInitialHash = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

// 处理一个 64 字节块。
void ProcessBlock(const U8* block, std::array<U32, 8>* hash) {
  std::array<U32, 64> schedule{};
  for (U32 i = 0; i < 16; ++i) {
    schedule[i] = (static_cast<U32>(block[i * 4]) << 24) |
                  (static_cast<U32>(block[i * 4 + 1]) << 16) |
                  (static_cast<U32>(block[i * 4 + 2]) << 8) |
                  static_cast<U32>(block[i * 4 + 3]);
  }
  for (U32 i = 16; i < 64; ++i) {
    const U32 s0 = RotateRight(schedule[i - 15], 7) ^
                   RotateRight(schedule[i - 15], 18) ^
                   (schedule[i - 15] >> 3);
    const U32 s1 = RotateRight(schedule[i - 2], 17) ^
                   RotateRight(schedule[i - 2], 19) ^
                   (schedule[i - 2] >> 10);
    schedule[i] = schedule[i - 16] + s0 + schedule[i - 7] + s1;
  }

  U32 a = (*hash)[0];
  U32 b = (*hash)[1];
  U32 c = (*hash)[2];
  U32 d = (*hash)[3];
  U32 e = (*hash)[4];
  U32 f = (*hash)[5];
  U32 g = (*hash)[6];
  U32 h = (*hash)[7];

  for (U32 i = 0; i < 64; ++i) {
    const U32 s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
    const U32 ch = (e & f) ^ ((~e) & g);
    const U32 temp1 = h + s1 + ch + kRoundConstants[i] + schedule[i];
    const U32 s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
    const U32 maj = (a & b) ^ (a & c) ^ (b & c);
    const U32 temp2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  (*hash)[0] += a;
  (*hash)[1] += b;
  (*hash)[2] += c;
  (*hash)[3] += d;
  (*hash)[4] += e;
  (*hash)[5] += f;
  (*hash)[6] += g;
  (*hash)[7] += h;
}

// 输出十六进制字符。
C8 HexDigit(U32 value) {
  return static_cast<C8>(value < 10 ? ('0' + value)
                                    : ('a' + (value - 10)));
}

}  // namespace

std::string Sha256Hex(const std::string& input) {
  std::vector<U8> bytes(input.begin(), input.end());
  const U64 bit_length = static_cast<U64>(bytes.size()) * 8u;
  bytes.push_back(0x80u);
  while (bytes.size() % 64u != 56u) {
    bytes.push_back(0u);
  }
  for (I32 shift = 56; shift >= 0; shift -= 8) {
    bytes.push_back(static_cast<U8>((bit_length >> shift) & 0xFFu));
  }

  std::array<U32, 8> hash = kInitialHash;
  for (Usize offset = 0; offset < bytes.size(); offset += 64) {
    ProcessBlock(bytes.data() + offset, &hash);
  }

  std::string output;
  output.reserve(64);
  for (const U32 word : hash) {
    for (I32 shift = 28; shift >= 0; shift -= 4) {
      output.push_back(HexDigit((word >> shift) & 0x0Fu));
    }
  }
  return output;
}

}  // namespace ocl
}  // namespace mobilenet
