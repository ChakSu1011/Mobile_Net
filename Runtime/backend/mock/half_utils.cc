// 模块：Mock 后端半精度转换实现
//
// 职责：
//   实现 float↔half、bfloat16→half 的位级转换。
//
// 边界：
//   不做批量/向量化；不处理设备内存；不抛异常。
//
// 依赖：
//   Runtime/backend/mock/half_utils.h、C++17 标准库。

#include "Runtime/backend/mock/half_utils.h"

#include <cstring>

namespace mobilenet {
namespace mock {
namespace {

// float32 的符号位掩码。
constexpr uint32_t kFloatSignMask = 0x80000000u;

// float32 的指数掩码。
constexpr uint32_t kFloatExponentMask = 0x7F800000u;

// float32 的尾数掩码。
constexpr uint32_t kFloatMantissaMask = 0x007FFFFFu;

// half 的符号位掩码。
constexpr uint16_t kHalfSignMask = 0x8000u;

// half 的指数掩码。
constexpr uint16_t kHalfExponentMask = 0x7C00u;

// half 的尾数掩码。
constexpr uint16_t kHalfMantissaMask = 0x03FFu;

// 把 float32 的位模式拆成 sign/exponent/mantissa。
struct FloatParts {
  uint32_t sign = 0;
  uint32_t exponent = 0;
  uint32_t mantissa = 0;
};

// 拆分 float32 位模式。
FloatParts SplitFloat(uint32_t bits) {
  FloatParts parts;
  parts.sign = bits & kFloatSignMask;
  parts.exponent = (bits & kFloatExponentMask) >> 23;
  parts.mantissa = bits & kFloatMantissaMask;
  return parts;
}

}  // namespace

uint16_t FloatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const FloatParts parts = SplitFloat(bits);
  const uint16_t sign = static_cast<uint16_t>(parts.sign >> 16);

  if (parts.exponent == 0xFFu) {
    // Inf 或 NaN：half 的指数全 1，NaN 保留尾数并强制非零。
    if (parts.mantissa == 0) {
      return static_cast<uint16_t>(sign | kHalfExponentMask);
    }
    const uint16_t mantissa =
        static_cast<uint16_t>(parts.mantissa >> 13);
    return static_cast<uint16_t>(sign | kHalfExponentMask |
                                 (mantissa != 0 ? mantissa : 1u));
  }

  const int32_t new_exponent =
      static_cast<int32_t>(parts.exponent) - 127 + 15;
  if (new_exponent >= 0x1F) {
    // 上溢：返回对应符号的 Inf。
    return static_cast<uint16_t>(sign | kHalfExponentMask);
  }
  if (new_exponent <= 0) {
    // 次正规数或零。
    if (new_exponent < -10) {
      return sign;
    }
    const uint32_t mantissa = parts.mantissa | 0x00800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - new_exponent);
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1u) != 0)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }

  // 正规数。
  uint16_t half = static_cast<uint16_t>(
      sign | (static_cast<uint16_t>(new_exponent) << 10) |
      static_cast<uint16_t>(parts.mantissa >> 13));
  const uint32_t remainder = parts.mantissa & 0x1FFFu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (half & 1u) != 0)) {
    ++half;
  }
  return half;
}

float HalfToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & kHalfSignMask) << 16;
  const uint32_t exponent = (value & kHalfExponentMask) >> 10;
  const uint32_t mantissa = value & kHalfMantissaMask;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // 次正规数：左移直到隐含位出现，再按正规数编码。
      uint32_t normalized = mantissa;
      int32_t shift = -1;
      do {
        ++shift;
        normalized <<= 1;
      } while ((normalized & 0x400u) == 0);
      const uint32_t float_exponent =
          static_cast<uint32_t>(127 - 15 - shift);
      bits = sign | (float_exponent << 23) |
             ((normalized & 0x3FFu) << 13);
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign |
           ((exponent - 15u + 127u) << 23) |
           (mantissa << 13);
  }

  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t BFloat16ToHalf(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return FloatToHalf(result);
}

}  // namespace mock
}  // namespace mobilenet
