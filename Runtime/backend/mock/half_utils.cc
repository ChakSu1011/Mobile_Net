// 模块：Mock 后端半精度转换实现
//
// 职责：
//   实现 FP32↔half、bfloat16→half 的位级转换。
//
// 边界：
//   不做批量/向量化；不处理设备内存；不抛异常。
//
// 依赖：
//   Runtime/backend/mock/half_utils.h、C++17 标准库。

#include "Runtime/backend/mock/half_utils.h"

#include <cstring>
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {
namespace {

// float32 的符号位掩码。
constexpr U32 kFloatSignMask = 0x80000000u;

// float32 的指数掩码。
constexpr U32 kFloatExponentMask = 0x7F800000u;

// float32 的尾数掩码。
constexpr U32 kFloatMantissaMask = 0x007FFFFFu;

// half 的符号位掩码。
constexpr U16 kHalfSignMask = 0x8000u;

// half 的指数掩码。
constexpr U16 kHalfExponentMask = 0x7C00u;

// half 的尾数掩码。
constexpr U16 kHalfMantissaMask = 0x03FFu;

// 把 float32 的位模式拆成 sign/exponent/mantissa。
struct FloatParts {
  U32 sign = 0;
  U32 exponent = 0;
  U32 mantissa = 0;
};

// 拆分 float32 位模式。
FloatParts SplitFloat(U32 bits) {
  FloatParts parts;
  parts.sign = bits & kFloatSignMask;
  parts.exponent = (bits & kFloatExponentMask) >> 23;
  parts.mantissa = bits & kFloatMantissaMask;
  return parts;
}

}  // namespace

FP16 FloatToHalf(FP32 value) {
  U32 bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const FloatParts parts = SplitFloat(bits);
  const U16 sign = static_cast<U16>(parts.sign >> 16);

  if (parts.exponent == 0xFFu) {
    // Inf 或 NaN：half 的指数全 1，NaN 保留尾数并强制非零。
    if (parts.mantissa == 0) {
      return static_cast<U16>(sign | kHalfExponentMask);
    }
    const U16 mantissa =
        static_cast<U16>(parts.mantissa >> 13);
    return static_cast<U16>(sign | kHalfExponentMask |
                                 (mantissa != 0 ? mantissa : 1u));
  }

  const I32 new_exponent =
      static_cast<I32>(parts.exponent) - 127 + 15;
  if (new_exponent >= 0x1F) {
    // 上溢：返回对应符号的 Inf。
    return static_cast<U16>(sign | kHalfExponentMask);
  }
  if (new_exponent <= 0) {
    // 次正规数或零。
    if (new_exponent < -10) {
      return sign;
    }
    const U32 mantissa = parts.mantissa | 0x00800000u;
    const U32 shift = static_cast<U32>(14 - new_exponent);
    U32 rounded = mantissa >> shift;
    const U32 remainder = mantissa & ((1u << shift) - 1u);
    const U32 halfway = 1u << (shift - 1u);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1u) != 0)) {
      ++rounded;
    }
    return static_cast<U16>(sign | rounded);
  }

  // 正规数。
  U16 half = static_cast<U16>(
      sign | (static_cast<U16>(new_exponent) << 10) |
      static_cast<U16>(parts.mantissa >> 13));
  const U32 remainder = parts.mantissa & 0x1FFFu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (half & 1u) != 0)) {
    ++half;
  }
  return half;
}

FP32 HalfToFloat(FP16 value) {
  const U32 sign = static_cast<U32>(value & kHalfSignMask) << 16;
  const U32 exponent = (value & kHalfExponentMask) >> 10;
  const U32 mantissa = value & kHalfMantissaMask;
  U32 bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // 次正规数：左移直到隐含位出现，再按正规数编码。
      U32 normalized = mantissa;
      I32 shift = -1;
      do {
        ++shift;
        normalized <<= 1;
      } while ((normalized & 0x400u) == 0);
      const U32 float_exponent =
          static_cast<U32>(127 - 15 - shift);
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

  FP32 result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

FP16 BFloat16ToHalf(BF16 value) {
  const U32 bits = static_cast<U32>(value) << 16;
  FP32 result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return FloatToHalf(result);
}

}  // namespace mock
}  // namespace mobilenet
