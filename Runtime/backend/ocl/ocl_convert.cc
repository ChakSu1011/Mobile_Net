// 模块：OCL 权重精度转换实现
//
// 职责：
//   实现 float↔half、bfloat16→half 的位级转换（RN-even）。
//
// 边界：
//   只做标量转换；不处理设备内存。
//
// 依赖：
//   Runtime/backend/ocl/ocl_convert.h、C++17 标准库。

#include "Runtime/backend/ocl/ocl_convert.h"

#include <cstring>

namespace mobilenet {
namespace ocl {
namespace {

constexpr U32 kFloatSignMask = 0x80000000u;
constexpr U32 kFloatExponentMask = 0x7F800000u;
constexpr U32 kFloatMantissaMask = 0x007FFFFFu;
constexpr U16 kHalfSignMask = 0x8000u;
constexpr U16 kHalfExponentMask = 0x7C00u;
constexpr U16 kHalfMantissaMask = 0x03FFu;

}  // namespace

U16 FloatToHalf(FP32 value) {
  U32 bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const U16 sign = static_cast<U16>((bits & kFloatSignMask) >> 16);
  const U32 exponent = (bits & kFloatExponentMask) >> 23;
  const U32 mantissa = bits & kFloatMantissaMask;

  if (exponent == 0xFFu) {
    if (mantissa == 0) {
      return static_cast<U16>(sign | kHalfExponentMask);
    }
    const U16 half_mantissa = static_cast<U16>(mantissa >> 13);
    return static_cast<U16>(sign | kHalfExponentMask |
                            (half_mantissa != 0 ? half_mantissa : 1u));
  }

  const I32 new_exponent = static_cast<I32>(exponent) - 127 + 15;
  if (new_exponent >= 0x1F) {
    return static_cast<U16>(sign | kHalfExponentMask);
  }
  if (new_exponent <= 0) {
    if (new_exponent < -10) {
      return sign;
    }
    const U32 full_mantissa = mantissa | 0x00800000u;
    const U32 shift = static_cast<U32>(14 - new_exponent);
    U32 rounded = full_mantissa >> shift;
    const U32 remainder = full_mantissa & ((1u << shift) - 1u);
    const U32 halfway = 1u << (shift - 1u);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1u) != 0)) {
      ++rounded;
    }
    return static_cast<U16>(sign | rounded);
  }

  U16 half = static_cast<U16>(
      sign | (static_cast<U16>(new_exponent) << 10) |
      static_cast<U16>(mantissa >> 13));
  const U32 remainder = mantissa & 0x1FFFu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (half & 1u) != 0)) {
    ++half;
  }
  return half;
}

FP32 HalfToFloat(U16 value) {
  const U32 sign = static_cast<U32>(value & kHalfSignMask) << 16;
  const U32 exponent = (value & kHalfExponentMask) >> 10;
  const U32 mantissa = value & kHalfMantissaMask;
  U32 bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      U32 normalized = mantissa;
      I32 shift = -1;
      do {
        ++shift;
        normalized <<= 1;
      } while ((normalized & 0x400u) == 0);
      bits = sign |
             (static_cast<U32>(127 - 15 - shift) << 23) |
             ((normalized & 0x3FFu) << 13);
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15u + 127u) << 23) | (mantissa << 13);
  }
  FP32 result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

U16 BFloat16ToHalf(U16 value) {
  const U32 bits = static_cast<U32>(value) << 16;
  FP32 result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return FloatToHalf(result);
}

}  // namespace ocl
}  // namespace mobilenet
