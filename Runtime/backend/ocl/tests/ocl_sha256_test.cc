// 模块：OCL SHA-256 工具测试
//
// 职责：
//   用已知向量验证 OclProgramCache 使用的 SHA-256 实现。
//
// 边界：
//   不需要 OpenCL 设备；只测试字符串哈希。
//
// 依赖：
//   Runtime/backend/ocl/ocl_sha256.h、C++17 标准库。

#include <iostream>
#include <string>

#include "Runtime/backend/ocl/ocl_sha256.h"

namespace {

using mobilenet::ocl::Sha256Hex;

// 输出一条断言结果。
void Expect(bool condition, const std::string& message, int* failures) {
  if (condition) {
    std::cout << "PASS " << message << "\n";
    return;
  }
  std::cout << "FAIL " << message << "\n";
  ++(*failures);
}

}  // namespace

int main() {
  int failures = 0;
  Expect(Sha256Hex("") ==
             "e3b0c44298fc1c149afbf4c8996fb924"
             "27ae41e4649b934ca495991b7852b855",
         "sha256 empty", &failures);
  Expect(Sha256Hex("abc") ==
             "ba7816bf8f01cfea414140de5dae2223"
             "b00361a396177a9cb410ff61f20015ad",
         "sha256 abc", &failures);
  Expect(Sha256Hex("The quick brown fox jumps over the lazy dog") ==
             "d7a8fbb307d7809469ca9abcb0082e4f"
             "8d5651e46d3cdb762d02d0bf37c9e592",
         "sha256 quick brown fox", &failures);
  Expect(Sha256Hex("abc") != Sha256Hex("abd"),
         "sha256 changes with input", &failures);
  if (failures == 0) {
    std::cout << "ALL TESTS PASSED\n";
    return 0;
  }
  std::cout << "FAILURES: " << failures << "\n";
  return 1;
}
