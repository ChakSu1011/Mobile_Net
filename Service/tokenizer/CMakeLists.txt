# 模块：Service/tokenizer 构建入口
#
# 职责：
#   构建内置 utf8proc、Tokenizer 静态库与单元测试。
#
# 边界：
#   - 复用 Runtime 公共库（Status/JSON/FileUtil）；
#   - 不构建 Engine 可执行文件；
#   - 不下载依赖。

add_library(
  mobilenet_utf8proc
  STATIC
  third_party/utf8proc/utf8proc.c
)
set_target_properties(
  mobilenet_utf8proc
  PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)
target_include_directories(
  mobilenet_utf8proc
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/utf8proc"
)
# Windows 下静态库必须关闭 dllimport/export 宏，否则定义符号报错。
target_compile_definitions(mobilenet_utf8proc PUBLIC UTF8PROC_STATIC)

add_library(
  mobilenet_tokenizer
  STATIC
  bpe.cc
  byte_level.cc
  chat_template.cc
  pretokenizer.cc
  tokenizer.cc
  tokenizer_config.cc
  unicode_util.cc
)

target_include_directories(
  mobilenet_tokenizer
  PUBLIC
    "${MOBILENET_ROOT}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(
  mobilenet_tokenizer
  PUBLIC mobilenet_runtime mobilenet_utf8proc
)

if(MSVC)
  target_compile_options(mobilenet_tokenizer PRIVATE /W4)
  target_compile_options(mobilenet_utf8proc PRIVATE /W3)
else()
  target_compile_options(mobilenet_tokenizer PRIVATE -Wall -Wextra)
  target_compile_options(mobilenet_utf8proc PRIVATE -Wall -Wextra)
endif()

# 单元测试：需要本地 Qwen3 Tokenizer 资产；资产缺失时测试返回 77（跳过）。
add_executable(tokenizer_test tests/tokenizer_test.cc)
target_link_libraries(tokenizer_test PRIVATE mobilenet_tokenizer)
if(MSVC)
  target_compile_options(tokenizer_test PRIVATE /W4)
else()
  target_compile_options(tokenizer_test PRIVATE -Wall -Wextra)
endif()
add_test(
  NAME tokenizer_test
  COMMAND tokenizer_test
    "${MOBILENET_ROOT}/Models/Qwen3"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/golden.json"
)
set_tests_properties(tokenizer_test PROPERTIES SKIP_RETURN_CODE 77)
