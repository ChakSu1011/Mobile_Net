# 模块：Service/session 构建入口
#
# 职责：
#   构建 Session 静态库、服务可执行文件与集成测试。
#
# 边界：
#   - 复用 Tokenizer 与 Runtime；
#   - 不构建模型包、不下载依赖；
#   - 测试使用 FakeEngine，避免加载 1.2 GB 模型。

add_library(
  mobilenet_session
  STATIC
  chat_service.cc
  engine_client.cc
  http_router.cc
  http_server.cc
  json_writer.cc
  log.cc
  openai_api.cc
  openapi_spec.cc
  pending_registry.cc
  server.cc
  server_config.cc
  session_manager.cc
)

target_include_directories(
  mobilenet_session
  PUBLIC
    "${MOBILENET_ROOT}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(
  mobilenet_session
  PUBLIC mobilenet_tokenizer mobilenet_runtime
)

if(MSVC)
  target_compile_options(mobilenet_session PRIVATE /W4)
else()
  target_compile_options(mobilenet_session PRIVATE -Wall -Wextra)
endif()

if(WIN32)
  target_link_libraries(mobilenet_session PUBLIC ws2_32)
endif()

add_executable(mobilenet_service main.cc)
target_link_libraries(
  mobilenet_service PRIVATE
  mobilenet_session
  mobilenet_tokenizer
  mobilenet_runtime
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,runtime_backend>"
  graph
  Threads::Threads
)
if(WIN32)
  target_link_libraries(mobilenet_service PRIVATE ws2_32)
endif()

add_executable(session_test tests/session_test.cc)
target_link_libraries(
  session_test PRIVATE
  mobilenet_session
  mobilenet_tokenizer
  mobilenet_runtime
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,runtime_backend>"
  graph
  Threads::Threads
)
if(WIN32)
  target_link_libraries(session_test PRIVATE ws2_32)
endif()

if(MSVC)
  target_compile_options(mobilenet_service PRIVATE /W4)
  target_compile_options(session_test PRIVATE /W4)
else()
  target_compile_options(mobilenet_service PRIVATE -Wall -Wextra)
  target_compile_options(session_test PRIVATE -Wall -Wextra)
endif()

add_test(
  NAME session_test
  COMMAND session_test "${MOBILENET_ROOT}/Models/Qwen3"
)
set_tests_properties(session_test PROPERTIES SKIP_RETURN_CODE 77)
