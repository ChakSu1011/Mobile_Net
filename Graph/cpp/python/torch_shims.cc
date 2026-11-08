// 模块：LibTorch 符号测试垫片
//
// 职责：
//   在 Zig/MinGW 测试构建中不链接 LibTorch，但 ATen 内联头文件会引用少量
//   c10/caffe2 外部符号。本文件提供元数据路径所需的最小实现，使 Graph IR
//   的 at::Tensor 元数据操作、Canonical JSON 与 Python 桥接可链接运行。
//
// 边界：
//   - 仅用于测试/工具链，不进入移动端运行时；
//   - 不实现算子、内存分配或自动微分；
//   - 未定义/越界场景直接抛异常或终止，不静默返回错误结果。
//
// 依赖：
//   ATen、c10 头文件与 CPython（仅用于 PyObject 引用计数）。

#include <ATen/ATen.h>
#include <Python.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/Exception.h>
#include <c10/util/typeid.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace c10 {
namespace detail {

// 替代 LibTorch 的检查失败实现；测试中转换为标准异常。
[[noreturn]] void torchCheckFail(const char* function, const char* file,
                                 uint32_t line, const std::string& message) {
  throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                           " in " + function + ": " + message);
}

[[noreturn]] void torchCheckFail(const char* function, const char* file,
                                 uint32_t line, const char* message) {
  torchCheckFail(function, file, line,
                 std::string(message == nullptr ? "" : message));
}

[[noreturn]] void torchInternalAssertFail(const char* function,
                                          const char* file, uint32_t line,
                                          const char* condition,
                                          const char* message) {
  torchCheckFail(function, file, line,
                 std::string(condition == nullptr ? "" : condition) + " " +
                     (message == nullptr ? "" : message));
}

[[noreturn]] void torchInternalAssertFail(const char* function,
                                          const char* file, uint32_t line,
                                          const char* condition,
                                          const std::string& message) {
  torchCheckFail(
      function, file, line,
      std::string(condition == nullptr ? "" : condition) + " " + message);
}

// 仅支持 int64_t 维度；越界直接抛异常。
template <>
int64_t maybe_wrap_dim_slow<int64_t>(int64_t dim, int64_t dim_post_expr,
                                     bool wrap_scalar) {
  if (wrap_scalar && dim_post_expr == 1 && (dim == 0 || dim == -1)) {
    return 0;
  }
  if (dim < 0) {
    dim += dim_post_expr;
  }
  if (dim < 0 || dim >= dim_post_expr) {
    throw std::out_of_range("dimension out of range");
  }
  return dim;
}

}  // namespace detail

// 测试中张量均不需要梯度；Graph 只做元数据转换。
bool TensorImpl::requires_grad() const { return false; }

// 维护 Python 包装对象引用计数，避免 THPVariable_Wrap 返回的包装被提前释放。
void TensorImpl::incref_pyobject() const noexcept {
  PyObject* object = pyobj_slot()->load_pyobj();
  Py_XINCREF(object);
}

void TensorImpl::decref_pyobject() const noexcept {
  PyObject* object = pyobj_slot()->load_pyobj();
  Py_XDECREF(object);
}

bool TensorImpl::try_incref_pyobject() const noexcept {
  PyObject* object = pyobj_slot()->load_pyobj();
  if (object == nullptr) {
    return false;
  }
  Py_XINCREF(object);
  return true;
}

// 返回一个未初始化的 UndefinedTensorImpl 哨兵；仅用于空 intrusive_ptr。
UndefinedTensorImpl& UndefinedTensorImpl::getInstance() {
  alignas(UndefinedTensorImpl) static unsigned char
      storage[sizeof(UndefinedTensorImpl)];
  return *reinterpret_cast<UndefinedTensorImpl*>(storage);
}

// Storage 数据访问错误路径。
[[noreturn]] void StorageImpl::throw_data_ptr_access_error() const {
  throw std::runtime_error("storage data pointer is not accessible");
}

}  // namespace c10

namespace caffe2 {

// 标量类型元数据表；本测试只使用 ScalarType 快速路径。
detail::TypeMetaData* TypeMeta::typeMetaDatas() {
  static detail::TypeMetaData instances[256];
  return instances;
}

// 非标量 TypeMeta 在测试中不应出现。
[[noreturn]] void TypeMeta::error_unsupported_typemeta(caffe2::TypeMeta) {
  std::abort();
}

}  // namespace caffe2
