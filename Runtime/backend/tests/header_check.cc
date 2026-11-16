// 模块：Backend 公共头、注册表与 ABI 检查
//
// 职责：
//   1. 包含 Backend/Kernel 全部公共头，验证声明自洽、可编译；
//   2. 验证 Precision↔DataType 映射；
//   3. 验证 BackendRegistry 的注册、创建、重复/非法注册与查询；
//   4. 验证 SetStepContext 的类型校验与空值处理；
//   5. 验证 KernelContext/TensorView 默认值。
//
// 边界：
//   - 不执行真实设备操作，不创建 OCL/mock 后端；
//   - 不验证 Kernel 数学实现与 Buffer 内容；
//   - 只用于 P0 接口/注册表回归；
//   - 失败返回非 0，成功打印 ALL TESTS PASSED。
//
// 依赖：
//   Runtime/backend 公共头、Graph 静态库、C++17 标准库。

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Runtime/backend/backend.h"
#include "Runtime/backend/backend_registry.h"
#include "Runtime/backend/exec_op_env.h"
#include "Runtime/backend/kernel.h"
#include "Runtime/backend/kernel_context.h"
#include "Runtime/backend/kernel_types.h"

namespace {

using mobilenet::Backend;
using mobilenet::BackendCapability;
using mobilenet::BackendConfig;
using mobilenet::BackendFactory;
using mobilenet::BackendRegistry;
using mobilenet::Buffer;
using mobilenet::BufferDesc;
using mobilenet::DataType;
using mobilenet::DataTypeFromPrecision;
using mobilenet::ErrorCode;
using mobilenet::ExecOpEnv;
using mobilenet::IsSupportedDataType;
using mobilenet::Kernel;
using mobilenet::KernelDesc;
using mobilenet::KvLayout;
using mobilenet::ParamDesc;
using mobilenet::PrecisionFromDataType;
using mobilenet::SetStepContext;
using mobilenet::Status;
using mobilenet::StepContext;
using mobilenet::Stream;
using mobilenet::runtime_graph::KernelContext;
using mobilenet::runtime_graph::OpEnv;
using mobilenet::runtime_graph::OpType;
using mobilenet::runtime_graph::Precision;

// 假设备缓冲：用自身地址作为不透明设备句柄。
class FakeBuffer final : public Buffer {
 public:
  explicit FakeBuffer(uint64_t bytes) : bytes_(bytes) {}

  void* DevicePtr() const override {
    return const_cast<FakeBuffer*>(this);
  }

  uint64_t Bytes() const override { return bytes_; }

 private:
  uint64_t bytes_ = 0;
};

// 假执行流：无状态。
class FakeStream final : public Stream {};

// 假 Kernel：只验证 ABI 可调用。
class FakeKernel final : public Kernel {
 public:
  Status Execute(const KernelContext&) override { return Status::Ok(); }
};

// 假后端：实现全部纯虚接口，用于注册表与 ABI 测试。
class FakeBackend final : public Backend {
 public:
  FakeBackend() {
    capability_.supported_dtypes = {DataType::kFloat16};
    capability_.supported_ops = {OpType::kAdd, OpType::kSample};
    capability_.h2d = true;
    capability_.d2h = true;
    capability_.max_buffer_bytes = 1ull << 30;
    capability_.min_alignment = 1;
    capability_.attention_block_size = 16;
    capability_.kv_layout = KvLayout::kNhd;
  }

  Status Init(const BackendConfig&) override { return Status::Ok(); }

  Status Deinit() override { return Status::Ok(); }

  Status Capability(BackendCapability* capability) const override {
    if (capability == nullptr) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          "FakeBackend::Capability: capability must not be null");
    }
    *capability = capability_;
    return Status::Ok();
  }

  Status CreateBuffer(const BufferDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override {
    if (buffer == nullptr || desc.bytes == 0) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          "FakeBackend::CreateBuffer: invalid argument");
    }
    *buffer = std::make_unique<FakeBuffer>(desc.bytes);
    return Status::Ok();
  }

  Status DestroyBuffer(std::unique_ptr<Buffer> buffer) override {
    buffer.reset();
    return Status::Ok();
  }

  Status ConvertParam(const ParamDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override {
    if (buffer == nullptr || desc.host_ptr == nullptr || desc.bytes == 0) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          "FakeBackend::ConvertParam: invalid argument");
    }
    *buffer = std::make_unique<FakeBuffer>(desc.bytes);
    return Status::Ok();
  }

  Status Upload(void* device_ptr, const void* host_ptr, uint64_t bytes,
                Stream*) override {
    if (device_ptr == nullptr || host_ptr == nullptr || bytes == 0) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          "FakeBackend::Upload: invalid argument");
    }
    return Status::Ok();
  }

  Status Download(void* host_ptr, const void* device_ptr, uint64_t bytes,
                  Stream*) override {
    if (host_ptr == nullptr || device_ptr == nullptr || bytes == 0) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          "FakeBackend::Download: invalid argument");
    }
    return Status::Ok();
  }

  Status CreateKernel(const KernelDesc& desc,
                      std::unique_ptr<Kernel>* kernel) override {
    if (kernel == nullptr) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          "FakeBackend::CreateKernel: kernel output must not be null");
    }
    if (desc.kernel_key.empty()) {
      return mobilenet::MakeStatus(
          ErrorCode::kUnsupportedOp,
          "FakeBackend::CreateKernel: empty kernel_key");
    }
    *kernel = std::make_unique<FakeKernel>();
    return Status::Ok();
  }

  Status DestroyKernel(std::unique_ptr<Kernel> kernel) override {
    kernel.reset();
    return Status::Ok();
  }

  Status Synchronize(Stream*) override { return Status::Ok(); }

 private:
  BackendCapability capability_;
};

// 假后端工厂。
Status CreateFakeBackend(const BackendConfig&,
                         std::unique_ptr<Backend>* backend) {
  if (backend == nullptr) {
    return mobilenet::MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateFakeBackend: backend output must not be null");
  }
  *backend = std::make_unique<FakeBackend>();
  return Status::Ok();
}

// 测试用执行型 OpEnv。
class TestExecOpEnv final : public ExecOpEnv {
 public:
  explicit TestExecOpEnv(mobilenet::runtime_graph::Op* op)
      : ExecOpEnv(op) {}

  Status Forward() override { return Status::Ok(); }

  bool NeedTempData() const override { return false; }
};

// 测试用普通 OpEnv：不是 ExecOpEnv。
class TestPlainOpEnv final : public OpEnv {
 public:
  explicit TestPlainOpEnv(mobilenet::runtime_graph::Op* op) : OpEnv(op) {}

  Status Forward() override { return Status::Ok(); }

  bool NeedTempData() const override { return false; }
};

// 输出一条断言结果。
void Expect(bool condition, const std::string& message, int* failures) {
  if (condition) {
    std::cout << "PASS " << message << "\n";
    return;
  }
  std::cout << "FAIL " << message << "\n";
  ++(*failures);
}

// 验证 Precision↔DataType 映射与支持性判断。
void CheckDataTypeMapping(int* failures) {
  Expect(DataTypeFromPrecision(Precision::kFloat16) == DataType::kFloat16,
         "precision float16 -> DataType float16", failures);
  Expect(PrecisionFromDataType(DataType::kFloat16) == Precision::kFloat16,
         "DataType float16 -> precision float16", failures);
  Expect(DataTypeFromPrecision(Precision::kFloat8E4M3) == DataType::kUnknown,
         "precision float8e4m3 has no DataType", failures);
  Expect(DataTypeFromPrecision(Precision::kInt16) == DataType::kUnknown,
         "precision int16 has no DataType", failures);
  Expect(!IsSupportedDataType(DataType::kUnknown),
         "DataType unknown is unsupported", failures);
  Expect(IsSupportedDataType(DataType::kFloat16),
         "DataType float16 is supported", failures);

  const Precision kPrecisions[] = {
      Precision::kFloat64, Precision::kFloat32, Precision::kFloat16,
      Precision::kBFloat16, Precision::kInt64,   Precision::kInt32,
      Precision::kInt8,    Precision::kUInt8,    Precision::kBool,
  };
  for (const Precision precision : kPrecisions) {
    const DataType dtype = DataTypeFromPrecision(precision);
    Expect(dtype != DataType::kUnknown,
           "supported precision maps to DataType", failures);
    Expect(PrecisionFromDataType(dtype) == precision,
           "supported precision round-trips", failures);
  }
}

// 验证注册表的注册、重复/非法注册、创建与枚举。
void CheckBackendRegistry(int* failures) {
  BackendRegistry& registry = BackendRegistry::Instance();
  Expect(registry.Register("fake", CreateFakeBackend).ok(),
         "register fake backend", failures);
  Expect(!registry.Register("fake", CreateFakeBackend).ok(),
         "duplicate backend is rejected", failures);
  Expect(!registry.Register("", CreateFakeBackend).ok(),
         "empty backend name is rejected", failures);
  Expect(!registry.Register("fake_null", BackendFactory{}).ok(),
         "empty backend factory is rejected", failures);

  const std::vector<std::string> names = registry.Names();
  Expect(std::find(names.begin(), names.end(), "fake") != names.end(),
         "registry names contain fake", failures);

  std::unique_ptr<Backend> missing;
  Expect(registry.Create("missing", BackendConfig{}, &missing).code() ==
             ErrorCode::kUnsupportedBackend,
         "unknown backend returns kUnsupportedBackend", failures);
  Expect(registry.Create("fake", BackendConfig{}, nullptr).code() ==
             ErrorCode::kInvalidArgument,
         "null backend output is rejected", failures);

  std::unique_ptr<Backend> backend;
  const Status created = registry.Create("fake", BackendConfig{}, &backend);
  Expect(created.ok() && backend != nullptr,
         "create fake backend", failures);
  if (backend != nullptr) {
    BackendCapability capability;
    Expect(backend->Capability(&capability).ok(),
           "fake backend capability", failures);
    Expect(capability.attention_block_size == 16,
           "capability attention block size", failures);
    Expect(!capability.supported_ops.empty(),
           "capability supported ops non-empty", failures);
  }

  // 验证自注册宏：静态注册器应在 main 之前完成。
  std::unique_ptr<Backend> macro_backend;
  Expect(registry.Create("fake_macro", BackendConfig{}, &macro_backend).ok(),
         "backend registered by macro is creatable", failures);
}

// 验证 SetStepContext 的类型校验与空值处理。
void CheckStepContext(int* failures) {
  StepContext context;
  TestExecOpEnv exec_env(nullptr);
  Expect(SetStepContext(&exec_env, &context).ok(),
         "set step context on ExecOpEnv", failures);
  Expect(SetStepContext(&exec_env, nullptr).ok(),
         "clear step context on ExecOpEnv", failures);

  TestPlainOpEnv plain_env(nullptr);
  Expect(SetStepContext(&plain_env, &context).code() == ErrorCode::kInternal,
         "non-ExecOpEnv is rejected with kInternal", failures);
  Expect(SetStepContext(nullptr, &context).code() ==
             ErrorCode::kInvalidArgument,
         "null OpEnv is rejected with kInvalidArgument", failures);
}

// 验证 KernelContext/TensorView 默认值。
void CheckKernelContextDefaults(int* failures) {
  KernelContext context;
  Expect(context.inputs == nullptr && context.input_count == 0,
         "KernelContext inputs default empty", failures);
  Expect(context.outputs == nullptr && context.output_count == 0,
         "KernelContext outputs default empty", failures);
  Expect(context.attrs == nullptr && context.attention == nullptr,
         "KernelContext optional pointers default null", failures);
  Expect(context.batch == 1 && context.token_count == 0 &&
             context.seq_len == 0,
         "KernelContext step fields default", failures);
  Expect(context.workspace.data == nullptr && context.workspace.bytes == 0,
         "KernelContext workspace default empty", failures);
  Expect(context.workspace.layout ==
             mobilenet::runtime_graph::Layout::kRowMajor,
         "TensorView layout defaults row major", failures);
}

}  // namespace

// 宏自注册：验证静态注册路径；名称固定，避免与测试内注册冲突。
MOBILENET_REGISTER_BACKEND("fake_macro", CreateFakeBackend);

int main() {
  int failures = 0;

  CheckDataTypeMapping(&failures);
  CheckBackendRegistry(&failures);
  CheckStepContext(&failures);
  CheckKernelContextDefaults(&failures);

  if (failures == 0) {
    std::cout << "ALL TESTS PASSED\n";
    return 0;
  }
  std::cout << "FAILURES: " << failures << "\n";
  return 1;
}
