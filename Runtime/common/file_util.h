// 模块：Runtime 文件读取工具
//
// 职责：
//   提供文件大小查询、区间读取与流式分块读取，供 Loader 读取 model.mgf
//   头部/manifest 与 1 GB 级 weights.bin；同时提供流式 SHA-256 计算。
//
// 边界：
//   - 只读文件，不创建/修改/删除；
//   - 不做内存映射；大文件按调用方给定的块大小顺序读取；
//   - 路径按平台本地编码解释；本机验证路径为 ASCII，非 ASCII 路径需
//     调用方自行保证编码；
//   - 不抛出异常，所有错误通过 Status 返回。
//
// 依赖：
//   Runtime/common/status.h、Runtime/common/sha256.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_FILE_UTIL_H_
#define MOBILENET_RUNTIME_COMMON_FILE_UTIL_H_

#include <fstream>
#include <string>
#include <vector>

#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {

// 查询文件字节数。
//
// 参数：
//   path：文件路径，非空；
//   size：输出参数，非空；成功时写入文件字节数。
// 返回：
//   成功返回 kOk；文件不存在/不可读返回 kIoError；路径为空或 size 为空
//   返回 kInvalidArgument。
Status GetFileSize(const std::string& path, U64* size);

// 一次性读取文件区间。
//
// 功能：
//   面向 manifest 等小数据；内部按 8 MB 分块循环读取，避免依赖单次
//   read 返回完整长度。
//
// 参数：
//   path：文件路径；
//   offset：起始偏移（字节）；
//   length：读取字节数；
//   out：输出参数，非空；成功时 resize 为 length 并写入数据。
// 返回：
//   成功返回 kOk；越界返回 kInvalidArgument；文件/读取失败返回 kIoError。
Status ReadFileRange(const std::string& path, U64 offset, U64 length,
                     std::vector<U8>* out);

// 流式文件读取器。
//
// 功能：
//   持有打开的二进制文件流，支持随机区间读取与顺序分块读取；支持超过
//   4 GB 的文件。
//
// 边界：
//   - 单线程使用；同一实例不允许并发读；
//   - Open 成功后 Size 固定；文件被外部截断时读取会返回 kIoError；
//   - 不缓存文件内容，调用方负责块大小与内存上限。
class FileReader {
 public:
  FileReader();

  ~FileReader();

  FileReader(const FileReader&) = delete;
  FileReader& operator=(const FileReader&) = delete;
  FileReader(FileReader&&) = delete;
  FileReader& operator=(FileReader&&) = delete;

  // 打开文件。
  //
  // 参数：
  //   path：文件路径，非空。
  // 返回：
  //   成功返回 kOk；路径为空或打开失败返回 kInvalidArgument/kIoError。
  Status Open(const std::string& path);

  // 返回文件是否已打开。
  Bool IsOpen() const;

  // 返回文件字节数。
  //
  // 参数：
  //   size：输出参数，非空。
  // 返回：
  //   成功返回 kOk；未打开返回 kStateError。
  Status Size(U64* size) const;

  // 读取指定区间到 out。
  //
  // 参数：
  //   offset：起始偏移；
  //   length：字节数；
  //   out：输出参数，成功时 resize 为 length。
  // 返回：
  //   成功返回 kOk；越界返回 kInvalidArgument；短读/IO 失败返回 kIoError。
  Status ReadAt(U64 offset, U64 length, std::vector<U8>* out);

  // 从当前顺序位置读取一块。
  //
  // 参数：
  //   max_length：最多读取字节数；
  //   out：输出参数，成功时 resize 为实际读取字节数；
  //   read_bytes：输出参数，实际读取字节数；0 表示已到文件尾。
  // 返回：
  //   成功返回 kOk；未打开返回 kStateError；读取失败返回 kIoError。
  Status ReadNext(U64 max_length, std::vector<U8>* out, Usize* read_bytes);

  // 关闭文件；幂等。
  void Close();

 private:
  // 打开的文件流。
  std::ifstream stream_;

  // 文件字节数。
  U64 size_ = 0;

  // 顺序读取位置。
  U64 position_ = 0;
};

// 流式计算文件 SHA-256。
//
// 参数：
//   path：文件路径；
//   chunk_bytes：单次读取块大小；0 表示默认 4 MB；
//   hex：输出参数，非空；成功时写入 64 字符小写十六进制摘要。
// 返回：
//   成功返回 kOk；文件/读取失败返回 kIoError；参数非法返回
//   kInvalidArgument。
Status HashFileSha256(const std::string& path, U64 chunk_bytes,
                      std::string* hex);

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_FILE_UTIL_H_
