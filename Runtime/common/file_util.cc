// 模块：Runtime 文件读取工具实现
//
// 职责：
//   实现文件大小查询、区间读取、顺序分块读取与流式 SHA-256。
//
// 边界：
//   - 只读；不修改文件；
//   - 区间与顺序读取都按 8 MB 内部块循环，兼容大文件与短读；
//   - 失败时返回带路径的错误消息，不抛出异常。
//
// 依赖：
//   Runtime/common/file_util.h、Runtime/common/sha256.h、C++17 标准库。

#include "Runtime/common/file_util.h"

#include <algorithm>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include "Runtime/common/sha256.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// 单次读取的默认/最大块大小（8 MB）。
constexpr U64 kReadChunkBytes = 8ULL * 1024ULL * 1024ULL;

// 构造带路径的 IO 错误。
//
// 参数：
//   action：动作描述，例如 "open"；
//   path：文件路径；
//   detail：额外原因，可为空。
// 返回：
//   kIoError 状态。
Status IoError(const C8* action, const std::string& path,
               const std::string& detail) {
  std::string message = "file ";
  message += action;
  message += " failed: ";
  message += path;
  if (!detail.empty()) {
    message += " (";
    message += detail;
    message += ")";
  }
  return MakeStatus(ErrorCode::kIoError, std::move(message));
}

}  // namespace

Status GetFileSize(const std::string& path, U64* size) {
  if (path.empty() || size == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "GetFileSize: path must not be empty and size must not be null");
  }
  std::ifstream stream(path, std::ios::binary | std::ios::in);
  if (!stream.is_open()) {
    return IoError("open", path, "");
  }
  stream.seekg(0, std::ios::end);
  const std::streampos end = stream.tellg();
  if (end == std::streampos(-1)) {
    return IoError("size", path, "tellg failed");
  }
  *size = static_cast<U64>(static_cast<std::streamoff>(end));
  return Status::Ok();
}

Status ReadFileRange(const std::string& path, U64 offset, U64 length,
                     std::vector<U8>* out) {
  if (out == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ReadFileRange: output must not be null");
  }
  FileReader reader;
  Status status = reader.Open(path);
  if (!status.ok()) {
    return status;
  }
  return reader.ReadAt(offset, length, out);
}

FileReader::FileReader() = default;

FileReader::~FileReader() { Close(); }

Status FileReader::Open(const std::string& path) {
  if (path.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "FileReader::Open: path must not be empty");
  }
  Close();
  stream_.clear();
  stream_.open(path, std::ios::binary | std::ios::in);
  if (!stream_.is_open()) {
    return IoError("open", path, "");
  }
  stream_.seekg(0, std::ios::end);
  const std::streampos end = stream_.tellg();
  if (end == std::streampos(-1)) {
    stream_.close();
    return IoError("size", path, "tellg failed");
  }
  size_ = static_cast<U64>(static_cast<std::streamoff>(end));
  position_ = 0;
  stream_.seekg(0, std::ios::beg);
  if (!stream_.good()) {
    stream_.close();
    size_ = 0;
    return IoError("seek", path, "seek to begin failed");
  }
  return Status::Ok();
}

Bool FileReader::IsOpen() const { return stream_.is_open(); }

Status FileReader::Size(U64* size) const {
  if (size == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "FileReader::Size: output must not be null");
  }
  if (!stream_.is_open()) {
    return MakeStatus(ErrorCode::kStateError,
                      "FileReader::Size: file is not open");
  }
  *size = size_;
  return Status::Ok();
}

Status FileReader::ReadAt(U64 offset, U64 length, std::vector<U8>* out) {
  if (out == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "FileReader::ReadAt: output must not be null");
  }
  if (!stream_.is_open()) {
    return MakeStatus(ErrorCode::kStateError,
                      "FileReader::ReadAt: file is not open");
  }
  if (offset > size_ || length > size_ - offset) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "FileReader::ReadAt: range exceeds file size");
  }
  out->resize(static_cast<Usize>(length));
  U64 done = 0;
  while (done < length) {
    const U64 remaining = length - done;
    const Usize chunk = static_cast<Usize>(
        std::min<U64>(remaining, kReadChunkBytes));
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(offset + done),
                  std::ios::beg);
    if (!stream_.good()) {
      return IoError("seek", "", "ReadAt seek failed");
    }
    stream_.read(reinterpret_cast<C8*>(out->data() + done),
                 static_cast<std::streamsize>(chunk));
    if (stream_.gcount() != static_cast<std::streamsize>(chunk)) {
      return IoError("read", "", "unexpected end of file");
    }
    done += chunk;
  }
  return Status::Ok();
}

Status FileReader::ReadNext(U64 max_length, std::vector<U8>* out,
                            Usize* read_bytes) {
  if (out == nullptr || read_bytes == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "FileReader::ReadNext: output/read_bytes must not be null");
  }
  if (!stream_.is_open()) {
    return MakeStatus(ErrorCode::kStateError,
                      "FileReader::ReadNext: file is not open");
  }
  const U64 remaining = size_ - position_;
  const U64 want = std::min<U64>(remaining, max_length);
  const Usize chunk = static_cast<Usize>(want);
  out->resize(chunk);
  if (chunk == 0) {
    *read_bytes = 0;
    return Status::Ok();
  }
  stream_.clear();
  stream_.seekg(static_cast<std::streamoff>(position_), std::ios::beg);
  if (!stream_.good()) {
    return IoError("seek", "", "ReadNext seek failed");
  }
  stream_.read(reinterpret_cast<C8*>(out->data()),
               static_cast<std::streamsize>(chunk));
  if (stream_.gcount() != static_cast<std::streamsize>(chunk)) {
    return IoError("read", "", "unexpected end of file");
  }
  position_ += want;
  *read_bytes = chunk;
  return Status::Ok();
}

void FileReader::Close() {
  if (stream_.is_open()) {
    stream_.close();
  }
  size_ = 0;
  position_ = 0;
}

Status HashFileSha256(const std::string& path, U64 chunk_bytes,
                      std::string* hex) {
  if (hex == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "HashFileSha256: output must not be null");
  }
  const U64 chunk =
      chunk_bytes == 0 ? 4ULL * 1024ULL * 1024ULL : chunk_bytes;
  FileReader reader;
  Status status = reader.Open(path);
  if (!status.ok()) {
    return status;
  }
  Sha256 sha;
  std::vector<U8> buffer;
  while (true) {
    Usize read_bytes = 0;
    status = reader.ReadNext(chunk, &buffer, &read_bytes);
    if (!status.ok()) {
      return status;
    }
    if (read_bytes == 0) {
      break;
    }
    sha.Update(buffer.data(), read_bytes);
  }
  *hex = sha.HexDigest();
  return Status::Ok();
}

}  // namespace runtime
}  // namespace mobilenet
