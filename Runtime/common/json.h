// 模块：Runtime 最小严格 JSON 解析器
//
// 职责：
//   为模型包 manifest 与运行配置提供不依赖第三方的严格 JSON 解析：支持
//   object/array/string/number/bool/null；数字按语法区分整数（I64）与浮点
//   （FP64）；错误信息携带字节偏移，便于定位损坏的 manifest。
//
// 边界：
//   - 只做解析与只读访问，不做序列化、查询语言或 JSON Pointer；
//   - 不支持注释、尾逗号、单引号、NaN/Infinity 等非标准扩展；
//   - 字符串转义按 RFC 8259 处理（含 \\uXXXX 与代理对），不额外校验
//     未转义非 ASCII 字节的 UTF-8 合法性；
//   - 对象重复键视为错误；嵌套深度上限 256，防止栈溢出；
//   - 解析器不抛出异常，全部错误通过 Status 返回。
//
// 依赖：
//   Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_JSON_H_
#define MOBILENET_RUNTIME_COMMON_JSON_H_

#include <string>
#include <vector>

#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {

class JsonValue;
struct JsonMember;

// 只读 JSON 值。
//
// 功能：
//   表示解析后的 null/bool/整数/浮点/字符串/数组/对象，提供类型判断与
//   默认值安全访问；对象成员按键线性查找。
//
// 边界：
//   - 值不可原地修改；需要修改时重新构造；
//   - 类型不匹配的 As* 访问返回类型默认值（0/false/空容器），调用方必须
//     先用 Is* 判断，或使用 Parse 返回的错误定位问题；
//   - 对象成员查找是线性复杂度，仅面向 manifest 这种小对象。
class JsonValue {
 public:
  // JSON 值类型。
  enum class Type : U8 {
    kNull = 0,
    kBool = 1,
    kNumber = 2,
    kString = 3,
    kArray = 4,
    kObject = 5,
  };

  JsonValue();
  JsonValue(const JsonValue& other);
  JsonValue(JsonValue&& other) noexcept;
  JsonValue& operator=(const JsonValue& other);
  JsonValue& operator=(JsonValue&& other) noexcept;
  ~JsonValue();

  // 构造 null 值。
  static JsonValue MakeNull();

  // 构造布尔值。
  static JsonValue MakeBool(Bool value);

  // 构造整数值；IsInteger() 为 true。
  static JsonValue MakeInt(I64 value);

  // 构造浮点值；IsInteger() 为 false。
  static JsonValue MakeFloat(FP64 value);

  // 构造字符串值。
  static JsonValue MakeString(std::string value);

  // 构造数组值。
  static JsonValue MakeArray(std::vector<JsonValue> values);

  // 构造对象值；不检查重复键（解析器负责）。
  static JsonValue MakeObject(std::vector<JsonMember> members);

  // 返回类型。
  Type type() const;

  // 类型判断。
  Bool IsNull() const;
  Bool IsBool() const;
  Bool IsNumber() const;
  Bool IsString() const;
  Bool IsArray() const;
  Bool IsObject() const;
  Bool IsInteger() const;

  // 返回布尔值；非布尔返回 false。
  Bool AsBool() const;

  // 返回整数；非整数返回 0。
  I64 AsInt64() const;

  // 返回浮点值；整数按 FP64 返回，非数字返回 0。
  FP64 AsFloat64() const;

  // 返回字符串；非字符串返回静态空串。
  const std::string& AsString() const;

  // 返回数组；非数组返回静态空数组。
  const std::vector<JsonValue>& AsArray() const;

  // 返回对象成员；非对象返回静态空数组。
  const std::vector<JsonMember>& AsObject() const;

  // 按键查找对象成员。
  //
  // 参数：
  //   key：成员名。
  // 返回：
  //   找到时返回非空指针；非对象或不存在返回 nullptr。
  const JsonValue* Find(const std::string& key) const;

 private:
  // 当前类型。
  Type type_ = Type::kNull;

  // 布尔值。
  Bool bool_value_ = false;

  // 数字是否为整数（true 时 integer_ 有效，否则 number_ 有效）。
  Bool is_integer_ = false;

  // 整数值。
  I64 integer_ = 0;

  // 浮点值。
  FP64 number_ = 0.0;

  // 字符串值。
  std::string string_value_;

  // 数组元素。
  std::vector<JsonValue> array_value_;

  // 对象成员；JsonMember 在定义时包含完整 JsonValue，vector 允许不完整
  // 类型，特殊成员函数均在本文件内 JsonMember 完整定义之后实现。
  std::vector<JsonMember> object_value_;
};

// JSON 对象成员：键 + 值，保留解析时的插入顺序。
//
// 说明：
//   JsonValue 内含 std::vector<JsonMember>，因此本结构必须在 JsonValue
//   之后定义；JsonValue 的特殊成员函数在 json.cc 中此定义可见后实现。
struct JsonMember {
  // 成员名；允许为空字符串（JSON 合法）。
  std::string key;

  // 成员值。
  JsonValue value;
};

// 解析严格 JSON 文本。
//
// 参数：
//   text：UTF-8 文本；必须恰好包含一个 JSON 值（允许首尾空白）；
//   value：输出参数，非空；成功时写入解析结果。
// 返回：
//   成功返回 kOk；文本非法、深度超限、数字溢出或存在尾随数据时返回
//   kInvalidArgument，message 中包含字节偏移与失败原因。
Status ParseJson(const std::string& text, JsonValue* value);

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_JSON_H_
