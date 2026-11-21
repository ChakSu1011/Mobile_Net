// 模块：Tokenizer Qwen3 预分词扫描器实现
//
// 职责：
//   在码点序列上按 Qwen3 正则的备选顺序与贪婪/回溯语义切分文本。
//
// 边界：
//   输入已由调用方完成 UTF-8 校验；本文件仍对非法输入返回错误。
//
// 主要逻辑：
//   1. 将文本解码为带字节位置的码点数组；
//   2. 在每个位置依次尝试 contraction、可选非字母+字母、单个数字、
//      标点、换行组、空白前瞻、空白七个分支；
//   3. 用码点的字节位置还原原始 UTF-8 片段。

#include "Service/tokenizer/pretokenizer.h"

#include "Service/tokenizer/unicode_util.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {
namespace {

// 正则允许的 contraction 后缀；顺序与正则备选一致。
constexpr const C8* kContractionSuffixes[] = {"s", "t", "re", "ve",
                                              "m", "ll", "d"};
constexpr Usize kContractionSuffixCount =
    sizeof(kContractionSuffixes) / sizeof(kContractionSuffixes[0]);

// 返回从 position 开始匹配的 contraction 码点数；不匹配返回 0。
//
// 参数：
//   codepoints：码点数组；
//   position：起始下标。
// 返回：
//   匹配长度（码点数）；0 表示不匹配。
Usize MatchContraction(const std::vector<Codepoint>& codepoints,
                       Usize position) {
  if (position >= codepoints.size() ||
      codepoints[position].value != static_cast<I32>('\'')) {
    return 0;
  }
  for (Usize suffix_index = 0; suffix_index < kContractionSuffixCount;
       ++suffix_index) {
    const C8* suffix = kContractionSuffixes[suffix_index];
    Usize cursor = position + 1;
    Bool matched = true;
    for (Usize char_index = 0; suffix[char_index] != '\0'; ++char_index) {
      if (cursor >= codepoints.size() ||
          ToLowerAscii(codepoints[cursor].value) !=
              static_cast<I32>(suffix[char_index])) {
        matched = false;
        break;
      }
      ++cursor;
    }
    if (matched) {
      return cursor - position;
    }
  }
  return 0;
}

// 返回可选非字母/数字前缀 + 字母串的匹配长度；不匹配返回 0。
Usize MatchOptionalPrefixLetters(const std::vector<Codepoint>& codepoints,
                                 Usize position) {
  const Usize size = codepoints.size();
  Usize cursor = position;
  if (cursor >= size) {
    return 0;
  }
  if (!IsLetter(codepoints[cursor].value)) {
    // 可选字符不能是 CR/LF、字母或数字。
    if (IsNewline(codepoints[cursor].value) ||
        IsNumber(codepoints[cursor].value)) {
      return 0;
    }
    if (cursor + 1 >= size || !IsLetter(codepoints[cursor + 1].value)) {
      return 0;
    }
    ++cursor;
  }
  while (cursor < size && IsLetter(codepoints[cursor].value)) {
    ++cursor;
  }
  return cursor - position;
}

// 返回标点串分支的匹配长度；不匹配返回 0。
Usize MatchPunctuationRun(const std::vector<Codepoint>& codepoints,
                          Usize position) {
  const Usize size = codepoints.size();
  Usize cursor = position;
  if (cursor >= size) {
    return 0;
  }
  // 可选的 ASCII 空格前缀。
  if (codepoints[cursor].value == static_cast<I32>(' ')) {
    if (cursor + 1 >= size ||
        IsWhitespace(codepoints[cursor + 1].value) ||
        IsLetter(codepoints[cursor + 1].value) ||
        IsNumber(codepoints[cursor + 1].value)) {
      return 0;
    }
    ++cursor;
  }
  if (cursor >= size || IsWhitespace(codepoints[cursor].value) ||
      IsLetter(codepoints[cursor].value) ||
      IsNumber(codepoints[cursor].value)) {
    return 0;
  }
  while (cursor < size && !IsWhitespace(codepoints[cursor].value) &&
         !IsLetter(codepoints[cursor].value) &&
         !IsNumber(codepoints[cursor].value)) {
    ++cursor;
  }
  while (cursor < size && IsNewline(codepoints[cursor].value)) {
    ++cursor;
  }
  return cursor - position;
}

// 返回 \s*[\r\n]+ 分支的匹配长度；不匹配返回 0。
Usize MatchWhitespaceNewline(const std::vector<Codepoint>& codepoints,
                             Usize position) {
  const Usize size = codepoints.size();
  if (position >= size || !IsWhitespace(codepoints[position].value)) {
    return 0;
  }
  Usize run_end = position;
  while (run_end < size && IsWhitespace(codepoints[run_end].value)) {
    ++run_end;
  }
  Usize last_newline = size;
  for (Usize cursor = position; cursor < run_end; ++cursor) {
    if (IsNewline(codepoints[cursor].value)) {
      last_newline = cursor;
    }
  }
  if (last_newline == size) {
    return 0;
  }
  Usize match_end = last_newline + 1;
  while (match_end < run_end && IsNewline(codepoints[match_end].value)) {
    ++match_end;
  }
  return match_end - position;
}

// 返回 \s+(?!\S) 分支的匹配长度；不匹配返回 0。
//
// 说明：
//   贪婪匹配整个空白串；若其后仍有非空白字符，则回退一个空白字符，
//   使负向前瞻成立。这正是 Rust regex 的回溯语义。
Usize MatchWhitespaceLookahead(const std::vector<Codepoint>& codepoints,
                               Usize position) {
  const Usize size = codepoints.size();
  if (position >= size || !IsWhitespace(codepoints[position].value)) {
    return 0;
  }
  Usize run_end = position;
  while (run_end < size && IsWhitespace(codepoints[run_end].value)) {
    ++run_end;
  }
  if (run_end == size) {
    return run_end - position;
  }
  if (run_end - position >= 2) {
    return run_end - position - 1;
  }
  return 0;
}

// 返回 \s+ 分支的匹配长度；不匹配返回 0。
Usize MatchWhitespaceRun(const std::vector<Codepoint>& codepoints,
                         Usize position) {
  const Usize size = codepoints.size();
  Usize cursor = position;
  while (cursor < size && IsWhitespace(codepoints[cursor].value)) {
    ++cursor;
  }
  return cursor - position;
}

}  // namespace

Status SplitPretoken(const std::string& text,
                     std::vector<std::string>* pieces) {
  if (pieces == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SplitPretoken: pieces must not be null");
  }
  pieces->clear();
  if (text.empty()) {
    return Status::Ok();
  }

  std::vector<Codepoint> codepoints;
  Status status = DecodeUtf8(text, &codepoints);
  if (!status.ok()) {
    return PrefixStatus(status, "SplitPretoken");
  }

  const Usize size = codepoints.size();
  Usize position = 0;
  while (position < size) {
    Usize match_length = MatchContraction(codepoints, position);
    if (match_length == 0) {
      match_length = MatchOptionalPrefixLetters(codepoints, position);
    }
    if (match_length == 0 &&
        IsNumber(codepoints[position].value)) {
      match_length = 1;
    }
    if (match_length == 0) {
      match_length = MatchPunctuationRun(codepoints, position);
    }
    if (match_length == 0) {
      match_length = MatchWhitespaceNewline(codepoints, position);
    }
    if (match_length == 0) {
      match_length = MatchWhitespaceLookahead(codepoints, position);
    }
    if (match_length == 0) {
      match_length = MatchWhitespaceRun(codepoints, position);
    }
    if (match_length == 0) {
      return MakeStatus(
          ErrorCode::kInternal,
          "SplitPretoken: no regex branch matched at byte " +
              std::to_string(codepoints[position].offset));
    }
    const Usize begin = codepoints[position].offset;
    const Codepoint& last = codepoints[position + match_length - 1];
    const Usize end = last.offset + last.length;
    pieces->push_back(text.substr(begin, end - begin));
    position += match_length;
  }
  return Status::Ok();
}

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet
