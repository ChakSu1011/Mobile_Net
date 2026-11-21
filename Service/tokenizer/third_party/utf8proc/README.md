# utf8proc v2.10.0

- 来源：https://github.com/JuliaStrings/utf8proc/tree/v2.10.0
- 许可：MIT（见 `LICENSE.md`）
- 文件：`utf8proc.c`、`utf8proc.h`、`utf8proc_data.c`
- 用途：Tokenizer 的 UTF-8 解码、NFC 归一化与 Unicode 类别查询。
- 构建：`Service/tokenizer/CMakeLists.txt` 编译为 `mobilenet_utf8proc`
  静态库，并定义 `UTF8PROC_STATIC`。
- 修改：未修改第三方源码。
