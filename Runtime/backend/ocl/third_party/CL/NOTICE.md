# Khronos OpenCL Headers（vendored）

本目录包含从 Khronos Group 的 OpenCL Headers 中取用的：

- `cl.h`
- `cl_platform.h`

来源：本地 `Reference/Neat/NEAT_Develop/NEAT_Develop/header/environment/ocl/inc/CL/`
（Khronos OpenCL 2.1 头，文件头保留原始 MIT 许可声明）。

用途：仅用于编译 Runtime 的 OCL 后端；不包含 NEAT 的专有源码。头文件许可为 MIT，
允许复制、修改与再分发，但必须保留版权与许可声明。

后续 P4 可替换为官方 Khronos `OpenCL-Headers`（Apache-2.0）的更新版本；
替换时同步更新本文件与 `cl.h`/`cl_platform.h` 版本说明。
