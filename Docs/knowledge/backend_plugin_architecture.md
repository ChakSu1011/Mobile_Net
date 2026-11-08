# 多后端插件架构知识库（NEAT 学习与 MobileNet 设计）

> 本文件是学习沉淀，不是交付设计文档。设计文档见
> `MobileNet_推理引擎_设计文档.md` 第 4.4 节。
>
> 参考工程：`D:\Neat\NEAT_Develop\NEAT_Develop`。仅学习其架构思想，
> 不复制其专有代码。

---

## 1. 目的与范围

本知识库回答三个问题：

1. NEAT 如何把 Host、NEON、OCL、HVX 等多个设备后端统一到一套运行时。
2. 为什么它的注册方式还不能做到“新增后端只加一个目录”。
3. MobileNet 如何改造为目录级静态插件，并保证 Loader/Destroy 不出现后端专有代码。

---

## 2. NEAT 架构要点

### 2.1 Env（后端环境）

- `ArcN_Env` 是一个后端的运行时抽象，内部持有 `vtable` 与 `accelerator`。
- Env 虚表包含六类能力：
  - `new_env` / `dctor`：生命周期；
  - `convert_param_blob` / `destroy_param_blob`：权重转换与释放；
  - `create_pool`：创建该后端的内存池；
  - `gen_op_env`：为某个 Op 生成该后端的算子实现。
- 代表文件：
  - `header/framework/inc/ArcN_Env.h`
  - `source/environment/ocl/src/ArcN_Env_Ocl.c`
  - `source/environment/host/src/ArcN_Env_Host.c`

### 2.2 OpEnv（算子实现）

- `ArcN_OpEnv` 表示“某后端上的某个算子”。
- 虚表包含 `realize`、`uninit`、`preforward`、`forward`；设计期还包含
  `optimize`、`valloc_free`。
- Env 的 `gen_op_env` 内部用宏表把 `ARCN_OPTYPE_*` 映射到具体的
  `ArcN_*_OpEnv_<Backend>` 类。Host 后端还按 dtype 区分浮点表与量化表。

### 2.3 注册与选择

- 后端枚举与 vtable ID 集中在 `source/glue/ArcN_VTableIDs.c`：
  - 后端枚举只追加，不删除、不重排；
  - 每个 op × 后端组合声明一个 vtable ID；
  - 全局 ID 用高位标记，运行时二分查找解析为本地索引；
  - `g_ArcN_Env_VTableArray` 把后端 ID 映射到 Env 虚表；
  - `ArcN_Accel_Name_2_GID("Ocl")` 完成名称到 ID 的转换。
- 运行/转换侧先 `ArcN_EnvCreate(&env, accel_id)`，再把 Env 绑定到 Net；
  Net 为每个 Op 调用 Env 的 `gen_op_env`。
- 编译期通过 `stdafx.h` 中的 `<Backend>_Env_Sel on/off` 决定哪些后端
  参与编译。

### 2.4 目录与构建

- 每个后端一套目录：
  - `header/environment/<backend>/inc`
  - `source/environment/<backend>/src`
- 目录内包含 Env、Pool、Data、Interface、Ops、Kernels。
- 构建工程按后端显式加入源文件；不同平台有各自的工程配置。

### 2.5 执行链路

```text
配置 accelerator 名
  → ArcN_Accel_Name_2_GID
  → ArcN_EnvCreate(&env, id)
  → Net 绑定 env
  → env->gen_op_env 逐 Op 生成 OpEnv
  → OpEnv->realize 分配资源
  → OpEnv->forward 执行
```

---

## 3. 优点与局限

### 3.1 优点

- Env 虚表把设备、内存池、权重转换、算子生成收敛到一个后端入口。
- OpEnv 虚表让同一算子在每个后端有独立实现，Host 可作为参考/兜底。
- 权重转换由后端负责，上层不感知 dtype 与布局差异。
- 编译期开关可以裁剪不需要的后端。

### 3.2 相对“只加一个目录”的局限

1. 新增后端必须修改中心文件 `ArcN_VTableIDs.c`，为每个
   op × 后端组合声明 vtable，中心表会成为多后端合并热点。
2. 新增后端必须修改各目标平台的 `stdafx.h` 开关。
3. 构建工程需要按后端显式加入源文件。
4. 算子映射是 Env 内部的中心宏表；新增算子仍要改中心注册表。
5. 注册是纯编译期的，没有运行时插件发现机制。

---

## 4. MobileNet 的设计映射

| NEAT | MobileNet | 说明 |
| --- | --- | --- |
| `ArcN_Env` | `Backend` | 设备、内存、权重转换、Kernel 工厂 |
| `ArcN_OpEnv` | `Kernel` | 单算子设备实现 |
| Accelerator ID | 后端名称 + `BackendRegistry` | 按名称选择，不再维护全局 op×后端矩阵 |
| `ENV_*_GEN_MAP` | 后端私有 Kernel 表 | 由 `Backend::CreateKernel` 暴露 |
| `ArcN_VTableIDs.c` | 各后端目录的自注册 | 新增后端不改中心文件 |
| `stdafx.h` 开关 | `MOBILENET_BACKEND_<NAME>` + 构建自动发现 | 新增目录即可参与构建 |
| `convert_param_blob` | `Backend::ConvertParam` | 权重 dtype/布局/量化预处理 |

改造原则：

1. 公共契约只保留 `Backend`、`Kernel` 与少量描述结构。
2. 每个后端目录自注册，不维护全局 op×后端矩阵。
3. 构建自动遍历 `Runtime/backend/*/CMakeLists.txt`。
4. 运行时按配置名选择后端，Loader 创建并注入 Executor。
5. 保留后端权重转换能力，但改为公共接口。
6. 增加加载期能力校验与统一契约测试。

---

## 5. 目录级插件规范

```text
Runtime/backend/
├── backend.h/.cc                   # Backend 抽象、配置、能力
├── backend_registry.h/.cc          # 注册表与自注册宏
├── ocl/                            # OpenCL 后端（首期）
│   ├── ocl_backend.h/.cc
│   ├── kernels/
│   ├── register_ocl.cc
│   └── CMakeLists.txt
├── vulkan/                         # 后续：只新增本目录
├── cuda/                           # 后续
└── template/                       # 新后端骨架
```

每个后端目录必须提供：

1. `Backend` 派生类，实现统一接口。
2. 该后端全部 Kernel。
3. 自注册入口。
4. `CMakeLists.txt`。

构建规则：

- 根构建自动发现子目录，新增目录无需改根 CMake。
- 每个后端以 object library 链接，或使用 whole-archive，防止自注册对象
  被链接器裁剪。
- 每个后端有 `MOBILENET_BACKEND_<NAME>` 开关；首期仅 OCL 默认开启。
- 测试用 `Runtime/backend/mock/` 仅验证插件机制，不进入产品配置。

---

## 6. 资源封装与生命周期

### 6.1 所有权

- Backend 实例：Loader 创建，Engine 持有，Executor 只持非拥有指针。
- 原生 Context/Queue/Program：Backend 独占，上层不可见。
- Buffer/Kernel 句柄：Backend 创建，运行时对象持有并通过 Backend 释放。
- `Backend::Deinit` 作为最后兜底，调试构建报告残留句柄。

### 6.2 释放顺序

```text
停止调度
  → Backend::Synchronize
  → Backend::DestroyKernel（逐 Node）
  → Backend::DestroyBuffer（KV/Activation/权重/暂存）
  → Backend::Deinit（Program 缓存/Context/Queue）
  → 销毁 Backend 实例
```

### 6.3 约束

- `Backend::Init` 全有或全无；失败时清理内部已创建资源。
- `Backend::Deinit` 幂等；重复 Destroy 不重复释放。
- 部分加载失败与正常 Destroy 使用同一逆序回滚路径。
- Loader/Destroy 不得直接调用任何设备 API。
- 公共头文件不得出现 `cl_*`、`Vk*`、`cuda*` 类型。
- Backend 调用由 Engine 串行化，不要求内部线程安全。
- 显存计划以统一对齐排版；Loader 用 `BackendCapability.min_alignment`
  校验，不满足则加载失败，不做静默重排。

---

## 7. 新后端接入清单

1. 复制 `Runtime/backend/template/` 为 `Runtime/backend/<name>/`。
2. 实现 Backend 接口并注册后端名。
3. 实现目标模型所需 Kernel 并注册。
4. 打开 `MOBILENET_BACKEND_<NAME>`，构建自动发现并编译。
5. 通过后端契约测试：缓冲往返、逐算子数值、错误路径、生命周期。
6. 推理配置指定 `backend=<name>`，加载并执行。
7. 检查上层源码零改动（git diff 仅新目录与配置）。

---

## 8. 常见陷阱

1. **静态注册被裁剪**：静态库中的自注册对象可能被链接器丢弃，必须用
   object library、whole-archive 或显式 anchor。
2. **公共头泄漏厂商类型**：`backend.h`、`kernel.h` 不得 include
   OpenCL/Vulkan/CUDA 头文件。
3. **Loader 出现设备分支**：一旦出现 `#ifdef OCL`，插件边界即被破坏。
4. **Destroy 顺序错误**：先 Deinit 再销毁 Buffer/Kernel 会导致悬空句柄。
5. **权重转换未封装**：dtype/布局差异若散落在 Loader，会迫使每个后端改
   Loader。
6. **对齐假设不一致**：离线 MemoryPlan 与后端 `min_alignment` 不一致时，
   应加载期报错，而不是运行时静默重排。
7. **能力校验缺失**：缺算子/缺 dtype 若拖到首个 Token 才暴露，定位成本高。
8. **误以为插件减少 Kernel 工作量**：插件只消除接入改造，Kernel 实现量
   不变。
9. **公共接口被单个后端修改**：公共契约变更需架构评审并同步所有后端；
   后端目录只能实现接口，不能修改接口。

---

## 9. 参考文件

- `header/framework/inc/ArcN_Env.h`
- `header/framework/inc/ArcN_OpEnv.h`
- `header/framework/inc/ArcN_Net.h`
- `header/framework/inc/ArcN_Pool.h`
- `source/glue/ArcN_VTableIDs.c`
- `source/environment/ocl/src/ArcN_Env_Ocl.c`
- `source/environment/host/src/ArcN_Env_Host.c`
- `project/android/arm/64/stdafx.h`
