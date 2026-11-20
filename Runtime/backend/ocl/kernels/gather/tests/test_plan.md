# Gather OCL 测试计划

## 1. 测试目标

- OCL Gather 与 Mock Gather 输出逐字节一致；
- 真实形状下测量 OCL kernel 时间、Mock 时间与加速比；
- 所有算子完成后参与端到端 OCL vs Mock Token 一致性。

## 2. 用例

| 用例 | shapes | 内容 | 通过标准 |
| --- | --- | --- | --- |
| small/default | B=1,S=8,H=64,V=128 | 随机合法 ids/权重 | `max_abs=0` |
| small/boundary | B=1,S=1,H=64,V=1 | 单 token、单行权重 | `max_abs=0` |
| small/tail | H=65,V=128 | 非 8 倍数 H | `max_abs=0` |
| real/sample | B=1,S=256,H=1024,V=151936 | 真实形状，合法 ids | `max_abs=0` |
| perf | real/sample | 预热 3、重复 10、`cl_event` | 记录中位数与带宽 |

## 3. 测试环境

- 设备：vivo V2426A / Mali-G925-Immortalis MC12 / OpenCL 3.0；
- 后端：`backend=ocl` vs `backend=mock`；
- 工具：NDK arm64 构建、`adb push`、`/data/local/tmp` 运行；
- 记录 GPU 驱动版本、Android 版本、温度/频率（若可读）。

## 4. 命令

```bash
# 构建（P4 实现后）
cmake -S Runtime/backend -B Runtime/backend/build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
  -DMOBILENET_BACKEND_OCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build Runtime/backend/build-android

# 部署
MSYS_NO_PATHCONV=1 adb push Runtime/backend/build-android/<test> /data/local/tmp/<test>
MSYS_NO_PATHCONV=1 adb shell "chmod 755 /data/local/tmp/<test> && /data/local/tmp/<test>"
```

## 5. 指标

- 数值：max_abs、relative_l2、cosine；Gather 要求 0 误差；
- 性能：kernel 时间中位数、Mock 时间、加速比、有效带宽（读+写字节 / 时间）；
- 端到端：同一输入 Token 的生成 Token 一致率、首个分歧 Step（若发生）。

## 6. 结果记录

实测后生成本目录 `test_results.md`，必须包含环境、命令、实际输出、指标、结论、
未覆盖项与风险；不得只写“通过”。
