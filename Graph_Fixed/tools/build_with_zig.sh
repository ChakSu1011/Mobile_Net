#!/usr/bin/env bash
# 模块：使用 Zig 构建 Runtime Graph 的 Git Bash 开发脚本。
# 职责：为无 MSVC 的 Windows/Git Bash 环境配置 CMake 并构建、运行测试。
# 边界：要求 PATH 中的 python 已安装 ziglang、cmake、ninja；不用于生产发布。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/Build/graph_fixed"

# CMake 在 Windows 上更偏好 D:/... 形式；Git Bash 提供 cygpath -m。
to_mixed_path() {
  local path
  if command -v cygpath >/dev/null 2>&1; then
    path="$(cygpath -m "$1")"
  else
    path="$1"
  fi
  # CMake 在 Windows 上会把盘符规范成小写；这里保持一致，避免与 .cmd
  # 脚本交替使用时因编译器路径大小写不同触发缓存重建并丢失归档器覆盖。
  printf '%s' "${path,}"
}

cmake -S "$(to_mixed_path "${ROOT}")" -B "$(to_mixed_path "${BUILD_DIR}")" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(to_mixed_path "${SCRIPT_DIR}/zig_cxx.cmd")" \
  -DGRAPH_FIXED_AR="$(to_mixed_path "${SCRIPT_DIR}/zig_ar.cmd")" \
  -DGRAPH_FIXED_RANLIB="$(to_mixed_path "${SCRIPT_DIR}/zig_ranlib.cmd")" \
  "$@"

cmake --build "$(to_mixed_path "${BUILD_DIR}")"
ctest --test-dir "$(to_mixed_path "${BUILD_DIR}")" --output-on-failure
