"""模块：Graph 模块 Zig 构建脚本（无 MSVC/LibTorch 环境）。

职责：
  在仅有 Zig 与 Python torch 头文件的 Windows 环境中编译 Graph 公共库、
  Graph IR、RuntimeGraph、Python 扩展与 C++ 单元测试，并可选执行测试。
  该脚本是 CI/本机复现手段；正式构建入口为 Graph/CMakeLists.txt。

边界：
  - Python 扩展使用 ``MOBILENET_GRAPH_HOST_TORCH_BRIDGE``，不链接 LibTorch，
    通过 torch_python.dll 动态包装 THPVariable；
  - 不下载依赖、不修改源码、不执行真实模型推理；
  - 仅支持 Windows + CPython + torch 已安装的环境。

依赖：
  Python 3.10+、``python -m ziglang``、已安装的 ``torch``。

用法：
  python Tools/build_graph_zig.py [--run-tests]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPOSITORY_ROOT / "Build" / "graph"
OBJECT_DIR = BUILD_ROOT / "obj"
PYTHON_DIR = BUILD_ROOT / "python"
TEST_DIR = BUILD_ROOT / "tests"

COMMON_SOURCES = (
    "Graph/cpp/common/src/graph_common.cc",
    "Graph/cpp/common/src/validation_common.cc",
)
IR_SOURCES = (
    "Graph/cpp/ir/src/fusion_rule.cc",
    "Graph/cpp/ir/src/graph.cc",
    "Graph/cpp/ir/src/graph_builder.cc",
    "Graph/cpp/ir/src/graph_validator.cc",
    "Graph/cpp/ir/src/tensor_desc.cc",
)
RUNTIME_SOURCES = (
    "Graph/cpp/runtime/src/runtime_graph.cc",
    "Graph/cpp/runtime/src/runtime_graph_builder.cc",
    "Graph/cpp/runtime/src/runtime_graph_forward.cc",
    "Graph/cpp/runtime/src/runtime_graph_validator.cc",
)
BASE_SOURCES = (
    "Runtime/common/data_types.cc",
    "Runtime/common/status.cc",
    "Runtime/kernel/kernel_types.cc",
)
PYTHON_SOURCES = (
    "Graph/cpp/python/graph_python.cc",
    "Graph/cpp/python/qwen3_fixture.cc",
    "Graph/cpp/python/torch_shims.cc",
)
TEST_SOURCE = "Tests/unit/graph/test_runtime_graph.cc"


def zig_command() -> list[str]:
    """返回 Zig C++ 驱动命令。"""
    return [sys.executable, "-m", "ziglang", "c++"]


def torch_include_dirs() -> list[Path]:
    """返回 torch 与 pybind11 头文件目录。"""
    import torch

    torch_root = Path(torch.__file__).resolve().parent
    return [
        torch_root / "include",
        torch_root / "include" / "torch" / "csrc" / "api" / "include",
    ]


def python_include_dir() -> Path:
    """返回当前 CPython 头文件目录。"""
    return Path(sys.base_prefix) / "Include"


def python_library_dir() -> Path:
    """返回当前 CPython 导入库目录。"""
    return Path(sys.base_prefix) / "libs"


def common_flags() -> list[str]:
    """返回所有编译单元共用的编译选项。"""
    return [
        "-std=c++17",
        # 说明：与正式 CMake/MSVC Release 对齐使用 -O2；保留 -g 便于定位。
        "-O2",
        "-g",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        f"-I{REPOSITORY_ROOT}",
        f"-I{REPOSITORY_ROOT / 'Graph' / 'cpp' / 'common' / 'include'}",
        f"-I{REPOSITORY_ROOT / 'Graph' / 'cpp' / 'ir' / 'include'}",
        f"-I{REPOSITORY_ROOT / 'Graph' / 'cpp' / 'runtime' / 'include'}",
    ]


def torch_flags() -> list[str]:
    """返回 Graph IR 编译所需 torch 头文件选项。"""
    flags = ["-DC10_USING_CUSTOM_GENERATED_MACROS"]
    flags.extend(f"-I{directory}" for directory in torch_include_dirs())
    return flags


def python_flags() -> list[str]:
    """返回 Python 扩展专用编译选项。"""
    flags = ["-DMOBILENET_GRAPH_HOST_TORCH_BRIDGE", f"-I{python_include_dir()}"]
    flags.extend(torch_flags())
    return flags


def run(command: list[str], description: str) -> None:
    """执行命令；失败时抛出带命令行的 RuntimeError。"""
    print("+ " + " ".join(command))
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code "
                           f"{completed.returncode}")


def compile_object(source: str, extra_flags: list[str],
                   output_name: str) -> Path:
    """编译单个源文件并返回目标文件路径。"""
    source_path = REPOSITORY_ROOT / source
    output_path = OBJECT_DIR / output_name
    command = zig_command() + common_flags() + extra_flags + [
        "-c",
        str(source_path),
        "-o",
        str(output_path),
    ]
    run(command, f"compile {source}")
    return output_path


def build_objects() -> tuple[list[Path], list[Path]]:
    """编译全部 C++ 源文件，返回需要链接的目标文件。"""
    OBJECT_DIR.mkdir(parents=True, exist_ok=True)
    objects: list[Path] = []
    for source in COMMON_SOURCES + RUNTIME_SOURCES + BASE_SOURCES:
        name = source.replace("/", "_").replace(".cc", ".o")
        objects.append(compile_object(source, [], name))
    for source in IR_SOURCES:
        name = source.replace("/", "_").replace(".cc", ".o")
        objects.append(compile_object(source, torch_flags(), name))
    python_objects: list[Path] = []
    for source in PYTHON_SOURCES:
        name = "python_" + source.replace("/", "_").replace(".cc", ".o")
        python_objects.append(compile_object(source, python_flags(), name))
    return objects, python_objects


def link_python_extension(objects: list[Path],
                          python_objects: list[Path]) -> Path:
    """链接 _graph.pyd。"""
    PYTHON_DIR.mkdir(parents=True, exist_ok=True)
    output = PYTHON_DIR / "_graph.pyd"
    command = zig_command() + [
        "-shared",
        "-o",
        str(output),
    ] + [str(path) for path in objects + python_objects] + [
        f"-L{python_library_dir()}",
        f"-lpython{sys.version_info.major}{sys.version_info.minor}",
    ]
    run(command, "link _graph.pyd")
    return output


def build_runtime_test(objects: list[Path]) -> Path:
    """编译并链接 RuntimeGraph 单元测试。"""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    test_object = compile_object(TEST_SOURCE, [], "test_runtime_graph.o")
    output = TEST_DIR / "test_runtime_graph.exe"
    runtime_objects = [
        path
        for path in objects
        if any(
            token in path.name
            for token in (
                "runtime_graph",
                "validation_common",
                "graph_common",
                "data_types",
                "status",
                "kernel_types",
                "fusion_rule",
            )
        )
    ]
    command = zig_command() + ["-o", str(output), str(test_object)]
    command.extend(str(path) for path in runtime_objects)
    run(command, "link test_runtime_graph.exe")
    return output


def run_python_tests() -> None:
    """执行 Python 单元与集成测试。"""
    environment = dict(os.environ)
    # 隔离模式（-I）会忽略 PYTHONPATH，测试脚本自行插入仓库内路径；
    # 这里只告知扩展模块所在目录，兼容不同构建输出位置。
    environment["MOBILENET_GRAPH_PYTHON_DIR"] = str(PYTHON_DIR)
    for test in (
        "Tests/unit/graph/test_graph_ir.py",
        "Tests/integration/graph/test_qwen3_cross_language.py",
    ):
        command = [sys.executable, "-I", test]
        print("+ " + " ".join(command))
        completed = subprocess.run(
            command, cwd=REPOSITORY_ROOT, env=environment, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{test} failed")


def main() -> int:
    """构建入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="构建完成后运行 C++ 与 Python 测试",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="删除 Build/graph 后重新构建",
    )
    arguments = parser.parse_args()
    if arguments.clean and BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "python": sys.version.split()[0],
        "repository": str(REPOSITORY_ROOT),
    }, ensure_ascii=False))
    objects, python_objects = build_objects()
    extension = link_python_extension(objects, python_objects)
    test_executable = build_runtime_test(objects)
    print(f"built {extension}")
    print(f"built {test_executable}")
    if arguments.run_tests:
        run([str(test_executable)], "runtime graph tests")
        run_python_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
