@echo off
rem Build Runtime Graph with Zig for local Windows validation only.
rem Requires python (with ziglang), cmake, and ninja on PATH.
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "TOOLS_DIR=%~dp0"

rem Use forward slashes so switching between this script and the Git Bash
rem script does not invalidate the CMake cache and lose archiver overrides.
set "ROOT=%ROOT:\=/%"
set "TOOLS_DIR=%TOOLS_DIR:\=/%"
set "BUILD_DIR=%ROOT%/build"

cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER="%TOOLS_DIR%zig_cxx.cmd" ^
  -DGRAPH_AR="%TOOLS_DIR%zig_ar.cmd" ^
  -DGRAPH_RANLIB="%TOOLS_DIR%zig_ranlib.cmd" %*
if errorlevel 1 exit /b %errorlevel%

cmake --build "%BUILD_DIR%"
if errorlevel 1 exit /b %errorlevel%

ctest --test-dir "%BUILD_DIR%" --output-on-failure
