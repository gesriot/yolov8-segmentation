# OpenCV source build on Windows

## Pinned baseline

- OpenCV tag: `4.14.0`
- Peeled tag commit: `0654a42e19215ef25b1d367d822f3c630447e7c7`
- Target: Windows x64 with the MSVC ABI
- Build type: Release by default
- Library type: shared (`.dll` plus MSVC import libraries)
- Install prefix: `third_party/opencv/4.14.0/install`

## Prerequisites

Install:

1. Visual Studio 2022 or newer with **Desktop development with C++** and a
   Windows 10/11 SDK;
2. CMake 3.24 or newer;
3. Git.

Ninja is optional but strongly recommended. Without it, the scripts use the
single-process `NMake Makefiles` generator. If Visual Studio cannot be
discovered, set `VCVARS64_PATH` to its `VC\Auxiliary\Build\vcvars64.bat` file.

## Build only OpenCV

```powershell
.\scripts\build_opencv_windows.ps1
```

The configuration enables `core`, `imgproc`, `imgcodecs` and `dnn`, uses
bundled Protobuf, and disables OpenCL, FFmpeg, GStreamer, IPP, tests, examples,
Java and Python bindings. It is safe to rerun; CMake rebuilds only invalidated
files.

Useful options:

```powershell
.\scripts\build_opencv_windows.ps1 -Configuration Debug -NinjaPath C:\tools\ninja.exe
```

## Use the CMake preset manually

The preset assumes MSVC is already active. Use a Developer PowerShell terminal.
The OpenCV `bin` directory must be in `PATH` when running the CLI manually;
CTest gets it through the test environment automatically:

```powershell
cmake --preset windows-release
cmake --build --preset windows-release
ctest --preset windows-release
```

## Generated directories

These directories are ignored by Git and may be removed independently for a
clean rebuild:

- `third_party/opencv/4.14.0/build-windows-ninja` (or `build-windows-nmake`);
- `third_party/opencv/4.14.0/install`;
- `third_party/opencv/4.14.0/src`;
- `build/windows-release`.
