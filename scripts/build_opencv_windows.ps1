[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',

    [ValidateRange(1, 256)]
    [int]$Jobs = [Environment]::ProcessorCount,

    [string]$NinjaPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$rootDirectory = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'windows_common.ps1')

$opencvVersion = '4.14.0'
$opencvCommit = '0654a42e19215ef25b1d367d822f3c630447e7c7'
$opencvRoot = if ($env:OPENCV_ROOT) {
    $env:OPENCV_ROOT
}
else {
    Join-Path $rootDirectory "third_party\opencv\$opencvVersion"
}
$sourceDirectory = Join-Path $opencvRoot 'src'
$installDirectory = Join-Path $opencvRoot 'install'

Import-MsvcEnvironment

if (-not (Test-Path -LiteralPath (Join-Path $sourceDirectory '.git') -PathType Container)) {
    New-Item -ItemType Directory -Force -Path $opencvRoot | Out-Null
    Invoke-Checked git @(
        'clone', '--branch', $opencvVersion, '--depth', '1',
        'https://github.com/opencv/opencv.git', $sourceDirectory
    )
}

$actualCommit = (& git -c "safe.directory=$sourceDirectory" -C $sourceDirectory rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to inspect the local OpenCV checkout.'
}
if ($actualCommit -ne $opencvCommit) {
    throw "Unexpected OpenCV commit: $actualCommit (expected $opencvCommit)"
}

$ninja = Get-NinjaPath $NinjaPath
$generator = if ($ninja) { 'Ninja' } else { 'NMake Makefiles' }
$generatorName = if ($ninja) { 'ninja' } else { 'nmake' }
$buildDirectory = Join-Path $opencvRoot "build-windows-$generatorName"
$configureArguments = @(
    '-S', $sourceDirectory,
    '-B', $buildDirectory,
    '-G', $generator,
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DCMAKE_INSTALL_PREFIX=$installDirectory",
    '-DBUILD_SHARED_LIBS=ON',
    '-DBUILD_LIST=core,imgproc,imgcodecs,dnn',
    '-DBUILD_TESTS=OFF',
    '-DBUILD_PERF_TESTS=OFF',
    '-DBUILD_EXAMPLES=OFF',
    '-DBUILD_opencv_apps=OFF',
    '-DBUILD_opencv_java=OFF',
    '-DBUILD_opencv_js=OFF',
    '-DBUILD_opencv_python2=OFF',
    '-DBUILD_opencv_python3=OFF',
    '-DBUILD_JAVA=OFF',
    '-DBUILD_PROTOBUF=ON',
    '-DOPENCV_FORCE_3RDPARTY_BUILD=ON',
    '-DOPENCV_BIN_INSTALL_PATH=bin',
    '-DOPENCV_LIB_INSTALL_PATH=lib',
    '-DOPENCV_3P_LIB_INSTALL_PATH=lib',
    '-DOPENCV_INCLUDE_INSTALL_PATH=include',
    '-DOPENCV_CONFIG_INSTALL_PATH=lib/cmake/opencv4',
    '-DOPENCV_GENERATE_PKGCONFIG=OFF',
    '-DWITH_FFMPEG=OFF',
    '-DWITH_GSTREAMER=OFF',
    '-DWITH_IPP=OFF',
    '-DWITH_ITT=OFF',
    '-DWITH_OPENCL=OFF',
    '-DOPENCV_DNN_OPENCL=OFF',
    '-DWITH_OPENEXR=OFF',
    '-DWITH_OPENJPEG=OFF',
    '-DWITH_TIFF=OFF',
    '-DWITH_WEBP=OFF'
)
if ($ninja) {
    $configureArguments += "-DCMAKE_MAKE_PROGRAM=$ninja"
}

Write-Host "Configuring OpenCV $opencvVersion ($generator, $Configuration)"
Invoke-Checked cmake $configureArguments

$buildArguments = @('--build', $buildDirectory)
if ($ninja) {
    $buildArguments += @('--parallel', $Jobs.ToString())
}
Invoke-Checked cmake $buildArguments
Invoke-Checked cmake @('--install', $buildDirectory)

$opencvCMakeDirectory = Get-OpenCvCMakeDirectory $installDirectory
Write-Host "OpenCV $opencvVersion installed in $installDirectory"
Write-Host "Use OpenCV_DIR=$opencvCMakeDirectory"
