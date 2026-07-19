[CmdletBinding()]
param(
    [switch]$SkipOpenCV,
    [switch]$SkipTests,

    [string]$NinjaPath
)

# Configure, build and test the project with the MSVC x64 toolchain.
# Run from the repository root: .\scripts\build_windows.ps1

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'windows_common.ps1')

$opencvInstall = Join-Path $repoRoot 'third_party\opencv\4.14.0\install'

if (-not $SkipOpenCV -and -not (Test-Path -LiteralPath $opencvInstall -PathType Container)) {
    & (Join-Path $PSScriptRoot 'build_opencv_windows.ps1') -NinjaPath $NinjaPath
}
if (-not (Test-Path -LiteralPath $opencvInstall -PathType Container)) {
    throw "OpenCV install was not found in $opencvInstall. Run scripts\build_opencv_windows.ps1 first."
}

& (Join-Path $PSScriptRoot 'get_onnxruntime_windows.ps1')

Import-MsvcEnvironment
$env:Path = "$(Join-Path $opencvInstall 'bin');$env:Path"

Push-Location $repoRoot
try {
    Invoke-Checked cmake @('--preset', 'windows-release')
    Invoke-Checked cmake @('--build', '--preset', 'windows-release')
    if (-not $SkipTests) {
        Invoke-Checked ctest @('--preset', 'windows-release')
    }
}
finally {
    Pop-Location
}
