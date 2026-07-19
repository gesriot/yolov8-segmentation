[CmdletBinding()]
param()

# Downloads the pinned ONNX Runtime CPU release into third_party/onnxruntime.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$rootDirectory = Split-Path -Parent $PSScriptRoot

$ortVersion = '1.27.1'
$ortSha256 = ''
$archiveName = "onnxruntime-win-x64-$ortVersion.zip"
$downloadUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$ortVersion/$archiveName"
$ortRoot = Join-Path $rootDirectory "third_party\onnxruntime\$ortVersion"

if (Test-Path -LiteralPath (Join-Path $ortRoot 'lib\onnxruntime.dll') -PathType Leaf) {
    Write-Host "ONNX Runtime $ortVersion is already installed in $ortRoot"
    return
}

New-Item -ItemType Directory -Force -Path $ortRoot | Out-Null
$archivePath = Join-Path $ortRoot $archiveName

Write-Host "Downloading $downloadUrl"
Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath

if ($ortSha256) {
    $actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ortSha256) {
        throw "Unexpected $archiveName hash: $actual (expected $ortSha256)"
    }
}

Expand-Archive -LiteralPath $archivePath -DestinationPath $ortRoot -Force
# The archive wraps everything in onnxruntime-win-x64-<version>/; flatten it.
$inner = Join-Path $ortRoot "onnxruntime-win-x64-$ortVersion"
foreach ($entry in Get-ChildItem -LiteralPath $inner) {
    Move-Item -LiteralPath $entry.FullName -Destination $ortRoot -Force
}
Remove-Item -LiteralPath $inner -Recurse -Force
Remove-Item -LiteralPath $archivePath -Force

if (-not (Test-Path -LiteralPath (Join-Path $ortRoot 'lib\onnxruntime.dll') -PathType Leaf)) {
    throw "ONNX Runtime extraction failed: onnxruntime.dll is missing in $ortRoot\lib"
}
Write-Host "ONNX Runtime $ortVersion installed in $ortRoot"
