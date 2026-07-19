Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter()]
        [string[]]$ArgumentList = @()
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-VcVarsPath {
    if ($env:VCVARS64_PATH -and (Test-Path -LiteralPath $env:VCVARS64_PATH -PathType Leaf)) {
        return (Resolve-Path -LiteralPath $env:VCVARS64_PATH).Path
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        $instances = & $vswhere -all -products '*' -format json | ConvertFrom-Json
        foreach ($instance in @($instances | Sort-Object installationVersion -Descending)) {
            $candidate = Join-Path $instance.installationPath 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return $candidate
            }
        }
    }

    $roots = @(
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio')
    )
    foreach ($root in $roots) {
        if (-not (Test-Path -LiteralPath $root -PathType Container)) {
            continue
        }
        $candidate = Get-ChildItem -LiteralPath $root -Filter vcvars64.bat -File -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
    }

    throw 'MSVC x64 tools were not found. Install the Visual Studio C++ desktop workload.'
}

function Import-MsvcEnvironment {
    $currentCompiler = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($currentCompiler -and $env:VSCMD_ARG_TGT_ARCH -eq 'x64') {
        return
    }

    $vcvars = Get-VcVarsPath
    Write-Host "Loading MSVC x64 environment from $vcvars"
    $environmentLines = & $env:ComSpec /d /s /c "`"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "vcvars64.bat failed with exit code $LASTEXITCODE"
    }

    # Some PowerShell hosts pass both PATH and Path to cmd.exe. vcvars updates
    # PATH, while the stale mixed-case duplicate can otherwise overwrite it.
    $importedNames = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    foreach ($line in $environmentLines) {
        if ($line -match '^([^=]+)=(.*)$') {
            if ($importedNames.Add($Matches[1])) {
                [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
            }
        }
    }

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw 'vcvars64.bat completed, but cl.exe is still unavailable.'
    }
}

function Get-NinjaPath {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        if (-not (Test-Path -LiteralPath $RequestedPath -PathType Leaf)) {
            throw "Ninja was not found at $RequestedPath"
        }
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    $command = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($command) {
        $item = Get-Item -LiteralPath $command.Source -Force
        if ($item.LinkType -eq 'SymbolicLink' -and $item.Target) {
            return [string]$item.Target
        }
        return $command.Source
    }

    return $null
}

function Get-OpenCvCMakeDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallDirectory
    )

    $preferred = Join-Path $InstallDirectory 'lib\cmake\opencv4'
    $preferredModuleConfig = Join-Path $preferred 'lib'
    if (Test-Path -LiteralPath (Join-Path $preferredModuleConfig 'OpenCVModules.cmake') -PathType Leaf) {
        return $preferredModuleConfig
    }
    if (Test-Path -LiteralPath (Join-Path $preferred 'OpenCVConfig.cmake') -PathType Leaf) {
        return $preferred
    }

    $config = Get-ChildItem -LiteralPath $InstallDirectory -Filter OpenCVConfig.cmake `
        -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $config) {
        throw "OpenCVConfig.cmake was not found below $InstallDirectory"
    }
    return $config.Directory.FullName
}
