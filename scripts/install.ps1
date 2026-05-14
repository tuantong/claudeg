# claudeg one-step installer for Windows (PowerShell 5.1+).
$ErrorActionPreference = "Stop"

$Repo    = if ($env:CLAUDEG_REPO)    { $env:CLAUDEG_REPO }    else { "tuantong/claudeg" }
$Version = if ($env:CLAUDEG_VERSION) { $env:CLAUDEG_VERSION } else { "latest" }

$Target = "x86_64-pc-windows-msvc"
if ($Version -eq "latest") {
    $Url = "https://github.com/$Repo/releases/latest/download/claudeg-$Target.zip"
} else {
    $Url = "https://github.com/$Repo/releases/download/$Version/claudeg-$Target.zip"
}

$InstallDir = if ($env:CLAUDEG_INSTALL_DIR) {
    $env:CLAUDEG_INSTALL_DIR
} else {
    if (-not $env:LOCALAPPDATA) {
        Write-Error "LOCALAPPDATA is not set; cannot pick a default install dir. Set CLAUDEG_INSTALL_DIR explicitly."
        exit 1
    }
    Join-Path $env:LOCALAPPDATA "Programs\claudeg"
}
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

$Tmp = New-Item -ItemType Directory -Force -Path (Join-Path $env:TEMP ([System.Guid]::NewGuid().ToString()))
try {
    Write-Host "Downloading $Url"
    $zip = Join-Path $Tmp "claudeg.zip"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $zip
    Expand-Archive -LiteralPath $zip -DestinationPath $Tmp -Force

    # Replace the binary atomically. Windows will block overwriting an EXE
    # that's currently running, and SmartScreen may cache the prior signature;
    # stop any running serve first, then delete-then-copy for a clean inode.
    $destExe = Join-Path $InstallDir "claudeg.exe"
    Get-CimInstance Win32_Process -Filter "Name='claudeg.exe'" |
        Where-Object { $_.CommandLine -match 'serve' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Remove-Item -Force -ErrorAction SilentlyContinue $destExe
    Copy-Item -Force (Join-Path $Tmp "claudeg.exe") $destExe
    Write-Host "Installed $destExe"

    # Persist on user PATH if missing.
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if (-not $userPath) { $userPath = "" }
    if (-not ($userPath -split ";" | Where-Object { $_ -eq $InstallDir })) {
        [Environment]::SetEnvironmentVariable("PATH", "$userPath;$InstallDir", "User")
        $env:PATH = "$env:PATH;$InstallDir"
        Write-Host "Added $InstallDir to user PATH (new shells will see it)."
    }

    Write-Host "Running setup..."
    & (Join-Path $InstallDir "claudeg.exe") setup
} finally {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}
