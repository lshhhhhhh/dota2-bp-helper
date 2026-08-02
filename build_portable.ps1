param(
    [string]$Python = "python",
    [string]$Version = "0.1.0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Join-Path $root "dist"
$name = "Dota2BPHelper-$Version-win64"

Set-Location $root
& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name "Dota2BPHelper" `
    --add-data "data;data" `
    --add-data "artifacts/mvp;artifacts/mvp" `
    --add-data "artifacts/models;artifacts/models" `
    "dota2_bp_helper.py"

$package = Join-Path $out $name
if (Test-Path -LiteralPath $package) {
    Remove-Item -LiteralPath $package -Recurse -Force
}
Move-Item -LiteralPath (Join-Path $out "Dota2BPHelper") -Destination $package
Copy-Item -LiteralPath "LICENSE", "THIRD_PARTY_NOTICES.md", "DATA_SOURCES.md" -Destination $package
if (Test-Path -LiteralPath "licenses") {
    Copy-Item -LiteralPath "licenses" -Destination $package -Recurse
}

$zip = "$package.zip"
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}
Compress-Archive -LiteralPath $package -DestinationPath $zip -CompressionLevel Optimal
Write-Output $zip
