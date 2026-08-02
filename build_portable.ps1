param(
    [string]$Python = "python",
    [string]$Version = "0.4.1"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Join-Path $root "dist"
$name = "Dota2BPHelper-$Version-win64"

Set-Location $root

# Ship only the versioned files under data/. The working tree also holds the
# training database, raw API responses, and machine-specific calibration, which
# .gitignore excludes and which must never reach a release.
$payload = Join-Path $root "build\data-payload"
if (Test-Path -LiteralPath $payload) {
    Remove-Item -LiteralPath $payload -Recurse -Force
}
$tracked = & git -C $root ls-files data
if ($LASTEXITCODE -ne 0 -or -not $tracked) {
    throw "could not list the versioned files under data/; refusing to package the whole directory"
}
foreach ($relative in $tracked) {
    $source = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
    $destination = Join-Path $payload $relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}
if (-not (Test-Path -LiteralPath (Join-Path $payload "data\heroes.json"))) {
    throw "data/heroes.json is missing from the staged payload"
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name "Dota2BPHelper" `
    --add-data "$payload\data;data" `
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

# Last line of defence before anything is published. Matching on the file name
# keeps this independent of -Include's path-dependent behaviour.
$forbidden = '(^\.env$|\.env\.|\.sqlite3(-|$)|^raw_details|\.jsonl\.gz$|^screen_config\.json$)'
$leaks = Get-ChildItem -LiteralPath $package -Recurse -File | Where-Object { $_.Name -match $forbidden }
if ($leaks) {
    $names = ($leaks | ForEach-Object { $_.FullName }) -join "`n  "
    throw "refusing to package private files:`n  $names"
}

$zip = "$package.zip"
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}
Compress-Archive -LiteralPath $package -DestinationPath $zip -CompressionLevel Optimal
Remove-Item -LiteralPath $payload -Recurse -Force
Write-Output $zip
