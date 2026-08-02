param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    Set-Location $projectRoot
    & $venvPython -m d2draft.desktop
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    Set-Location $projectRoot
    & $pyLauncher.Source -3 -m d2draft.desktop
    exit $LASTEXITCODE
}
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $python = $pythonCommand.Source
} else {
    throw "Python 3.11+ was not found. Install Python and run: python -m pip install -e ."
}

Set-Location $projectRoot
& $python -m d2draft.desktop
exit $LASTEXITCODE
