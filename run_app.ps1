$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Error 'Virtual environment not found. Create it and install requirements first.'
}

& $python (Join-Path $PSScriptRoot 'app.py')
