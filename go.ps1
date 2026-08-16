<#
.SYNOPSIS
    Activate the project virtualenv (.venv) in the current PowerShell session.

.DESCRIPTION
    Run  .\go.ps1  in a fresh terminal instead of typing the Scripts\Activate
    path. If the venv is already active, it does nothing. After it runs, `python`
    and `pip` resolve to the project's Python 3.12 venv, so:

        .\go.ps1
        python -m autocut all context

    Because .venv\Scripts\Activate.ps1 installs its prompt and `deactivate` into
    the global scope and sets process-level env vars, plain  .\go.ps1  activates
    the session fully (you don't need to dot-source it).
#>

$ErrorActionPreference = 'Stop'

$venvDir  = Join-Path $PSScriptRoot '.venv'
$activate = Join-Path $venvDir 'Scripts\Activate.ps1'

# Already active for THIS venv? Nothing to do (avoids stacking activations).
if ($env:VIRTUAL_ENV) {
    try { $current = (Resolve-Path -LiteralPath $env:VIRTUAL_ENV).Path } catch { $current = $env:VIRTUAL_ENV }
    try { $target  = (Resolve-Path -LiteralPath $venvDir).Path }        catch { $target  = $venvDir }
    if ($current -eq $target) {
        Write-Host "venv already active: $env:VIRTUAL_ENV" -ForegroundColor DarkGray
        return
    }
    Write-Host "switching from active venv: $env:VIRTUAL_ENV" -ForegroundColor DarkYellow
}

if (-not (Test-Path -LiteralPath $activate)) {
    Write-Error "No virtualenv at $venvDir. Create it first:  py -3.12 -m venv .venv; .\.venv\Scripts\python -m pip install -e '.[cuda,dev]'"
    return
}

# Dot-source so Activate.ps1's global:/env changes land in this session.
. $activate

Write-Host "activated $env:VIRTUAL_ENV" -ForegroundColor Green
Write-Host "  $(python --version 2>&1)  ->  $((Get-Command python).Source)" -ForegroundColor DarkGray
