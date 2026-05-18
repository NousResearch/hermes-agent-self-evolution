param(
    [string]$Skill = "code-review",
    [int]$Iterations = 5,
    [switch]$Execute,
    [string]$OptimizerModel = "openai/gpt-4.1",
    [string]$EvalModel = "openai/gpt-4.1-mini"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not $env:HERMES_AGENT_REPO) {
    throw "HERMES_AGENT_REPO is required. Example: `$env:HERMES_AGENT_REPO='C:\Users\hotep\.pi\agent'"
}
if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is required."
}

if (-not $env:OPENAI_BASE_URL) {
    $env:OPENAI_BASE_URL = "https://api.openai.com/v1"
}
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $repoRoot "output\evolution_runs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir "$Skill`_$stamp.log"

$args = @(
    "-m", "evolution.skills.evolve_skill",
    "--skill", $Skill,
    "--iterations", "$Iterations",
    "--eval-source", "synthetic",
    "--optimizer-model", $OptimizerModel,
    "--eval-model", $EvalModel,
    "--run-tests"
)

if (-not $Execute) {
    $args += "--dry-run"
}

Write-Host "Starting self-evolution run"
Write-Host "repo: $($env:HERMES_AGENT_REPO)"
Write-Host "skill: $Skill"
Write-Host "iterations: $Iterations"
Write-Host "base_url: $($env:OPENAI_BASE_URL)"
Write-Host "mode: $(if ($Execute) { 'execute' } else { 'dry-run' })"
Write-Host ""

& python -X utf8 @args 2>&1 | Tee-Object -FilePath $logPath

Write-Host ""
Write-Host "Log written to: $logPath"
