$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

if (-not $env:DASHSCOPE_API_KEY -and -not $env:LLM_API_KEY) {
  Write-Host "请先设置通义 API Key：" -ForegroundColor Yellow
  Write-Host '$env:DASHSCOPE_API_KEY="你的通义API Key"'
  exit 1
}

python .\run.py
