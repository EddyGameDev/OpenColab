param(
    [Parameter(Mandatory=$true, HelpMessage="The Cloudflare tunnel URL, e.g. https://abc-xyz.trycloudflare.com")]
    [string]$TunnelUrl,

    [Parameter(Mandatory=$true, HelpMessage="The Bearer API key printed by your Colab session")]
    [string]$ApiKey,

    [string]$ModelId = "SparkLLM/Spark-X2.6-4B:q4_k_m",

    [string]$ModelName = "Colab Model",

    [ValidateSet("ollama", "llamacpp", "vllm", "sglang", "transformers", "auto")]
    [string]$Backend = "ollama",

    [int]$ContextWindow = 131072
)

$ErrorActionPreference = "Stop"

# Clean up URL format
$cleanUrl = $TunnelUrl.Trim().TrimEnd('/')
if ($cleanUrl -match '/v1$') {
    $cleanUrl = $cleanUrl.Substring(0, $cleanUrl.Length - 3)
}
$baseUrl = "$cleanUrl/v1"

Write-Host "Configuring OpenCode for OpenColab Multi-Backend Model..." -ForegroundColor Cyan
Write-Host "Inference Backend : $Backend" -ForegroundColor Gray
Write-Host "Base URL          : $baseUrl" -ForegroundColor Gray
Write-Host "Model ID          : $ModelId" -ForegroundColor Gray
Write-Host "Context Window    : $ContextWindow tokens" -ForegroundColor Gray
Write-Host "API Key           : $ApiKey" -ForegroundColor Gray

# Configuration JSON template
$configTemplate = @"
{
  "`$schema": "https://opencode.ai/config.json",
  "model": "colab/model",
  "providers": {
    "colab": {
      "name": "Colab Provider ($Backend - $ModelName)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "$baseUrl",
        "apiKey": "$ApiKey",
        "headerTimeout": 120000,
        "chunkTimeout": 300000
      },
      "compatibility": {
        "requireFinishReason": false
      },
      "models": {
        "model": {
          "modelID": "$ModelId",
          "name": "$ModelName ($Backend)",
          "compatibility": {
            "requireFinishReason": false
          },
          "capabilities": {
            "tools": true,
            "input": [
              "text"
            ],
            "output": [
              "text"
            ]
          },
          "limit": {
            "context": $ContextWindow,
            "output": $ContextWindow
          }
        }
      }
    }
  }
}
"@

# 1. Update Global Config (~/.config/opencode/opencode.jsonc)
$globalConfigDir = "$env:USERPROFILE\.config\opencode"
if (-not (Test-Path $globalConfigDir)) {
    New-Item -ItemType Directory -Path $globalConfigDir -Force | Out-Null
}
$globalConfigFile = Join-Path $globalConfigDir "opencode.jsonc"
Set-Content -Path $globalConfigFile -Value $configTemplate -Encoding utf8
Write-Host "[OK] Updated global config: $globalConfigFile" -ForegroundColor Green

# 2. Update Current Project Config (opencode.jsonc)
$projectConfigFile = Join-Path (Get-Location).Path "opencode.jsonc"
Set-Content -Path $projectConfigFile -Value $configTemplate -Encoding utf8
Write-Host "[OK] Updated project config: $projectConfigFile" -ForegroundColor Green

# 3. Reload OpenCode service
Write-Host "Reloading OpenCode service..." -ForegroundColor Cyan
try {
    & opencode reload
    Write-Host "[OK] OpenCode service successfully reloaded!" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Start or restart OpenCode to load new config." -ForegroundColor Yellow
}

# 4. Quick Connectivity Check
Write-Host "Checking tunnel connectivity..." -ForegroundColor Cyan
try {
    $models = Invoke-RestMethod -Uri "$baseUrl/models" -Headers @{ Authorization = "Bearer $ApiKey" } -TimeoutSec 15
    $activeBackend = $models.data[0].owned_by
    Write-Host "[SUCCESS] Live tunnel connection verified! Active Model: $($models.data[0].id) (Backend: $activeBackend)" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Could not reach $baseUrl/models yet. DNS might still be propagating (takes 5-10s)." -ForegroundColor Yellow
}

Write-Host "`nAll set! OpenCode model 'colab/model' ($Backend) is ready with $ContextWindow context." -ForegroundColor Cyan
