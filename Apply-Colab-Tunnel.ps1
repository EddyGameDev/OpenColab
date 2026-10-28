param(
    [Parameter(Mandatory=$true, HelpMessage="The Cloudflare tunnel URL, e.g. https://abc-xyz.trycloudflare.com")]
    [string]$TunnelUrl,

    [Parameter(Mandatory=$true, HelpMessage="The Bearer API key printed by your Colab session")]
    [string]$ApiKey,

    [string]$ModelId = "qwen2.5-coder:7b",

    [string]$ModelName = "Colab Model",

    [int]$ContextWindow = 131072
)

$ErrorActionPreference = "Stop"

# Clean up URL format
$cleanUrl = $TunnelUrl.Trim().TrimEnd('/')
if ($cleanUrl -match '/v1$') {
    $cleanUrl = $cleanUrl.Substring(0, $cleanUrl.Length - 3)
}
$baseUrl = "$cleanUrl/v1"

Write-Host "Configuring OpenCode for Colab Hosted Model..." -ForegroundColor Cyan
Write-Host "Base URL       : $baseUrl" -ForegroundColor Gray
Write-Host "Model ID       : $ModelId" -ForegroundColor Gray
Write-Host "Context Window : $ContextWindow tokens" -ForegroundColor Gray
Write-Host "API Key        : $ApiKey" -ForegroundColor Gray

# Configuration JSON template
$configTemplate = @"
{
  "`$schema": "https://opencode.ai/config.json",
  "model": "colab/model",
  "providers": {
    "colab": {
      "name": "Colab Provider ($ModelName)",
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
          "name": "$ModelName",
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
  },
  "mcp": {
    "servers": {
      "colab-mcp": {
        "command": "uv",
        "args": [
          "run",
          "colab-mcp"
        ],
        "cwd": "C:/Users/dariu/Downloads/Spark_X2_6_Colab_OpenCode_v2 (1)/colab-mcp",
        "timeout": 30000
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

# 2. Update Project Config (F:\MyProject\opencode.jsonc)
$projectConfigFile = "F:\MyProject\opencode.jsonc"
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
    Write-Host "[SUCCESS] Live tunnel connection verified! Available model: $($models.data[0].id)" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Could not reach $baseUrl/models yet. DNS might still be propagating (takes 5-10s)." -ForegroundColor Yellow
}

Write-Host "`nAll set! OpenCode model 'colab/model' is ready with $ContextWindow context." -ForegroundColor Cyan
