# Step-by-Step Guide: Running Spark-X2.6-4B in OpenCode via Colab

## 1. Quick Reconnect Procedure (Whenever Colab Restarts)

When your Google Colab runtime restarts or times out, follow these 3 simple steps:

### Step 1: Run the Backend Script in Colab
Paste and run `COLAB_SETUP.py` into a code cell on Google Colab (with GPU enabled).
Wait ~1 minute for it to print the completion block:
```text
============================================================
             COLAB BACKEND READY
============================================================
Tunnel Public URL : https://<random-subdomain>.trycloudflare.com
Base URL          : https://<random-subdomain>.trycloudflare.com/v1
API Bearer Key    : <YOUR_GENERATED_API_KEY>
Model ID          : <YOUR_MODEL_ID>
============================================================
```

### Step 2: Update the Tunnel URL on Windows
Open `C:\Users\dariu\.config\opencode\opencode.jsonc` (and optionally `F:\MyProject\opencode.jsonc`) and update the `baseURL` property with the new live URL:
```json
"baseURL": "https://<new-subdomain>.trycloudflare.com/v1"
```

### Step 3: Reload OpenCode
In PowerShell, run:
```powershell
opencode reload
```
Or if you are opening OpenCode afresh:
```powershell
opencode
```
Select the model `colab/spark`.

---

## 2. Configuration File Locations

OpenCode checks two configuration locations on Windows:
1. **Global Configuration**: `C:\Users\dariu\.config\opencode\opencode.jsonc`
   * Controls default models, providers, and MCP servers across all folders.
2. **Project Configuration**: `F:\MyProject\opencode.jsonc`
   * Project-specific overrides when opening `F:\MyProject`.

Both have been synchronized with the live configuration.

---

## 3. Verifying Everything Works

### Verify Model List
Run in PowerShell:
```powershell
opencode models
```
`colab/spark` should appear at the top of the list.

### Test External Endpoint Directly
Run in PowerShell:
```powershell
Invoke-RestMethod -Uri "https://<YOUR-SUBDOMAIN>.trycloudflare.com/v1/models" -Headers @{Authorization="Bearer <YOUR_API_KEY>"}
```
Should return the model object for `SparkLLM/Spark-X2.6-4B:q4_k_m`.
