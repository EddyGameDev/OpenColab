# Bring Your Own Ollama Model Guide (Colab + OpenCode)

This guide shows how to run **any model** you want on Google Colab and tunnel it directly into **OpenCode v2** with **full context (up to 128k)** and **tool use**.

---

## 3 Ways to Bring Your Own Model

### Method 1: Any Model from the Ollama Library (Recommended)
Ollama hosts thousands of pre-quantized models. You just need the model name.

Popular coding & agent models:
- **`qwen2.5-coder:7b`** (or `1.5b`, `3b`, `14b`, `32b`) — Exceptional at code generation & editing
- **`deepseek-r1:8b`** (or `1.5b`, `7b`, `14b`) — Deep reasoning and chain-of-thought
- **`llama3.1:8b`** — Strong general purpose and agent tool use
- **`mistral:7b`** or **`codestral:22b`** (on A100 runtime)
- **`starcoder2:7b`**
- **`SparkLLM/Spark-X2.6-4B:q4_k_m`**

#### How to use:
In [COLAB_SETUP.py](file:///F:/MyProject/OpenColab/COLAB_SETUP.py), set:
```python
MODEL_ID = "qwen2.5-coder:7b"
CONTEXT_WINDOW = 131072  # or 32768, 65536
```
Run the cell in Colab.

---

### Method 2: Any Custom GGUF from Hugging Face
If you have a fine-tuned model or a GGUF file from Hugging Face:

1. Download the `.gguf` file to Colab:
```python
!wget -O /content/my_model.gguf "https://huggingface.co/author/repo/resolve/main/model-q4_k_m.gguf"
```
2. In [COLAB_SETUP.py](file:///F:/MyProject/OpenColab/COLAB_SETUP.py), set:
```python
MODEL_ID = "/content/my_model.gguf"
CONTEXT_WINDOW = 131072
```
The script will automatically detect the `.gguf` file, create an Ollama model from it, and configure your chosen context window.

---

### Method 3: Any GGUF from Google Drive
1. Mount Google Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```
2. In [COLAB_SETUP.py](file:///F:/MyProject/OpenColab/COLAB_SETUP.py), set:
```python
MODEL_ID = "/content/drive/MyDrive/models/my_custom_model.gguf"
CONTEXT_WINDOW = 65536
```

---

## Recommended Context Windows by Model & GPU

| Model | Native Context | Free T4 (15 GB) | Colab Pro A100 (40/80 GB) |
|-------|----------------|-----------------|---------------------------|
| **Qwen 2.5 Coder (7B/14B)** | 128k (up to 1M via YaRN) | `131072` (128k) | **`1048576` (Up to 1M)** |
| **DeepSeek-R1 (7B/8B/14B)** | 128k | `131072` (128k) | `131072` (128k) |
| **Llama 3.1 (8B)** | 128k | `131072` (128k) | `131072` (128k) |
| **Spark-X2.6-4B** | 128k | `131072` (128k) | `131072` (128k) |
| **Mistral / Codestral** | 32k – 256k | `32768` | `262144` (256k) |
| **Older Llama 3 (8B)** | 8k | `8192` | `8192` |

> [!TIP]
> Ollama runs with 4-bit KV Cache Compression (`OLLAMA_KV_CACHE_TYPE="q4_0"`) and Flash Attention (`OLLAMA_FLASH_ATTENTION="1"`). This reduces VRAM consumption for large context by over 70%, allowing 128k on a free Colab Tesla T4 (15 GB VRAM) and up to **1M tokens (1,048,576)** on an A100 GPU!

---

## Connecting to OpenCode (Windows)

Whenever Colab outputs a new tunnel URL, run in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File "F:\MyProject\OpenColab\Apply-Colab-Tunnel.ps1" -TunnelUrl "https://xyz-abc.trycloudflare.com" -ApiKey "<YOUR_GENERATED_API_KEY>" -ModelId "qwen2.5-coder:7b" -ModelName "Qwen 2.5 Coder 7B" -ContextWindow 131072
```

This single command updates all OpenCode configs, sets the limits to 128k, and reloads OpenCode automatically!
