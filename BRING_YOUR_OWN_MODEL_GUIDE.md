# Bring Your Own Model & Backend Guide (OpenColab + OpenCode)

This guide shows how to run **any open-source model** on Google Colab across **5 inference backends** (`ollama`, `llamacpp`, `vllm`, `sglang`, and `transformers`) and tunnel it directly into **OpenCode v2** with **up to 1M context** and **full agent tool use**.

---

## 5 Supported Inference Backends

### 1. Ollama Backend (`BACKEND = "ollama"`) — Default
Ollama hosts thousands of pre-quantized models and supports custom Modelfiles with Flash Attention and 4-bit KV cache (`q4_0`).

Popular coding & agent models:
- **`SparkLLM/Spark-X2.6-4B:q4_k_m`** — Ultra-fast lightweight agent model
- **`qwen2.5-coder:7b`** (or `1.5b`, `3b`, `14b`, `32b`) — Exceptional at code generation & editing
- **`deepseek-r1:8b`** (or `1.5b`, `7b`, `14b`, `32b`) — Deep reasoning and chain-of-thought
- **`llama3.1:8b`** — Strong general purpose and agent tool use
- **`mistral:7b`** or **`codestral:22b`** (on L4/A100 runtime)

```python
BACKEND = "ollama"
MODEL_ID = "SparkLLM/Spark-X2.6-4B:q4_k_m"
CONTEXT_WINDOW = 131072
```

---

### 2. Native llama.cpp Server (`BACKEND = "llamacpp"`)
Runs `llama-cpp-python[server]` with full CUDA GPU offload (`--n_gpu_layers -1`) and Flash Attention directly on any `.gguf` file without needing Ollama wrapper manifests.

Supports **3 ways** to specify `MODEL_ID`:
1. **Hugging Face `repo_id:filename.gguf`** (auto-downloads via `huggingface_hub`):
   ```python
   BACKEND = "llamacpp"
   MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf"
   CONTEXT_WINDOW = 131072
   ```
2. **Direct HTTPS URL to a `.gguf` file**:
   ```python
   BACKEND = "llamacpp"
   MODEL_ID = "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
   CONTEXT_WINDOW = 131072
   ```
3. **Local or Google Drive `.gguf` file path**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   BACKEND = "llamacpp"
   MODEL_ID = "/content/drive/MyDrive/models/my_custom_model.gguf"
   CONTEXT_WINDOW = 65536
   ```

---

### 3. vLLM PagedAttention Server (`BACKEND = "vllm"`)
Ideal for running Hugging Face Safetensors, AWQ, or GPTQ models at maximum throughput with PagedAttention and native tool-call parsing (`--enable-auto-tool-choice`).

```python
BACKEND = "vllm"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
CONTEXT_WINDOW = 65536
GPU_MEM_UTIL = 0.90
VLLM_TOOL_PARSER = "hermes"  # or "llama3_json", "mistral"
```

---

### 4. SGLang RadixAttention Server (`BACKEND = "sglang"`)
Ideal for multi-turn agent workflows in OpenCode where prefix caching (RadixAttention) dramatically accelerates repeated repository context prompts.

```python
BACKEND = "sglang"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
CONTEXT_WINDOW = 65536
GPU_MEM_UTIL = 0.88
```

---

### 5. Hugging Face Transformers + PyTorch (`BACKEND = "transformers"`)
Uses Google Colab's pre-installed `torch`, `transformers`, and `accelerate` libraries with `TextIteratorStreamer` for zero-compile startup. Optionally enable 4-bit `bitsandbytes` quantization via `LOAD_IN_4BIT=1`.

```python
import os
os.environ["LOAD_IN_4BIT"] = "1"  # Optional: 4-bit quantization for larger models on T4

BACKEND = "transformers"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
CONTEXT_WINDOW = 32768
```

---

## Recommended Context Windows by Model & GPU

| Model / Engine | Native Context | Free T4 (15 GB) | Colab Pro A100 (40/80 GB) |
|----------------|----------------|-----------------|---------------------------|
| **Spark-X2.6-4B (`ollama` / `llamacpp`)** | 128k | `131072` (128k) | `131072` (128k) |
| **Qwen 3.6 9B / Qwen 2.5 Coder 7B** | 128k (up to 1M) | `131072` (128k) | **`1048576` (Up to 1M)** |
| **Qwen 3.8 27B / Qwen 2.5 Coder 32B** | 128k – 256k | N/A (Use L4/A100) | `131072` – `262144` |
| **DeepSeek-R1 (7B/8B/14B)** | 128k | `131072` (128k) | `131072` (128k) |
| **vLLM / SGLang AWQ Models (7B–14B)** | 32k – 128k | `32768` – `65536` | `131072` (128k) |

---

## Connecting to OpenCode (Windows)

Whenever Colab outputs a new tunnel URL, run in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File ".\Apply-Colab-Tunnel.ps1" `
  -TunnelUrl "https://xyz-abc.trycloudflare.com" `
  -ApiKey "<YOUR_GENERATED_API_KEY>" `
  -ModelId "SparkLLM/Spark-X2.6-4B:q4_k_m" `
  -ModelName "Spark-X2.6 4B" `
  -Backend "ollama" `
  -ContextWindow 131072
```

This single command updates all OpenCode configs, sets the limits, verifies `/v1/models`, and reloads OpenCode automatically!
