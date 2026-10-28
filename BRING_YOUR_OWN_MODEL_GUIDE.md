# Bring Your Own Model & Backend Guide (OpenColab + OpenCode)

This guide shows how to run **open-source LLMs** on Google Colab across **5 inference backends** (`ollama`, `llamacpp`, `vllm`, `sglang`, and `transformers`) and tunnel them into **OpenCode v2** with **configurable context** and **agent tool use**.

---

## ⚠️ Crucial Reality Check: 1M Context, VRAM Math & Google Colab ToS

Before choosing a model and context size, please read these 3 hardware and policy constraints:

### 1. Does Your Model Actually Support 1M Context?
OpenColab unlocks the software context parameter (`num_ctx` / `--max-model-len`), **but the model weights themselves must be trained for that context length**:
- **Most standard models (`Qwen 2.5 Coder`, `DeepSeek-R1`, `Llama 3.1`, `Spark-X2.6-4B`) have a hard architectural ceiling of `131,072` (128k) tokens** (or `32,768` for older models).
- Setting `CONTEXT_WINDOW = 1048576` (1M) on a 128k-native model **will not give it 1M comprehension**—attention quality will collapse past 128k, or strict engines like `vllm`/`sglang` will reject startup with a `max_position_embeddings` error.
- **Only use `1048576` (1M) with models explicitly built or RoPE/YaRN-scaled for 1M tokens** (such as `Qwen2.5-7B-Instruct-1M` or `Qwen2.5-14B-Instruct-1M`) on an **80 GB A100 GPU**.

### 2. Why a 32B Model + 1M Context Will Crash a 16 GB GPU
Total GPU VRAM = **`Model Weights + KV Cache Buffer + CUDA Overhead`**:
- **Model Weights Alone**:
  - **4B Q4_K_M**: ~2.6 GB VRAM
  - **7B–9B Q4_K_M**: ~4.8 – 6.0 GB VRAM
  - **14B Q4_K_M**: ~9.0 GB VRAM
  - **27B–32B Q4_K_M**: **~17.5 – 20.0 GB VRAM** *(Already exceeds a 15 GB Tesla T4 before allocating a single token of context!)*
- **KV Cache Buffer (even with 4-bit `q4_0` KV cache enabled)**:
  - **128k tokens on a 4B–8B model**: ~2.5 – 5.5 GB extra VRAM
  - **128k tokens on a 32B model**: ~12 – 18 GB extra VRAM (Needs **35+ GB total VRAM** ➔ requires **A100 40GB/80GB**)
  - **1M (`1,048,576`) tokens on a 7B–14B model**: **~20 – 45 GB extra VRAM** (Requires **Colab Pro+ A100 80GB**)

### 3. Google Colab Terms of Service & Free-Tier Termination
- **Free Accounts Will Terminate**: Google Colab Free Tier is strictly for short, interactive educational notebooks. Running persistent tunnels (`cloudflared`) and heavy LLM servers on a Free account will trigger automated disconnects and **may violate Google Colab's Terms of Service**.
- **Paid Plan Required**: To run this setup legitimately in accordance with Google Colab's policies for heavier/sustained workloads, use a **Paid Google Colab Plan (Colab Pro, Colab Pro+, or Pay-As-You-Go Compute Units)** and keep your notebook session interactive.

---

## 5 Supported Inference Backends

### 1. Ollama Backend (`BACKEND = "ollama"`) — Default
Ollama hosts thousands of pre-quantized models and supports custom Modelfiles with Flash Attention and 4-bit KV cache (`q4_0`).

Popular coding & agent models:
- **`SparkLLM/Spark-X2.6-4B:q4_k_m`** — Ultra-fast lightweight agent model (Fits easily on T4/L4 at 128k)
- **`qwen2.5-coder:7b`** (or `1.5b`, `3b`, `14b`, `32b`) — Exceptional at code generation & editing
- **`deepseek-r1:8b`** (or `1.5b`, `7b`, `14b`, `32b`) — Deep reasoning and chain-of-thought
- **`llama3.1:8b`** — Strong general purpose and agent tool use
- **`mistral:7b`** or **`codestral:22b`** (22B requires L4 24GB or A100)

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

## Realistic Context Windows by Model & GPU VRAM

| Model & Parameter Size | Model's Native Max Context | T4 GPU (15 GB VRAM) | L4 GPU (24 GB VRAM) | A100 GPU (40 GB / 80 GB VRAM) |
|---|---|---|---|---|
| **Spark-X2.6-4B (Q4)** | 128k (`131072`) | `65536` – `131072` (128k) | `131072` (128k) | `131072` (128k) |
| **Qwen 2.5 Coder 7B / Qwen 3.6 9B (Q4)** | 128k (`131072`) | `32768` – `131072` (128k) | `131072` (128k) | `131072` (128k) |
| **Qwen 2.5 7B/14B-Instruct-1M (YaRN 1M)** | **1M (`1048576`)** | `65536` – `131072` *(VRAM limited)* | `131072` – `262144` *(VRAM limited)* | **`524288` – `1048576` (1M on 80GB A100)** |
| **Qwen 2.5 Coder 14B / DeepSeek-R1 14B (Q4)** | 128k (`131072`) | `16384` – `32768` *(Tight VRAM!)* | `65536` – `131072` (128k) | `131072` (128k) |
| **Qwen 3.8 27B / Qwen 2.5 Coder 32B (Q4)** | 128k (`131072`) | ❌ **Will NOT fit (OOM)** | `16384` – `32768` *(27B Q4 only)* | `65536` – `131072` (128k) |
| **DeepSeek-R1 70B / Llama 3.1 70B (Q4)** | 128k (`131072`) | ❌ **Will NOT fit (OOM)** | ❌ **Will NOT fit (OOM)** | `32768` – `131072` *(80GB A100 only)* |

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
