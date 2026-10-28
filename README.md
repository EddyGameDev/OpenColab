# 🚀 OpenColab

> **Run open-source LLMs or custom GGUFs on Google Colab using Ollama, llama.cpp, vLLM, SGLang, or HuggingFace Transformers — connected to OpenCode v2 with Configurable Context (32k, 128k, and up to 1M on supported models & hardware), Agent Tools, and Real-Time SSE Streaming.**

---

## ⚠️ Important: Read Before Running (1M Context, VRAM Limits & Google Colab ToS)

Before configuring high context lengths or large models, please understand how **model architecture limits**, **GPU VRAM math**, and **Google Colab's Terms of Service** work in practice:

### 1. "Up to 1M Context" Requires a Model That Actually Supports 1M Tokens
- OpenColab removes Ollama's default **4,096-token software cap** by baking `num_ctx` (or `--max-model-len`) into the backend configuration.
- **However, OpenColab cannot magically give 1M context to a model trained only for 32k or 128k.**
  - Most current open-weight models (`Qwen 2.5 Coder`, `DeepSeek-R1`, `Llama 3.1`, `Spark-X2.6-4B`) have a **native ceiling of 128k (`131,072`) tokens** (or 32k–64k for older models).
  - Setting `CONTEXT_WINDOW = 1048576` (1M) only works properly on models explicitly built or extended via YaRN/RoPE for 1M context (such as `Qwen2.5-7B-Instruct-1M` or `Qwen2.5-14B-Instruct-1M`). Forcing 1M context on a 128k-native model will result in garbled output beyond 128k or backend validation errors.

### 2. GPU VRAM Math: You Cannot Fit a 32B Model + 1M Context on 16 GB VRAM
Total GPU memory required is **`Model Weights + KV Cache + Activation Overhead`**:
- **A 32B model (even quantized to 4-bit Q4_K_M) requires ~19–20 GB of VRAM just for the weights alone.** It **will not fit** on a 15 GB Tesla T4 GPU—even at a tiny 4k context! Attempting to load a 32B model on a 15 GB GPU will either crash with `CUDA Out of Memory` or offload layers to slow CPU RAM (<1 token/sec).
- **1M Token KV Cache requires massive VRAM**: Even with 4-bit KV cache (`q4_0`) and Flash Attention, a 1,048,576-token context buffer consumes **15 GB to 40+ GB of extra VRAM** depending on model size.
- **Rule of Thumb**:
  - **15 GB VRAM (Tesla T4)**: Stick to **4B – 9B models** with **32k – 128k context** (or a 14B Q4 model with **16k – 32k context**).
  - **24 GB VRAM (Nvidia L4)**: Run **14B – 27B (Q4) models** with **64k – 128k context**.
  - **40 GB – 80 GB VRAM (Nvidia A100)**: Required for **27B – 70B models** at **128k – 256k context**, or **7B – 14B 1M-native models** at **1,048,576 context**.

### 3. Google Colab Terms of Service (ToS) & Free Tier Disconnections
- **Paid Plan Strongly Required**: Running background tunnels (`cloudflared`), API servers, or sustained heavy LLM inference on a **Google Colab Free account** will result in **frequent, unannounced runtime terminations** and may violate [Google Colab's Terms of Service](https://research.google.com/colaboratory/faq.html) regarding free-tier resource usage and remote proxy hosting.
- **Use Paid Compute Units (Colab Pro / Pro+ / Pay-As-You-Go)**: This project is designed for developers utilizing **paid Google Colab compute units (Colab Pro / Pro+)** who want to use their purchased GPU allocation interactively alongside their IDE. Always keep your Colab notebook tab open and abide by Google's Terms of Service.

---

## 🌟 Highlights

- **5 Pluggable Inference Backends**: Choose the best engine for your workload via `BACKEND`:
  - **`ollama`** *(Default)*: Ollama daemon with Flash Attention (`OLLAMA_FLASH_ATTENTION=1`), 4-bit KV cache (`q4_0`), and automatic GPU VRAM pre-warming (`OLLAMA_KEEP_ALIVE=-1`).
  - **`llamacpp`**: Native `llama.cpp` CUDA server (`llama-cpp-python[server]`) for direct `.gguf` files, URLs, or HuggingFace GGUF repos.
  - **`vllm`**: High-throughput PagedAttention OpenAI server for Safetensors, AWQ, GPTQ, and FP16/BF16 models with automatic tool-call parsing.
  - **`sglang`**: RadixAttention engine for ultra-fast prefix caching and structured agent loops.
  - **`transformers`**: Zero-install native Hugging Face `transformers` + `torch` streaming engine (with optional 4-bit `bitsandbytes` quantization).
- **Bypasses the 4096 Default Cap**: Configures custom context limits (`32k`, `128k`, `256k`, or up to `1M` on supported 1M models + A100 GPUs).
- **Zero Cloudflare 530 Errors**: Uses a unified FastAPI proxy server (`:8000`) that verifies upstream backend readiness prior to exposing the Cloudflare tunnel.
- **Universal Reasoning & SSE Normalization**: Preserves EventStream framing (`\n\n`), normalizes both `reasoning` (Ollama) and `reasoning_content` (vLLM / SGLang / llama.cpp) deltas, and strips backend-incompatible payload fields automatically.
- **Zero Hardcoded Secrets**: Uses cryptographically secure random bearer tokens generated fresh per runtime session.

---

## 🔌 Supported Inference Backends

| `BACKEND` Value | Engine | Best For | Example `MODEL_ID` |
|---|---|---|---|
| `"ollama"` *(default)* | **Ollama** | Pre-quantized registry models & local GGUFs | `"SparkLLM/Spark-X2.6-4B:q4_k_m"`, `"qwen2.5-coder:7b"`, `"deepseek-r1:8b"` |
| `"llamacpp"` | **llama.cpp Server** | Direct HuggingFace or local `.gguf` files with full GPU offload | `"Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf"` or `"/content/model.gguf"` |
| `"vllm"` | **vLLM** | High-speed Safetensors / AWQ / GPTQ models & parallel throughput | `"Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"`, `"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"` |
| `"sglang"` | **SGLang** | Fast RadixAttention prompt caching & multi-turn agent loops | `"Qwen/Qwen2.5-Coder-7B-Instruct"` |
| `"transformers"` | **HF Transformers** | Instant startup using Colab's pre-installed PyTorch + optional 4-bit BnB | `"Qwen/Qwen2.5-Coder-3B-Instruct"` |
| `"auto"` | **Auto-Detect** | Automatically selects `llamacpp` for `.gguf`, `vllm` for `org/repo`, or `ollama` | Any of the above |

---

## ⚡ Realistic Hardware & Colab Plan Sizing Guide

| Colab Plan | GPU & VRAM | Realistic Model Size (Q4) | Realistic Context Window | Notes & ToS |
|---|---|---|---|---|
| **Colab Free** *(Testing Only)* | Tesla T4 (**15 GB**) | **3B – 8B** (`Spark-X2.6-4B`, `Qwen 2.5 7B`) | **32k – 128k** tokens | **Frequent disconnects.** Do not use for persistent tunnels; 14B+ or >128k will OOM. |
| **Colab Pro / Pay-As-You-Go** *(Recommended)* | Nvidia L4 (**24 GB**) or A100 (**40 GB**) | **7B – 27B/32B** (`Qwen 3.8 27B`, `Qwen 2.5 14B/32B`, `DeepSeek-R1 14B`) | **64k – 128k** (on 27B/32B)<br>**128k – 256k** (on 7B–14B) | Paid compute units required for sustained interactive IDE sessions. |
| **Colab Pro+** *(Heavy Workloads)* | Nvidia A100 (**80 GB**) | **32B – 70B** (at 64k–128k)<br>**7B – 14B 1M-Models** (at 1M) | **Up to 1M (`1,048,576`)** *(Only on 1M-capable models like Qwen2.5-1M)* | High VRAM headroom + priority background execution. |

---

## 🚀 Quick Start (In 2 Simple Steps)

### Step 1: Run the Backend on Google Colab

1. Open [Google Colab](https://colab.research.google.com) (using a **Paid Colab Pro / Pro+** account for sustained usage) and create a new notebook.
2. Under **Runtime** -> **Change runtime type**, select a **GPU** (T4, L4, or A100).
3. Copy the code from [`COLAB_SETUP.py`](COLAB_SETUP.py) into a code cell.
4. Set your desired backend, model, and context window at the top of the cell (matching your GPU's VRAM and the model's native context limit):
   ```python
   BACKEND = "ollama"                          # "ollama", "llamacpp", "vllm", "sglang", "transformers", or "auto"
   MODEL_ID = "SparkLLM/Spark-X2.6-4B:q4_k_m"  # Model tag, HF repo, or .gguf path
   CONTEXT_WINDOW = 131072                     # 32768 (32k), 65536 (64k), or 131072 (128k)
   ```
5. Run the cell. Once initialized and pre-warmed in GPU VRAM, it will print your session credentials:
   ```text
   =================================================================
                OPENCOLAB BACKEND READY (MULTI-BACKEND)
   =================================================================
   Inference Backend : ollama
   Active Model      : SparkLLM/Spark-X2.6-4B:q4_k_m
   Context Window    : 131072 tokens
   Tunnel Public URL : https://<random-subdomain>.trycloudflare.com
   Base URL          : https://<random-subdomain>.trycloudflare.com/v1
   API Bearer Key    : <your-secure-random-key>
   =================================================================
   ```

---

### Step 2: Configure OpenCode on Windows (1 Command)

Open **PowerShell** and run `Apply-Colab-Tunnel.ps1` with your printed URL, Key, and Backend:

```powershell
powershell -ExecutionPolicy Bypass -File ".\Apply-Colab-Tunnel.ps1" `
  -TunnelUrl "https://<random-subdomain>.trycloudflare.com" `
  -ApiKey "<your-secure-random-key>" `
  -ModelId "SparkLLM/Spark-X2.6-4B:q4_k_m" `
  -ModelName "Spark-X2.6 4B" `
  -Backend "ollama" `
  -ContextWindow 131072
```

This single command:
- Automatically configures OpenCode's global (`~/.config/opencode/opencode.jsonc`) and project settings.
- Sets your custom context and output limits with tool calling enabled.
- Verifies live tunnel connectivity and reloads the OpenCode service instantly.

---

### Step 3: Launch OpenCode

In your terminal:
```bash
opencode
```
Select model **`colab/model`** and start coding!

---

## 🧠 Bring Your Own Model & Backend Options

Check out [`BRING_YOUR_OWN_MODEL_GUIDE.md`](BRING_YOUR_OWN_MODEL_GUIDE.md) for full instructions on:
1. **Ollama Registry Models** (`BACKEND="ollama"`): `SparkLLM/Spark-X2.6-4B:q4_k_m`, `qwen2.5-coder`, `deepseek-r1`, `llama3.1`, etc.
2. **Native GGUF Server** (`BACKEND="llamacpp"`): Direct Hugging Face `owner/repo:file.gguf`, HTTPS URLs, or Google Drive `.gguf` files.
3. **vLLM & SGLang High-Throughput Engines** (`BACKEND="vllm"` / `BACKEND="sglang"`): Full Safetensors, AWQ, and GPTQ models from Hugging Face.
4. **Native Hugging Face Transformers** (`BACKEND="transformers"`): Zero-install PyTorch + `TextIteratorStreamer` with optional 4-bit `bitsandbytes` quantization (`LOAD_IN_4BIT=1`).

---

## 🔒 Security

- Cloudflare Quick Tunnels generate temporary HTTPS URLs.
- The FastAPI proxy rejects any request that lacks the generated `Authorization: Bearer <KEY>` header.
- The internal backend port (`11434`) is bound strictly to `127.0.0.1` and never exposed directly to the public Internet.
- No personal API keys or tunnel URLs are ever stored in version control.

---

## 📄 License

This project is licensed under the **PolyForm Noncommercial License 1.0.0 (Source-Available)**. See [`LICENSE`](LICENSE) and the [Source-Available Guide](SOURCE_AVAILABLE_GUIDE.md) for details.
Free for personal use, research, and non-commercial development.
