# 🚀 OpenColab

> **Run ANY open-source LLM or custom GGUF on Google Colab (Free or Paid Colab Pro / Pro+) using Ollama, llama.cpp, vLLM, SGLang, or HuggingFace Transformers — connected directly to OpenCode v2 with Up to 1M Context, Agent Tools, and Real-Time SSE Streaming.**

---

## 🌟 Highlights

- **5 Pluggable Inference Backends**: Choose the best engine for your workload via `BACKEND`:
  - **`ollama`** *(Default)*: Ollama daemon with Flash Attention (`OLLAMA_FLASH_ATTENTION=1`) & 4-bit KV cache (`q4_0`).
  - **`llamacpp`**: Native `llama.cpp` CUDA server (`llama-cpp-python[server]`) for direct `.gguf` files, URLs, or HuggingFace GGUF repos.
  - **`vllm`**: High-throughput PagedAttention OpenAI server for Safetensors, AWQ, GPTQ, and FP16/BF16 models with automatic tool-call parsing.
  - **`sglang`**: RadixAttention engine for ultra-fast prefix caching and structured agent loops.
  - **`transformers`**: Zero-install native Hugging Face `transformers` + `torch` streaming engine (with optional 4-bit `bitsandbytes` quantization).
- **Bypasses the 4096 Context Ceiling**: Configures custom context limits (from 128k up to **1M / 1,048,576 tokens**) across all supported backends.
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

## ⚡ Using with a Paid Google AI Plan (Google Colab Pro / Pro+)

This setup scales from **Free Colab** up to **Google Colab Pro / Pro+ / Compute Units**:

| Plan | Recommended GPU | Max Model Size | Max Context Window | Advantages |
|------|-----------------|----------------|--------------------|------------|
| **Colab Free** | Tesla T4 (15 GB) | 3B – 9B models (e.g. Spark-X2.6-4B, Qwen 3.6 9B, Qwen 2.5 Coder 7B) | 32k – 128k tokens | Zero cost |
| **Colab Pro / Pay-As-You-Go** | Nvidia L4 (24 GB) or A100 (40 GB) | 14B – 32B models (e.g. Qwen 3.8 27B, Qwen 2.5 Coder 14B/32B, DeepSeek-R1 14B) | 128k – 256k tokens | Faster generation (~60-110 tokens/sec), background execution |
| **Colab Pro+** | Nvidia A100 (80 GB) | Up to 70B models (quantized) | **Up to 1M tokens (1,048,576)** | Massive VRAM headroom for extreme context lengths up to 24h continuous runtimes |

---

## 🚀 Quick Start (In 2 Simple Steps)

### Step 1: Run the Backend on Google Colab

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook.
2. Under **Runtime** -> **Change runtime type**, select a **GPU** (T4, L4, or A100).
3. Copy the code from [`COLAB_SETUP.py`](COLAB_SETUP.py) into a code cell.
4. Set your desired backend, model, and context window at the top of the cell (or via environment variables):
   ```python
   BACKEND = "ollama"               # "ollama", "llamacpp", "vllm", "sglang", "transformers", or "auto"
   MODEL_ID = "SparkLLM/Spark-X2.6-4B:q4_k_m"  # Model tag, HF repo, or .gguf path
   CONTEXT_WINDOW = 131072          # 128k, 256k (262144), or up to 1M (1048576)
   ```
5. Run the cell. Once initialized, it will print your session credentials:
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
