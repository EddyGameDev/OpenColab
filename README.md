# 🚀 OpenColab

> **Run ANY open-source LLM or custom GGUF on Google Colab (Free or Paid Colab Pro / Pro+) and connect it directly to OpenCode v2 with Up to 1M Context, Agent Tools, and Real-Time SSE Streaming.**

---

## 🌟 Highlights

- **Bypasses the 4096 Context Ceiling**: Compiles custom context limits (from 128k up to **1M / 1,048,576 tokens**) directly into Ollama's model manifest.
- **Zero Cloudflare 530 Errors**: Uses a background FastAPI proxy server that verifies origin health prior to exposing the tunnel.
- **Universal Model Support**: Plug in any model from the [Ollama Library](https://ollama.com/library) (Qwen 2.5 Coder, DeepSeek-R1, Llama 3.1, Mistral, Starcoder) or load custom Hugging Face GGUFs.
- **Native SSE Streaming**: Preserves proper EventStream framing (`\n\n`) and maps reasoning deltas for smooth OpenCode real-time streaming.
- **Memory Efficient**: Leverages 4-bit KV Cache Compression (`q4_0`) and Flash Attention so large context buffers run comfortably without OOM.
- **Zero Hardcoded Secrets**: Uses cryptographically secure random bearer tokens generated fresh per runtime session.

---

## ⚡ Using with a Paid Google AI Plan (Google Colab Pro / Pro+)

This setup scales from **Free Colab** up to **Google Colab Pro / Pro+ / Compute Units**:

| Plan | Recommended GPU | Max Model Size | Max Context Window | Advantages |
|------|-----------------|----------------|--------------------|------------|
| **Colab Free** | Tesla T4 (15 GB) | 3B – 8B models (e.g. Qwen 2.5 Coder 7B, Llama 3.1 8B) | 32k – 128k tokens | Zero cost |
| **Colab Pro / Pay-As-You-Go** | Nvidia L4 (24 GB) or A100 (40 GB) | 7B – 32B models (e.g. Qwen 2.5 Coder 14B/32B, DeepSeek-R1 14B) | 128k – 256k tokens | Faster generation (~60-90 tokens/sec), background execution |
| **Colab Pro+** | Nvidia A100 (80 GB) | Up to 70B models (quantized) | **Up to 1M tokens (1,048,576)** | Massive VRAM headroom for extreme context lengths up to 24h continuous runtimes |

---

## 🚀 Quick Start (In 2 Simple Steps)

### Step 1: Run the Backend on Google Colab

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook.
2. Under **Runtime** -> **Change runtime type**, select a **GPU** (T4, L4, or A100).
3. Copy the code from [`COLAB_SETUP.py`](COLAB_SETUP.py) into a code cell.
4. Set your desired model and context window at the top of the cell:
   ```python
   MODEL_ID = "qwen2.5-coder:7b"     # Any Ollama model tag or .gguf path
   CONTEXT_WINDOW = 131072          # 128k, 256k (262144), or up to 1M (1048576)
   ```
5. Run the cell. In ~1 minute, it will print your session credentials:
   ```text
   ============================================================
                COLAB BACKEND READY (ANY MODEL)
   ============================================================
   Active Model      : qwen2.5-coder:7b
   Context Window    : 131072 tokens
   Tunnel Public URL : https://<random-subdomain>.trycloudflare.com
   Base URL          : https://<random-subdomain>.trycloudflare.com/v1
   API Bearer Key    : <your-secure-random-key>
   ============================================================
   ```

---

### Step 2: Configure OpenCode on Windows (1 Command)

Open **PowerShell** and run `Apply-Colab-Tunnel.ps1` with your printed URL and Key:

```powershell
powershell -ExecutionPolicy Bypass -File "Apply-Colab-Tunnel.ps1" `
  -TunnelUrl "https://<random-subdomain>.trycloudflare.com" `
  -ApiKey "<your-secure-random-key>" `
  -ModelId "qwen2.5-coder:7b" `
  -ModelName "Qwen 2.5 Coder 7B" `
  -ContextWindow 131072
```

This single command:
- Automatically configures OpenCode's global (`~/.config/opencode/opencode.jsonc`) and project settings.
- Sets your custom context and output limits with tool calling enabled.
- Reloads the OpenCode service instantly.

---

### Step 3: Launch OpenCode

In your terminal:
```bash
opencode
```
Select model **`colab/model`** and start coding!

---

## 🧠 Bring Your Own Model Options

Check out [`BRING_YOUR_OWN_MODEL_GUIDE.md`](BRING_YOUR_OWN_MODEL_GUIDE.md) for full instructions on:
1. **Ollama Registry Models**: `qwen2.5-coder`, `deepseek-r1`, `llama3.1`, `codestral`, etc.
2. **Hugging Face GGUFs**: Direct download via `wget` into Colab.
3. **Google Drive Models**: Mount your Drive directly into the runtime.

---

## 🔒 Security

- Cloudflare Quick Tunnels generate temporary HTTPS URLs.
- The FastAPI proxy rejects any request that lacks the generated `Authorization: Bearer <KEY>` header.
- Ollama's unauthenticated local port (`11434`) is never exposed to the public Internet.
- No personal API keys or tunnel URLs are ever stored in version control.

---

## 📄 License

This project is licensed under the **PolyForm Noncommercial License 1.0.0 (Source-Available)**. See [`LICENSE`](LICENSE) and the [Source-Available Guide](SOURCE_AVAILABLE_GUIDE.md) for details.
Free for personal use, research, and non-commercial development.
