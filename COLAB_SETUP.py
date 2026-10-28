# ==============================================================================
# OpenColab — Universal Multi-Backend Cloud LLM Server for OpenCode v2
# Supports: Ollama | llama.cpp | vLLM | SGLang | HuggingFace Transformers
# Run this entire script in Google Colab on a GPU runtime (T4, L4, A100, etc.)
# ==============================================================================

import subprocess
import sys
import os
import time
import re
import secrets
import urllib.request
import requests

# ------------------------------------------------------------------------------
# CONFIGURATION: Choose your Backend, Model, and Context Window!
# ------------------------------------------------------------------------------
# Supported BACKEND values:
#   - "ollama"       : Ollama daemon with Flash Attention & q4_0 KV cache (Default)
#   - "llamacpp"     : Native llama.cpp server (llama-cpp-python CUDA / GGUF)
#   - "vllm"         : vLLM PagedAttention OpenAI server (Safetensors / AWQ / GPTQ)
#   - "sglang"       : SGLang RadixAttention server for fast structured outputs
#   - "transformers" : Native HuggingFace Transformers + PyTorch streaming server
#   - "auto"         : Auto-detects backend from MODEL_ID format
BACKEND = os.environ.get("BACKEND", "ollama").strip().lower()

# Examples of MODEL_ID by backend:
#   [ollama]       "SparkLLM/Spark-X2.6-4B:q4_k_m", "qwen2.5-coder:7b", "deepseek-r1:8b"
#   [llamacpp]     "/content/model.gguf" or "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf"
#   [vllm/sglang]  "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#   [transformers] "Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_ID = os.environ.get("MODEL_ID", "SparkLLM/Spark-X2.6-4B:q4_k_m")

# Context window: 131072 (128k), 262144 (256k), up to 1048576 (1M context on A100)
CONTEXT_WINDOW = int(os.environ.get("CONTEXT_WINDOW", "131072"))

# Optional backend tuning parameters
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"
VLLM_TOOL_PARSER = os.environ.get("VLLM_TOOL_PARSER", "hermes")

# Automatically generate a secure, random bearer token per session (or set via env)
API_KEY = os.environ.get("COLAB_API_KEY", secrets.token_urlsafe(32))

UPSTREAM_PORT = 11434
UPSTREAM_BASE = f"http://127.0.0.1:{UPSTREAM_PORT}"
PROXY_PORT = 8000


def resolve_backend(backend: str, model_id: str) -> str:
    valid = {"ollama", "llamacpp", "vllm", "sglang", "transformers"}
    if backend in valid:
        return backend
    if backend in {"llama.cpp", "llama-cpp", "gguf"}:
        return "llamacpp"
    if backend in {"hf", "huggingface"}:
        return "transformers"
    # Auto-detection heuristics
    if model_id.endswith(".gguf") or ".gguf" in model_id.lower():
        return "llamacpp"
    if "/" in model_id and ":" not in model_id and not os.path.exists(model_id):
        return "vllm"
    return "ollama"


ACTIVE_BACKEND = resolve_backend(BACKEND, MODEL_ID)

print("=" * 65)
print("  OPENCOLAB MULTI-BACKEND INITIALIZATION")
print("=" * 65)
print(f"Selected Backend : {ACTIVE_BACKEND.upper()} (requested: {BACKEND})")
print(f"Target Model     : {MODEL_ID}")
print(f"Context Window   : {CONTEXT_WINDOW} tokens")
print("=" * 65)

print("\n1. Installing core proxy dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "httpx", "huggingface_hub"], check=True)

print("2. Terminating any stale backend or tunnel processes...")
for proc_pattern in ["uvicorn", "ollama", "llama_cpp", "llama-server", "vllm", "sglang", "hf_worker", "cloudflared"]:
    subprocess.run(f"pkill -9 -f {proc_pattern} || true", shell=True)
time.sleep(2)


def wait_for_upstream(url: str, timeout_sec: int = 180, headers: dict = None, label: str = "Upstream backend"):
    for _ in range(timeout_sec):
        try:
            r = requests.get(url, headers=headers or {}, timeout=3)
            if r.ok:
                print(f"{label} online at {url}!")
                return True
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"{label} failed to start within {timeout_sec}s. Check /content/backend.log")


def ensure_gguf_file(model_spec: str) -> str:
    """Resolves a local path, direct URL, or HuggingFace 'repo_id:filename.gguf' into a local .gguf path."""
    if os.path.exists(model_spec):
        return os.path.abspath(model_spec)
    if model_spec.startswith("http://") or model_spec.startswith("https://"):
        dest = "/content/downloaded_model.gguf"
        print(f"Downloading GGUF from URL to {dest}...")
        subprocess.run(["curl", "-L", "-o", dest, model_spec], check=True)
        return dest
    if ":" in model_spec and "/" in model_spec and model_spec.lower().endswith(".gguf"):
        repo_id, filename = model_spec.split(":", 1)
        print(f"Downloading {filename} from Hugging Face repo {repo_id}...")
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir="/content/models")
    raise FileNotFoundError(
        f"Could not locate GGUF file '{model_spec}'. Provide a local .gguf path, HTTPS URL, or 'owner/repo:file.gguf'."
    )


# ==============================================================================
# STEP 3–6: LAUNCH SELECTED INFERENCE BACKEND ON PORT 11434
# ==============================================================================
SERVE_MODEL_ID = MODEL_ID

if ACTIVE_BACKEND == "ollama":
    print("3. Installing Ollama & zstd...")
    subprocess.run("sudo apt-get update -qq && sudo apt-get install -y -qq zstd", shell=True, check=True)
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

    print("4. Launching Ollama daemon with Flash Attention & 4-bit KV Cache...")
    ollama_env = os.environ.copy()
    ollama_env["OLLAMA_HOST"] = f"127.0.0.1:{UPSTREAM_PORT}"
    ollama_env["OLLAMA_FLASH_ATTENTION"] = "1"
    ollama_env["OLLAMA_KV_CACHE_TYPE"] = "q4_0"
    ollama_env["OLLAMA_NUM_PARALLEL"] = "1"
    ollama_env["OLLAMA_NUM_CTX"] = str(CONTEXT_WINDOW)
    ollama_env["OLLAMA_KEEP_ALIVE"] = "-1"

    backend_log = open("/content/backend.log", "w")
    subprocess.Popen(["ollama", "serve"], stdout=backend_log, stderr=subprocess.STDOUT, env=ollama_env)
    wait_for_upstream(f"{UPSTREAM_BASE}/api/tags", timeout_sec=45, label="Ollama daemon")

    if os.path.exists(MODEL_ID) or MODEL_ID.lower().endswith(".gguf"):
        gguf_path = ensure_gguf_file(MODEL_ID)
        print(f"5. Building custom Ollama model from GGUF: {gguf_path}...")
        modelfile_content = f"FROM {gguf_path}\nPARAMETER num_ctx {CONTEXT_WINDOW}\n"
        with open("/content/Modelfile", "w") as f:
            f.write(modelfile_content)
        SERVE_MODEL_ID = "custom-gguf-model"
        subprocess.run(["ollama", "create", SERVE_MODEL_ID, "-f", "/content/Modelfile"], check=True)
    else:
        print(f"5. Pulling Ollama model weights: {MODEL_ID}...")
        subprocess.run(["ollama", "pull", MODEL_ID], check=True)
        print(f"6. Baking {CONTEXT_WINDOW} context limit directly into Ollama manifest...")
        modelfile_content = f"FROM {MODEL_ID}\nPARAMETER num_ctx {CONTEXT_WINDOW}\n"
        with open("/content/Modelfile", "w") as f:
            f.write(modelfile_content)
        subprocess.run(["ollama", "create", MODEL_ID, "-f", "/content/Modelfile"], check=True)
        SERVE_MODEL_ID = MODEL_ID

    print(f"6b. Pre-warming {SERVE_MODEL_ID} into GPU VRAM ({CONTEXT_WINDOW} KV cache)...")
    try:
        requests.post(
            f"{UPSTREAM_BASE}/api/generate",
            json={"model": SERVE_MODEL_ID, "prompt": "hi", "stream": False, "keep_alive": -1, "options": {"num_ctx": CONTEXT_WINDOW, "num_predict": 1}},
            timeout=300
        )
        print("Model pre-warmed and pinned in GPU VRAM!")
    except Exception as e:
        print(f"Warmup note: {e}")

elif ACTIVE_BACKEND == "llamacpp":
    print("3. Preparing llama.cpp CUDA server backend...")
    gguf_path = ensure_gguf_file(MODEL_ID)
    SERVE_MODEL_ID = os.path.basename(gguf_path).replace(".gguf", "")

    # Install llama-cpp-python[server] with CUDA support if not already installed
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "llama-cpp-python[server]",
            "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"
        ],
        check=True
    )

    print(f"4. Launching llama.cpp server for {gguf_path} (n_ctx={CONTEXT_WINDOW})...")
    backend_log = open("/content/backend.log", "w")
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", gguf_path,
        "--model_alias", SERVE_MODEL_ID,
        "--n_ctx", str(CONTEXT_WINDOW),
        "--n_gpu_layers", "-1",
        "--flash_attn", "True",
        "--host", "127.0.0.1",
        "--port", str(UPSTREAM_PORT)
    ]
    subprocess.Popen(cmd, stdout=backend_log, stderr=subprocess.STDOUT)
    wait_for_upstream(f"{UPSTREAM_BASE}/v1/models", timeout_sec=180, label="llama.cpp server")

elif ACTIVE_BACKEND == "vllm":
    print("3. Installing vLLM engine (this may take 1-2 minutes on first run)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

    print(f"4. Launching vLLM OpenAI server for {MODEL_ID} (max-model-len={CONTEXT_WINDOW})...")
    backend_log = open("/content/backend.log", "w")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--served-model-name", SERVE_MODEL_ID,
        "--max-model-len", str(CONTEXT_WINDOW),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--trust-remote-code",
        "--dtype", "auto",
        "--enable-auto-tool-choice",
        "--tool-call-parser", VLLM_TOOL_PARSER,
        "--host", "127.0.0.1",
        "--port", str(UPSTREAM_PORT)
    ]
    subprocess.Popen(cmd, stdout=backend_log, stderr=subprocess.STDOUT)
    wait_for_upstream(f"{UPSTREAM_BASE}/v1/models", timeout_sec=360, label="vLLM OpenAI server")

elif ACTIVE_BACKEND == "sglang":
    print("3. Installing SGLang RadixAttention engine...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sglang[all]"], check=True)

    print(f"4. Launching SGLang server for {MODEL_ID} (context-length={CONTEXT_WINDOW})...")
    backend_log = open("/content/backend.log", "w")
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL_ID,
        "--served-model-name", SERVE_MODEL_ID,
        "--context-length", str(CONTEXT_WINDOW),
        "--mem-fraction-static", str(GPU_MEM_UTIL),
        "--trust-remote-code",
        "--host", "127.0.0.1",
        "--port", str(UPSTREAM_PORT)
    ]
    subprocess.Popen(cmd, stdout=backend_log, stderr=subprocess.STDOUT)
    wait_for_upstream(f"{UPSTREAM_BASE}/v1/models", timeout_sec=360, label="SGLang server")

elif ACTIVE_BACKEND == "transformers":
    print("3. Preparing native HuggingFace Transformers + PyTorch backend...")
    pip_pkgs = ["transformers", "accelerate", "sentencepiece"]
    if LOAD_IN_4BIT:
        pip_pkgs.append("bitsandbytes")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pip_pkgs, check=True)

    hf_worker_code = f'''
import time, json, uuid, torch
from threading import Thread
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "{MODEL_ID}"
CONTEXT_WINDOW = {CONTEXT_WINDOW}
LOAD_IN_4BIT = {str(LOAD_IN_4BIT)}

print(f"Loading HuggingFace model {{MODEL_ID}}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model_kwargs = {{"device_map": "auto", "trust_remote_code": True}}
if torch.cuda.is_available():
    model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
if LOAD_IN_4BIT and torch.cuda.is_available():
    from transformers import BitsAndBytesConfig
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
model.eval()
print("Model loaded successfully!")

app = FastAPI()

@app.get("/v1/models")
async def list_models():
    return {{"object": "list", "data": [{{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "transformers"}}]}}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))
    max_new_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
    temperature = float(body.get("temperature", 0.7))

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\\n".join([f"{{m.get('role', 'user')}}: {{m.get('content', '')}}" for m in messages]) + "\\nassistant: "

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=CONTEXT_WINDOW).to(model.device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.01),
        temperature=max(temperature, 0.01),
        pad_token_id=tokenizer.eos_token_id
    )

    req_id = f"chatcmpl-{{uuid.uuid4().hex[:12]}}"
    created = int(time.time())

    if not stream:
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return JSONResponse({{
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{{"index": 0, "message": {{"role": "assistant", "content": text}}, "finish_reason": "stop"}}]
        }})

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs["streamer"] = streamer
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    def event_stream():
        for token_text in streamer:
            chunk = {{
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{{"index": 0, "delta": {{"content": token_text}}, "finish_reason": None}}]
            }}
            yield f"data: {{json.dumps(chunk, ensure_ascii=False)}}\\n\\n"
        done_chunk = {{
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{{"index": 0, "delta": {{}}, "finish_reason": "stop"}}]
        }}
        yield f"data: {{json.dumps(done_chunk)}}\\n\\ndata: [DONE]\\n\\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
'''
    with open("/content/hf_worker.py", "w") as f:
        f.write(hf_worker_code.strip())

    print(f"4. Launching Transformers server on port {UPSTREAM_PORT}...")
    backend_log = open("/content/backend.log", "w")
    subprocess.Popen(
        ["uvicorn", "hf_worker:app", "--host", "127.0.0.1", "--port", str(UPSTREAM_PORT)],
        cwd="/content",
        stdout=backend_log,
        stderr=subprocess.STDOUT
    )
    wait_for_upstream(f"{UPSTREAM_BASE}/v1/models", timeout_sec=300, label="Transformers server")


# ==============================================================================
# STEP 7–8: UNIFIED FASTAPI OPENAI-COMPATIBLE TRANSLATION PROXY (:8000)
# ==============================================================================
print("7. Writing unified FastAPI OpenAI translation & auth proxy...")
proxy_code = f'''
import os, json, time, httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse

UPSTREAM_BASE = "{UPSTREAM_BASE}"
ACTIVE_BACKEND = "{ACTIVE_BACKEND}"
API_KEY = "{API_KEY}"
SERVE_MODEL_ID = "{SERVE_MODEL_ID}"
CONTEXT_WINDOW = {CONTEXT_WINDOW}

app = FastAPI(title="OpenColab Unified Multi-Backend Proxy")
client = httpx.AsyncClient(timeout=600.0)

def verify_auth(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if auth.split("Bearer ", 1)[1].strip() != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.get("/health")
async def health_check():
    return {{
        "status": "ok",
        "backend": ACTIVE_BACKEND,
        "model": SERVE_MODEL_ID,
        "context_window": CONTEXT_WINDOW
    }}

@app.get("/v1/models")
async def list_models(request: Request):
    verify_auth(request)
    return {{
        "object": "list",
        "data": [
            {{
                "id": SERVE_MODEL_ID,
                "object": "model",
                "created": 1677610143,
                "owned_by": ACTIVE_BACKEND,
                "context_window": CONTEXT_WINDOW
            }}
        ]
    }}

def translate_sse_chunk(line: str) -> str:
    if not line.startswith("data: "):
        return line
    payload = line[6:].strip()
    if payload == "[DONE]":
        return line
    try:
        obj = json.loads(payload)
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta")
            if isinstance(delta, dict):
                # Normalize reasoning tokens across Ollama ('reasoning') and vLLM/SGLang/llama.cpp ('reasoning_content')
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                content = delta.get("content")
                if reasoning and not content:
                    delta["content"] = reasoning
                delta.pop("reasoning", None)
                delta.pop("reasoning_content", None)
        return "data: " + json.dumps(obj, ensure_ascii=False)
    except Exception:
        return line

async def sse_stream_generator(upstream_response):
    buffer = b""
    async for chunk in upstream_response.aiter_bytes():
        buffer += chunk
        while b"\\n" in buffer:
            line_bytes, buffer = buffer.split(b"\\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\\r")
            if not line:
                yield b"\\n"
                continue
            yield (translate_sse_chunk(line) + "\\n").encode("utf-8")
    if buffer:
        line = buffer.decode("utf-8", errors="replace").rstrip("\\r")
        if line:
            yield (translate_sse_chunk(line) + "\\n\\n").encode("utf-8")

@app.api_route("/{{path:path}}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_all(path: str, request: Request):
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={{
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            }}
        )
    verify_auth(request)
    if not path.startswith("v1/"):
        raise HTTPException(status_code=404, detail="Only /v1 endpoints are exposed")
    target = f"{{UPSTREAM_BASE}}/{{path}}"
    if request.url.query:
        target += "?" + request.url.query
    body_bytes = await request.body()
    is_stream = False
    if request.method == "POST" and path == "v1/chat/completions":
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
            payload["model"] = SERVE_MODEL_ID
            # Only inject Ollama-specific options.num_ctx when using Ollama;
            # strict OpenAI servers (vLLM, SGLang, llama.cpp) reject unknown top-level options.
            if ACTIVE_BACKEND == "ollama":
                options = payload.get("options", {{}})
                options["num_ctx"] = CONTEXT_WINDOW
                payload["options"] = options
            else:
                payload.pop("options", None)
            is_stream = bool(payload.get("stream", False))
            body_bytes = json.dumps(payload).encode("utf-8")
        except Exception:
            pass
    headers = {{"Content-Type": "application/json", "Accept": request.headers.get("accept", "*/*")}}
    req = client.build_request(request.method, target, headers=headers, content=body_bytes)
    upstream = await client.send(req, stream=is_stream)
    if is_stream:
        return StreamingResponse(
            sse_stream_generator(upstream),
            status_code=upstream.status_code,
            headers={{
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }}
        )
    else:
        resp_body = await upstream.aread()
        try:
            data = json.loads(resp_body.decode("utf-8"))
            if "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0].get("message", {{}})
                reasoning = msg.get("reasoning") or msg.get("reasoning_content")
                if reasoning and not msg.get("content"):
                    msg["content"] = reasoning
            resp_body = json.dumps(data).encode("utf-8")
        except Exception:
            pass
        return Response(content=resp_body, status_code=upstream.status_code, headers={{"Content-Type": "application/json"}})
'''

with open("/content/proxy.py", "w") as f:
    f.write(proxy_code.strip())

print(f"8. Launching FastAPI server on port {PROXY_PORT}...")
proxy_log = open("/content/proxy.log", "w")
subprocess.Popen(
    ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", str(PROXY_PORT)],
    cwd="/content",
    stdout=proxy_log,
    stderr=subprocess.STDOUT
)

wait_for_upstream(
    f"http://127.0.0.1:{PROXY_PORT}/v1/models",
    timeout_sec=20,
    headers={"Authorization": f"Bearer {API_KEY}"},
    label="FastAPI proxy"
)

# ==============================================================================
# STEP 9: CLOUDFLARE QUICK TUNNEL
# ==============================================================================
print("9. Creating Cloudflare HTTPS tunnel...")
if not os.path.exists("/content/cloudflared"):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        "/content/cloudflared"
    )
    subprocess.run("chmod +x /content/cloudflared", shell=True, check=True)

tunnel_log_path = "/content/cloudflared.log"
subprocess.Popen(
    ["/content/cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PROXY_PORT}", "--no-autoupdate"],
    stdout=open(tunnel_log_path, "w"),
    stderr=subprocess.STDOUT
)

tunnel_url = None
for _ in range(30):
    time.sleep(2)
    if os.path.exists(tunnel_log_path):
        with open(tunnel_log_path) as f:
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", f.read())
            if match:
                tunnel_url = match.group(0)
                break

if not tunnel_url:
    raise RuntimeError("Tunnel failed to start.")

print("\n" + "=" * 65)
print("             OPENCOLAB BACKEND READY (MULTI-BACKEND)")
print("=" * 65)
print(f"Inference Backend : {ACTIVE_BACKEND}")
print(f"Active Model      : {SERVE_MODEL_ID}")
print(f"Context Window    : {CONTEXT_WINDOW} tokens")
print(f"Tunnel Public URL : {tunnel_url}")
print(f"Base URL          : {tunnel_url}/v1")
print(f"API Bearer Key    : {API_KEY}")
print("=" * 65)
print("\nRun on Windows:")
print(
    f'powershell -ExecutionPolicy Bypass -File .\\Apply-Colab-Tunnel.ps1 '
    f'-TunnelUrl "{tunnel_url}" -ApiKey "{API_KEY}" '
    f'-ModelId "{SERVE_MODEL_ID}" -Backend "{ACTIVE_BACKEND}" -ContextWindow {CONTEXT_WINDOW}'
)
