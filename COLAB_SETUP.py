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

# Context window:
#   - 32768 (32k) to 131072 (128k): Recommended for 4B-9B models on 15GB T4 or 14B-27B on L4/A100
#   - Up to 1048576 (1M): ONLY works on models trained/extended for 1M context (e.g. Qwen2.5-1M)
#     running on a high-VRAM GPU (Colab Pro+ A100 80GB). A 32B model will NOT fit on 16GB VRAM!
#   - NOTE: Sustained usage & background tunnels require a Paid Google Colab Plan (Colab Pro / Pro+)
#     to avoid free-tier runtime termination and comply with Google Colab Terms of Service.
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


def check_hardware_and_limits(model_id: str, context_window: int):
    gpu_name = "Unknown / CPU"
    vram_gb = 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        gpu_name = parts[0]
        vram_gb = float(parts[1]) / 1024.0
    except Exception:
        pass

    print("=" * 65)
    print("  OPENCOLAB MULTI-BACKEND INITIALIZATION")
    print("=" * 65)
    print(f"Detected GPU     : {gpu_name} ({vram_gb:.1f} GB VRAM)" if vram_gb > 0 else "Detected GPU     : None (WARNING: No NVIDIA GPU detected!)")
    print(f"Selected Backend : {ACTIVE_BACKEND.upper()} (requested: {BACKEND})")
    print(f"Target Model     : {model_id}")
    print(f"Context Window   : {context_window} tokens")
    print("-" * 65)
    print("  [NOTICE] Google Colab Terms of Service & Hardware Limits:")
    print("  • Free-tier Colab accounts terminate idle/long-running sessions.")
    print("    A Paid Google Colab Plan (Colab Pro / Pro+) is required for")
    print("    sustained usage in compliance with Google Colab ToS.")
    if context_window > 131072:
        print(f"  • [WARNING] CONTEXT_WINDOW={context_window} (>128k): Ensure '{model_id}'")
        print("    natively supports >128k/1M tokens (e.g. YaRN/1M models) and that")
        print("    you are running on an A100 (40GB/80GB) GPU to avoid OOM crashes.")
    if 0 < vram_gb <= 16.5:
        if any(sz in model_id.lower() for sz in ["27b", "30b", "32b", "34b", "70b", "72b"]):
            print(f"  • [CRITICAL VRAM WARNING] '{model_id}' is too large for {vram_gb:.1f} GB VRAM!")
            print("    A 27B/32B Q4 model requires ~18-20 GB VRAM just for weights alone.")
            print("    Use a 4B-9B model on a 15GB T4, or switch to an L4 (24GB) / A100 GPU.")
        elif context_window > 131072:
            print(f"  • [CRITICAL VRAM WARNING] {context_window} context on {vram_gb:.1f} GB VRAM")
            print("    will likely exceed GPU memory! Stick to 32k-128k on a 15GB T4.")
    print("=" * 65)


check_hardware_and_limits(MODEL_ID, CONTEXT_WINDOW)

print("\n1. Installing core proxy dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "httpx", "huggingface_hub"], check=True)

print("2. Terminating any stale backend or tunnel processes...")
for proc_pattern in ["uvicorn", "ollama", "llama_cpp", "llama-server", "vllm", "sglang", "hf_worker", "cloudflared"]:
    subprocess.run(f"pkill -9 -f {proc_pattern} || true", shell=True)
time.sleep(2)


def wait_for_upstream(url: str, timeout_sec: int = 180, headers: dict = None, label: str = "Upstream backend", proc=None):
    for _ in range(timeout_sec):
        # Fail fast if the backend process already crashed instead of polling until the timeout.
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"{label} exited early with code {proc.returncode}. Last lines of /content/backend.log:\n"
                + _tail_backend_log()
            )
        try:
            r = requests.get(url, headers=headers or {}, timeout=3)
            if r.ok:
                print(f"{label} online at {url}!")
                return True
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(
        f"{label} failed to start within {timeout_sec}s. Last lines of /content/backend.log:\n" + _tail_backend_log()
    )


def _tail_backend_log(lines: int = 40) -> str:
    try:
        with open("/content/backend.log", "r", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    except Exception:
        return "(no backend.log available)"


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
    backend_proc = subprocess.Popen(["ollama", "serve"], stdout=backend_log, stderr=subprocess.STDOUT, env=ollama_env)
    wait_for_upstream(f"{UPSTREAM_BASE}/api/tags", timeout_sec=45, label="Ollama daemon", proc=backend_proc)

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
    # The prebuilt cu124 llama-cpp-python wheel links against libcudart.so.12 / libcublas.so.12, but newer
    # Colab images ship CUDA 13 system-wide. The CUDA 12 runtime libs are bundled in pip's nvidia-* packages,
    # so expose them to the server process (otherwise it crashes with "libcudart.so.12: cannot open shared object file").
    import glob
    llama_env = os.environ.copy()
    nvidia_libs = glob.glob(f"{sys.prefix}/lib/python3*/dist-packages/nvidia/*/lib") + \
        glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib") + \
        glob.glob("/usr/local/lib/python3*/site-packages/nvidia/*/lib")
    llama_env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(nvidia_libs)) + ":" + llama_env.get("LD_LIBRARY_PATH", "")
    backend_proc = subprocess.Popen(cmd, stdout=backend_log, stderr=subprocess.STDOUT, env=llama_env)
    wait_for_upstream(f"{UPSTREAM_BASE}/v1/models", timeout_sec=180, label="llama.cpp server", proc=backend_proc)

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
    backend_proc = subprocess.Popen(cmd, stdout=backend_log, stderr=subprocess.STDOUT)
    wait_for_upstream(f"{UPSTREAM_BASE}/v1/models", timeout_sec=360, label="vLLM OpenAI server", proc=backend_proc)

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
    backend_proc = subprocess.Popen(cmd, stdout=backend_log, stderr=subprocess.STDOUT)
    wait_for_upstream(f"{UPSTREAM_BASE}/v1/models", timeout_sec=360, label="SGLang server", proc=backend_proc)

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
    backend_proc = subprocess.Popen(
        ["uvicorn", "hf_worker:app", "--host", "127.0.0.1", "--port", str(UPSTREAM_PORT)],
        cwd="/content",
        stdout=backend_log,
        stderr=subprocess.STDOUT
    )
    wait_for_upstream(f"{UPSTREAM_BASE}/v1/models", timeout_sec=300, label="Transformers server", proc=backend_proc)


# ==============================================================================
# STEP 7–8: UNIFIED FASTAPI OPENAI-COMPATIBLE TRANSLATION PROXY (:8000)
# ==============================================================================
# Tool-call parsing for backends whose server doesn't return structured tool_calls.
#   TOOL_CALL_PARSER = "auto" (on for llamacpp/transformers), "on", or "off"
TOOL_CALL_PARSER = os.environ.get("TOOL_CALL_PARSER", "auto").strip().lower()
BUFFER_TOOL_RESPONSES = TOOL_CALL_PARSER == "on" or (
    TOOL_CALL_PARSER == "auto" and ACTIVE_BACKEND in ("llamacpp", "transformers")
)

# Text-embedded tool-call parsing.
# Some GGUF / llama.cpp models (e.g. Spark-X2.6, GLM-style) emit tool calls as plain text:
#   <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>   (GLM style)
#   <tool_call>{"name": "NAME", "arguments": {...}}</tool_call>               (Hermes style)
# and leak "...</think>" into content. llama_cpp.server returns that as raw text, so OpenAI clients
# (OpenCode) never see structured `tool_calls`. This converts it back to the OpenAI format.
TOOL_PARSER_CODE = r"""
import re, uuid
THINK_CLOSE = "</think>"
_TC_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

def _tool_props(tools):
    props = {}
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        if fn.get("name"):
            props[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    return props

def _parse_tool_call_block(block, props):
    block = block.strip()
    if block.startswith("{"):
        try:
            obj = json.loads(block)
            name = obj.get("name")
            args = obj.get("arguments", obj.get("parameters", {}))
            if name:
                return name, (args if isinstance(args, str) else json.dumps(args, ensure_ascii=False))
        except Exception:
            pass
    idx = block.find("<arg_key>")
    name = (block if idx == -1 else block[:idx]).strip()
    if not name:
        return None
    schema = props.get(name, {})
    args = {}
    for k, v in _ARG_RE.findall(block):
        k, v = k.strip(), v.strip()
        if (schema.get(k) or {}).get("type") == "string":
            args[k] = v
        else:
            try:
                args[k] = json.loads(v)
            except Exception:
                args[k] = v
    return name, json.dumps(args, ensure_ascii=False)

def normalize_message(msg, tools=None):
    # Split leaked </think> into reasoning_content and convert text tool calls to tool_calls.
    content = msg.get("content")
    if not isinstance(content, str):
        return False
    changed = False
    if THINK_CLOSE in content and not (msg.get("reasoning_content") or msg.get("reasoning")):
        head, tail = content.split(THINK_CLOSE, 1)
        msg["reasoning_content"] = head.replace("<think>", "").strip()
        content = tail.lstrip()
        changed = True
    if not msg.get("tool_calls") and "<tool_call>" in content:
        props = _tool_props(tools)
        calls = []
        for block in _TC_RE.findall(content):
            parsed = _parse_tool_call_block(block, props)
            if parsed:
                calls.append({
                    "id": "call_" + uuid.uuid4().hex[:24],
                    "type": "function",
                    "function": {"name": parsed[0], "arguments": parsed[1]},
                })
        if calls:
            content = _TC_RE.sub("", content).strip()
            msg["tool_calls"] = calls
            changed = True
    if changed:
        msg["content"] = content if (content or not msg.get("tool_calls")) else None
    return changed

def tool_calls_to_text(tool_calls):
    # Render OpenAI tool_calls history back into the model's native GLM-style text so chat templates
    # that require dict arguments (and servers that only accept string arguments) both work.
    out = ""
    for tc in tool_calls or []:
        fn = (tc or {}).get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except Exception:
                args = {"input": args}
        out += "<tool_call>" + name
        for k, v in (args if isinstance(args, dict) else {}).items():
            val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            out += "<arg_key>" + str(k) + "</arg_key><arg_value>" + val + "</arg_value>"
        out += "</tool_call>"
    return out

def postprocess_completion(data, tools=None):
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return data
    for ch in choices:
        msg = ch.get("message")
        if isinstance(msg, dict) and normalize_message(msg, tools) and msg.get("tool_calls"):
            ch["finish_reason"] = "tool_calls"
    return data

def completion_to_sse(data):
    # Re-emit a buffered chat.completion as OpenAI SSE chunks (list of strings).
    base = {"id": data.get("id", "chatcmpl-" + uuid.uuid4().hex[:12]), "object": "chat.completion.chunk",
            "created": data.get("created", int(time.time())), "model": data.get("model", SERVE_MODEL_ID)}
    def chunk(delta, finish=None):
        return "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}, ensure_ascii=False) + "\n\n"
    out = []
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    out.append(chunk({"role": "assistant"}))
    if msg.get("reasoning_content"):
        out.append(chunk({"reasoning_content": msg["reasoning_content"]}))
    if msg.get("content"):
        out.append(chunk({"content": msg["content"]}))
    for i, tc in enumerate(msg.get("tool_calls") or []):
        out.append(chunk({"tool_calls": [{"index": i, "id": tc.get("id"), "type": "function", "function": tc.get("function")}]}))
    out.append(chunk({}, choice.get("finish_reason") or "stop"))
    out.append("data: [DONE]\n\n")
    return out
"""

print("7. Writing unified FastAPI OpenAI translation & auth proxy...")
proxy_code = f'''
import os, json, time, asyncio, httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse

UPSTREAM_BASE = "{UPSTREAM_BASE}"
ACTIVE_BACKEND = "{ACTIVE_BACKEND}"
API_KEY = "{API_KEY}"
SERVE_MODEL_ID = "{SERVE_MODEL_ID}"
CONTEXT_WINDOW = {CONTEXT_WINDOW}
BUFFER_TOOL_RESPONSES = {BUFFER_TOOL_RESPONSES}
{TOOL_PARSER_CODE}
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

def sanitize_chat_payload(payload: dict) -> dict:
    payload["model"] = SERVE_MODEL_ID
    messages = payload.get("messages")
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            # 1. Normalize tool response content: flatten content block arrays into a string
            #    so GGUF Jinja templates that concatenate message.content directly never fail.
            if role == "tool":
                content = m.get("content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            parts.append(str(part.get("text") or part.get("content") or json.dumps(part)))
                        else:
                            parts.append(str(part))
                    m["content"] = "\\n".join(parts)
                elif content is None:
                    m["content"] = ""
                elif not isinstance(content, str):
                    m["content"] = json.dumps(content)
            # 2. Sanitize assistant messages in multi-turn tool histories:
            #    Keep reasoning inside reasoning_content (never raw content before <tool_call>)
            #    so reasoning models (e.g. Spark-X2.6, Qwen3, DeepSeek-R1) render valid <think> tags.
            elif role == "assistant":
                # llama_cpp.server (pydantic) rejects assistant messages whose content is null;
                # OpenAI clients send content=null alongside tool_calls, so normalize it to "".
                if m.get("content") is None:
                    m["content"] = ""
                if m.get("tool_calls"):
                    content = m.get("content")
                    if isinstance(content, str) and content.strip():
                        if not m.get("reasoning_content") and not m.get("reasoning"):
                            m["reasoning_content"] = content[:4000]
                        m["content"] = ""
                    for tc in m.get("tool_calls", []):
                        if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                            args = tc["function"].get("arguments")
                            if isinstance(args, dict):
                                tc["function"]["arguments"] = json.dumps(args)
                    if BUFFER_TOOL_RESPONSES:
                        # llama_cpp.server/transformers templates choke on structured tool_calls history
                        # (string vs dict arguments). Replay them as native <tool_call> text instead.
                        m["content"] = (m.get("content") or "") + tool_calls_to_text(m.get("tool_calls"))
                        m.pop("tool_calls", None)
                else:
                    content = m.get("content")
                    if isinstance(content, str) and len(content) > 16000:
                        m["content"] = content[:4000] + "\\n...[truncated]..."
                for r_key in ("reasoning_content", "reasoning"):
                    r_val = m.get(r_key)
                    if isinstance(r_val, str) and len(r_val) > 4000:
                        m[r_key] = r_val[-4000:]

    # Default to concise thinking effort during tool-calling turns unless overridden
    if payload.get("tools") and "reasoning_effort" not in payload:
        payload["reasoning_effort"] = "low"

    # Only inject Ollama-specific options.num_ctx when using Ollama;
    # strict OpenAI servers (vLLM, SGLang, llama.cpp) reject unknown top-level options.
    if ACTIVE_BACKEND == "ollama":
        options = payload.get("options", {{}})
        options["num_ctx"] = CONTEXT_WINDOW
        payload["options"] = options
    else:
        payload.pop("options", None)
    return payload

def translate_sse_chunk(line: str, keep_reasoning_separate: bool = True) -> str:
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
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                content = delta.get("content")
                if reasoning:
                    if keep_reasoning_separate:
                        delta["reasoning_content"] = reasoning
                        delta.pop("reasoning", None)
                        if content == "":
                            delta.pop("content", None)
                    else:
                        if not content:
                            delta["content"] = reasoning
                        delta.pop("reasoning", None)
                        delta.pop("reasoning_content", None)
        return "data: " + json.dumps(obj, ensure_ascii=False)
    except Exception:
        return line

async def sse_stream_generator(upstream_response, keep_reasoning_separate: bool = True):
    queue = asyncio.Queue()

    async def pump_upstream():
        try:
            async for chunk in upstream_response.aiter_bytes():
                await queue.put(("chunk", chunk))
            await queue.put(("eof", b""))
        except Exception as exc:
            await queue.put(("error", exc))

    pump_task = asyncio.create_task(pump_upstream())
    buffer = b""
    try:
        while True:
            try:
                msg_type, data = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                # While Ollama buffers large <tool_call> payloads (e.g. writing a full file),
                # emit a valid no-op OpenAI SSE delta chunk every 2s so Cloudflare, Bun fetch,
                # and OpenCode's SSE reader never hit idle socket or chunk timeouts.
                keepalive_chunk = {{
                    "id": "chatcmpl-keepalive",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": SERVE_MODEL_ID,
                    "choices": [{{"index": 0, "delta": {{}}, "finish_reason": None}}]
                }}
                yield f"data: {{json.dumps(keepalive_chunk)}}\\n\\n".encode("utf-8")
                continue
            if msg_type in ("eof", "error"):
                break
            buffer += data
            while b"\\n" in buffer:
                line_bytes, buffer = buffer.split(b"\\n", 1)
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\\r")
                if not line:
                    yield b"\\n"
                    continue
                yield (translate_sse_chunk(line, keep_reasoning_separate) + "\\n").encode("utf-8")
        if buffer:
            line = buffer.decode("utf-8", errors="replace").rstrip("\\r")
            if line:
                yield (translate_sse_chunk(line, keep_reasoning_separate) + "\\n\\n").encode("utf-8")
    finally:
        pump_task.cancel()
        await upstream_response.aclose()

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
    has_tools = False
    client_stream = False
    tools = None
    if request.method == "POST" and path == "v1/chat/completions":
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
            tools = payload.get("tools")
            has_tools = bool(tools)
            payload = sanitize_chat_payload(payload)
            client_stream = bool(payload.get("stream", False))
            # Tool requests on backends that return tool calls as raw text are buffered so the
            # reply can be parsed into structured tool_calls, then re-streamed to the client.
            if has_tools and BUFFER_TOOL_RESPONSES:
                payload["stream"] = False
                payload.pop("stream_options", None)
            is_stream = bool(payload.get("stream", False))
            body_bytes = json.dumps(payload).encode("utf-8")
        except Exception:
            pass
    headers = {{"Content-Type": "application/json", "Accept": request.headers.get("accept", "*/*")}}
    req = client.build_request(request.method, target, headers=headers, content=body_bytes)
    if has_tools and BUFFER_TOOL_RESPONSES and client_stream:
        async def buffered_sse():
            task = asyncio.create_task(client.send(req, stream=False))
            while True:
                done, _ = await asyncio.wait({{task}}, timeout=2.0)
                if done:
                    break
                keepalive = {{"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()),
                              "model": SERVE_MODEL_ID, "choices": [{{"index": 0, "delta": {{}}, "finish_reason": None}}]}}
                yield f"data: {{json.dumps(keepalive)}}\\n\\n".encode("utf-8")
            try:
                up = task.result()
                data = json.loads(up.content.decode("utf-8"))
                if up.status_code >= 400 or "choices" not in data:
                    yield f"data: {{json.dumps(data)}}\\n\\n".encode("utf-8")
                    yield b"data: [DONE]\\n\\n"
                    return
                data = postprocess_completion(data, tools)
                for part in completion_to_sse(data):
                    yield part.encode("utf-8")
            except Exception as exc:
                yield f"data: {{json.dumps({{'error': {{'message': str(exc)}}}})}}\\n\\n".encode("utf-8")
                yield b"data: [DONE]\\n\\n"
        return StreamingResponse(
            buffered_sse(),
            headers={{
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }}
        )
    upstream = await client.send(req, stream=is_stream)
    if is_stream:
        return StreamingResponse(
            sse_stream_generator(upstream, keep_reasoning_separate=True),
            status_code=upstream.status_code,
            headers={{
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }}
        )
    else:
        resp_body = await upstream.aread()
        await upstream.aclose()
        try:
            data = json.loads(resp_body.decode("utf-8"))
            if BUFFER_TOOL_RESPONSES and path == "v1/chat/completions":
                data = postprocess_completion(data, tools)
            if "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0].get("message", {{}})
                reasoning = msg.get("reasoning") or msg.get("reasoning_content")
                if reasoning:
                    msg["reasoning_content"] = reasoning
                    if not msg.get("content") and not msg.get("tool_calls"):
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
