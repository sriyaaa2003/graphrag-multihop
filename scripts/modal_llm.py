"""Serve an open instruct model with vLLM on a Modal GPU, behind an OpenAI-compatible endpoint.

    modal deploy scripts/modal_llm.py        # prints the URL
    modal app stop graphhop-llm              # when finished

The endpoint is used by the LLM extractor (`GH_LLM_BASE_URL=<url>/v1`, provider `openai_compat`).
"""

from __future__ import annotations

import subprocess

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.5", "transformers==4.51.3", "huggingface_hub")
)
cache = modal.Volume.from_name("graphhop-hf-cache", create_if_missing=True)
app = modal.App("graphhop-llm", image=image)


@app.function(
    gpu="A10G",
    volumes={"/root/.cache/huggingface": cache},
    scaledown_window=300,
    timeout=3600,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=900)
def serve() -> None:
    subprocess.Popen(
        [
            "vllm", "serve", MODEL,
            "--host", "0.0.0.0", "--port", str(PORT),
            "--max-model-len", "4096",
            "--served-model-name", "extractor",
            "--gpu-memory-utilization", "0.92",
        ]
    )
