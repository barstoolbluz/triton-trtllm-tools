# TRT-LLM Model Conversion Tools

Self-contained Flox environment for converting HuggingFace models to TRT-LLM engine format. Engines built here are served by the TensorRT-LLM backend in the companion [triton-runtime](../triton-runtime/) environment.

- **tensorrt_llm**: 1.1.0
- **PyTorch**: 2.9.0a0
- **Python**: 3.12
- **TensorRT**: 10.13
- **CUDA**: 13 (bundled; coexists with host CUDA 12.x)
- **Platform**: Linux only (x86_64)

## Quick start

```bash
flox activate
trtllm-build --help
```

## Available tools

| Tool | Description |
|------|-------------|
| `trtllm-build` | Build TRT-LLM engines from checkpoints |
| `trtllm-bench` | Benchmark TRT-LLM engine performance |
| `trtllm-eval` | Evaluate model accuracy |
| `trtllm-prune` | Prune model weights |
| `trtllm-refit` | Refit engine weights without rebuilding |
| `trtllm-serve` | Launch a local TRT-LLM serving endpoint |
| `trtexec` | TensorRT engine profiling and validation |
| `trtllm-llmapi-launch` | Launch the TRT-LLM LLM API server |

All tools are wrapper scripts that set up `PYTHONHOME`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `CUDA_HOME`, `OPAL_PREFIX`, `CPATH`, and `TRITON_PTXAS_PATH` before delegating to the bundled Python 3.12 interpreter and libraries.

## Model conversion workflow

Converting a HuggingFace model to a TRT-LLM engine is a two-step process. This example uses Qwen2.5-0.5B.

### Step 1: HuggingFace to TRT-LLM checkpoint

Write a Python script that loads the HuggingFace model and exports a TRT-LLM checkpoint:

```python
#!/usr/bin/env python3
"""convert_qwen.py -- HuggingFace to TRT-LLM checkpoint."""
import sys
from tensorrt_llm.models import QWenForCausalLM

def main():
    hf_path = sys.argv[1]    # e.g. Qwen/Qwen2.5-0.5B
    ckpt_dir = sys.argv[2]   # output checkpoint directory

    model = QWenForCausalLM.from_hugging_face(hf_path, dtype="float16")
    model.save_checkpoint(ckpt_dir)
    print(f"Checkpoint saved to {ckpt_dir}")

# REQUIRED: MPI re-executes this script in worker processes.
# Without this guard the script will crash.
if __name__ == "__main__":
    main()
```

Run it inside the environment:

```bash
flox activate
python3 convert_qwen.py Qwen/Qwen2.5-0.5B /data/checkpoints/qwen2.5-0.5b
```

### Step 2: Build the TRT-LLM engine

```bash
trtllm-build \
  --checkpoint_dir /data/checkpoints/qwen2.5-0.5b \
  --output_dir /data/engines/qwen2.5-0.5b \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16
```

On an RTX 5090, Qwen2.5-0.5B builds in about 6 seconds and produces a ~1.2 GB engine.

## Serving converted models

Converted engines are served by the TensorRT-LLM backend in the [triton-runtime](../triton-runtime/) environment. Place the engine directory in the model repository following Triton's layout:

```
$TRITON_MODEL_REPOSITORY/
  qwen2.5-0.5b/
    config.pbtxt          # TRT-LLM backend config (50+ required params)
    1/
      model/              # engine output directory contents
```

The TRT-LLM backend requires a complete `config.pbtxt` with all parameters from the [NVIDIA template](https://github.com/NVIDIA/TensorRT-LLM/blob/main/triton_backend/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt). See the triton-runtime README for full serving instructions.

## Environment details

### Package architecture

The environment is split into five Nix sub-packages to stay under the Flox catalog's 5 GB NAR limit:

| Sub-package | Size | Contents |
|-------------|------|----------|
| `trtllm-tools-libs-cuda` | 2.9 GB | CUDA 13, cuDNN 9.14, NCCL, MPI, native shared libs |
| `trtllm-tools-libs-ml` | 3.5 GB | TensorRT 10.13, MKL, TBB |
| `trtllm-tools-python` | 4.4 GB | Python 3.12 interpreter + stdlib + ~290 packages |
| `trtllm-tools-engine` | 4.2 GB | PyTorch 2.9.0a0, tensorrt_llm 1.1.0, torchvision |
| `trtllm-tools` | 0.24 GB | Wrapper scripts, trtexec, cuda/, hpcx/ompi/ |

Total footprint: ~15 GB installed.

All five packages are extracted from the NGC container `nvcr.io/nvidia/tritonserver:26.02-trtllm-python-py3` and published to the Flox catalog.

### Python version

This environment uses Python 3.12. The triton-runtime environment uses Python 3.13. The `tensorrt_llm` package is not pip-installable on Python 3.13, which is why model conversion is a separate environment.

## Troubleshooting

### `if __name__` guard required

MPI re-executes your Python script in worker processes. Any top-level code that is not behind an `if __name__ == "__main__":` guard will run multiple times and likely crash. Always wrap your conversion logic in a `main()` function with the guard.

### flashinfer JIT compilation

Some TRT-LLM code paths trigger flashinfer JIT compilation, which requires `ninja` (already in the manifest) and CUDA headers under `CUDA_HOME/include`. The bundled `CUDA_HOME` does not include a full `include/` directory (`cuda_runtime.h` is absent), so flashinfer JIT will fail. Prefer using the `trtllm-build` CLI or the low-level Python checkpoint API instead of the `LLM()` high-level API.

### `LLM()` high-level API

The `tensorrt_llm.LLM()` high-level API defaults to the PyTorch backend, which requires flashinfer JIT compilation with full CUDA toolkit headers. For model conversion, use the two-step checkpoint + `trtllm-build` workflow documented above instead.

### `TORCH_CUDA_ARCH_LIST` warning

You may see a warning about `TORCH_CUDA_ARCH_LIST` when importing PyTorch. This is cosmetic and safe to ignore -- the bundled PyTorch was built with the correct architecture support.
