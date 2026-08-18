from pathlib import Path
import sys

PROJECT = Path(
    "/kaggle/working/LTX-13B-AI-Video-Model"
)

MODEL_FILES = {
    "LTX Q4": Path(
        "/kaggle/input/datasets/shihoos/ltx13b-q4/"
        "LTXV-13B-0.9.8-distilled-Q4_K_M.gguf"
    ),
    "T5 Q4": Path(
        "/kaggle/input/datasets/shihoos/ltx13b-t5/"
        "t5-v1_1-xxl-encoder-Q4_K_M.gguf"
    ),
    "VAE": Path(
        "/kaggle/input/datasets/shihoos/ltx13b-vae/"
        "LTXV-13B-0.9.8-distilled-VAE.safetensors"
    ),
    "IC-LoRA": Path(
        "/kaggle/input/datasets/shihoos/ltx13b-enhancers/"
        "ltxv-098-ic-lora-detailer-comfyui.safetensors"
    ),
    "Spatial upscaler": Path(
        "/kaggle/input/datasets/shihoos/ltx13b-enhancers/"
        "ltxv-spatial-upscaler-0.9.8.safetensors"
    ),
}


print("=" * 80)
print("LTX-13B MODERN STACK PREFLIGHT")
print("=" * 80)

if not PROJECT.exists():
    raise FileNotFoundError(
        f"Project not found: {PROJECT}"
    )

print(
    f"✅ Project: {PROJECT}"
)

print(
    f"✅ Python: {sys.version.split()[0]}"
)

import torch

print(
    f"✅ PyTorch: {torch.__version__}"
)

print(
    f"✅ CUDA build: {torch.version.cuda}"
)

print(
    f"CUDA available: "
    f"{torch.cuda.is_available()}"
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU is OFF. Turn Kaggle GPU ON."
    )

if not torch.__version__.startswith(
    "2.10.0+cu128"
):
    raise RuntimeError(
        "Unexpected Torch version.\n"
        f"Expected: 2.10.0+cu128\n"
        f"Found: {torch.__version__}"
    )

for index in range(
    torch.cuda.device_count()
):
    props = torch.cuda.get_device_properties(
        index
    )

    print(
        f"GPU {index}: "
        f"{props.name} | "
        f"{props.total_memory / (1024**3):.2f} GB"
    )

try:
    import psutil

    ram_gb = (
        psutil.virtual_memory()
        .total
        / (1024**3)
    )

    print(
        f"System RAM: {ram_gb:.2f} GB"
    )

    if ram_gb < 32:
        print(
            "⚠️ RAM below 32 GB; "
            "the tested detailer approached 28–29 GB."
        )

except Exception:
    print(
        "ℹ️ RAM check unavailable."
    )

for label, path in MODEL_FILES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{label} missing:\n{path}"
        )

    print(
        f"✅ {label}: "
        f"{path.name}"
    )

print()
print(
    "✅ PREFLIGHT PASSED"
)
