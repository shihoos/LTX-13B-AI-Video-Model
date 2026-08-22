from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT / "ComfyUI"


REQUIRED_FILES = [
    COMFY / "main.py",
    COMFY / "custom_nodes" / "ComfyUI-GGUF",
    COMFY / "custom_nodes" / "ComfyUI-H3-Multishot",
    COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite",
    COMFY / "models" / "diffusion_models" / "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
    COMFY / "models" / "text_encoders" / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    COMFY / "models" / "vae" / "minimax_h3_video_vae_fp16.safetensors",
    COMFY / "models" / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
]


def main() -> int:
    print("=" * 72)
    print("MINIMAX H3 REF2VA PREFLIGHT")
    print("=" * 72)

    failed = False

    for path in REQUIRED_FILES:
        ok = path.exists()
        print(("OK   " if ok else "FAIL "), path)

        if not ok:
            failed = True

    if failed:
        print("\nPreflight failed.")
        return 2

    # Import the native H3 node modules without starting generation.
    sys.path.insert(0, str(COMFY))

    try:
        import comfy_extras.nodes_minimax_h3  # noqa: F401
    except Exception as exc:
        print("\nNative MiniMax H3 import failed:")
        print(exc)
        return 3

    print("\nPASS — H3 runtime files and native H3 support are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
