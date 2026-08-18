from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = PROJECT_ROOT / "kaggle" / "compatibility_lock.yaml"


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_lock() -> dict:
    try:
        import yaml
    except ImportError as error:
        fail(f"PyYAML unavailable:\n{error}")
    if not LOCK_FILE.is_file():
        fail(f"Compatibility lock missing:\n{LOCK_FILE}")
    data = yaml.safe_load(LOCK_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail("compatibility_lock.yaml must contain a mapping.")
    return data


def get_json(base_url: str, path: str, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        fail(f"ComfyUI request failed for {path}:\n{error}")
    if not isinstance(data, dict):
        fail(f"ComfyUI returned a non-object response for {path}.")
    return data


def wait_for_comfy(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            get_json(base_url, "/system_stats", 5)
            return
        except RuntimeError as error:
            last_error = error
            time.sleep(2)
    fail(
        "ComfyUI did not become ready before the timeout.\n"
        f"URL: {base_url}\nLast error: {last_error}"
    )


def choices(object_info: dict, node_name: str, input_name: str) -> list[str]:
    node = object_info.get(node_name)
    if not isinstance(node, dict):
        fail(f"Live ComfyUI node is missing: {node_name}")
    required = node.get("input", {}).get("required", {})
    spec = required.get(input_name)
    if not isinstance(spec, list) or not spec:
        fail(f"Live node schema is missing {node_name}.{input_name}")
    options = spec[0]
    if not isinstance(options, list):
        fail(
            f"Live node schema changed: {node_name}.{input_name} "
            "does not expose model choices."
        )
    return [str(option) for option in options]


def require_choice(
    object_info: dict,
    node_name: str,
    input_name: str,
    expected: str,
) -> None:
    available = choices(object_info, node_name, input_name)
    if expected not in available:
        fail(
            f"Locked model is not discoverable by {node_name}.{input_name}.\n"
            f"Expected: {expected}\nAvailable: {available}"
        )
    print(f"OK   {node_name}.{input_name}: {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the live, already-started LTX/ComfyUI runtime."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8219")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    lock = load_lock()
    wait_for_comfy(args.url, args.timeout)
    object_info = get_json(args.url, "/object_info", 30)

    required_nodes = {
        "LTXVBaseSampler",
        "LTXVConditioning",
        "STGGuiderAdvanced",
        "FloatToSigmas",
        "StringToFloatList",
        "UnetLoaderGGUF",
        "CLIPLoaderGGUF",
        "VAELoader",
        "Set VAE Decoder Noise",
        "VHS_VideoCombine",
        "VHS_LoadVideo",
        "LTXVLoopingSampler",
        "LTXVLatentUpsampler",
        "LTXVLatentUpsamplerModelLoader",
        "LTXVTiledVAEDecode",
        "LTXVFilmGrain",
        "LoraLoaderModelOnly",
        "LatentUpscaleModelLoader",
    }
    missing = sorted(required_nodes - set(object_info))
    if missing:
        fail("Live ComfyUI is missing required nodes:\n" + "\n".join(missing))
    print(f"OK   {len(required_nodes)} required live nodes")

    models = lock["models"]
    require_choice(
        object_info,
        "UnetLoaderGGUF",
        "unet_name",
        models["ltx_q4"]["filename"],
    )
    require_choice(
        object_info,
        "CLIPLoaderGGUF",
        "clip_name",
        models["t5_q4"]["filename"],
    )
    require_choice(
        object_info,
        "VAELoader",
        "vae_name",
        models["vae"]["filename"],
    )
    require_choice(
        object_info,
        "LoraLoaderModelOnly",
        "lora_name",
        models["ic_lora"]["filename"],
    )
    require_choice(
        object_info,
        "LatentUpscaleModelLoader",
        "model_name",
        models["spatial_upscaler"]["filename"],
    )
    print("✅ LIVE COMFYUI RUNTIME VERIFIED")


if __name__ == "__main__":
    main()
