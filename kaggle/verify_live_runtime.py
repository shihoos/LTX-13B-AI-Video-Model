#!/usr/bin/env python3

"""
Verify the already-running ComfyUI runtime.

Canonical ComfyUI port:

    8188

This verifier checks:

1. ComfyUI HTTP readiness
2. /system_stats
3. /object_info
4. Required node registrations
5. Locked model filenames
6. Modern latent upscaler
7. Required legacy LTX compatibility loader
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

LOCK_FILE = (
    PROJECT_ROOT
    / "kaggle"
    / "compatibility_lock.yaml"
)

DEFAULT_URL = "http://127.0.0.1:8188"


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_lock() -> dict:

    try:
        import yaml
    except ImportError as error:
        fail(
            "PyYAML unavailable:\n"
            f"{error}"
        )

    if not LOCK_FILE.is_file():
        fail(
            f"Compatibility lock missing:\n{LOCK_FILE}"
        )

    try:
        data = yaml.safe_load(
            LOCK_FILE.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        fail(
            "Could not parse compatibility_lock.yaml:\n"
            f"{error}"
        )

    if not isinstance(data, dict):
        fail(
            "compatibility_lock.yaml must contain a mapping."
        )

    return data


def get_json(
    base_url: str,
    path: str,
    timeout: float,
) -> dict:

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        headers={
            "Accept": "application/json",
        },
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:

            payload = response.read()

    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
    ) as error:

        fail(
            f"ComfyUI request failed:\n"
            f"{base_url}{path}\n"
            f"{error}"
        )

    try:

        data = json.loads(
            payload.decode("utf-8")
        )

    except json.JSONDecodeError as error:

        fail(
            f"ComfyUI returned invalid JSON:\n"
            f"{path}\n"
            f"{error}"
        )

    if not isinstance(data, dict):
        fail(
            f"ComfyUI returned a non-object response:\n"
            f"{path}"
        )

    return data


def wait_for_comfy(
    base_url: str,
    timeout: float,
) -> None:

    deadline = (
        time.monotonic()
        + timeout
    )

    last_error = None

    while time.monotonic() < deadline:

        try:

            get_json(
                base_url,
                "/system_stats",
                5,
            )

            print(
                "OK   ComfyUI HTTP ready"
            )

            return

        except RuntimeError as error:

            last_error = error
            time.sleep(2)

    fail(
        "ComfyUI did not become ready.\n"
        f"URL: {base_url}\n"
        f"Last error: {last_error}"
    )


def choices(
    object_info: dict,
    node_name: str,
    input_name: str,
) -> list[str]:

    node = object_info.get(
        node_name
    )

    if not isinstance(node, dict):

        fail(
            "Live ComfyUI node is missing:\n"
            f"{node_name}"
        )

    input_data = node.get(
        "input",
        {},
    )

    if not isinstance(
        input_data,
        dict,
    ):
        fail(
            f"Invalid input schema for {node_name}."
        )

    required = input_data.get(
        "required",
        {},
    )

    if not isinstance(
        required,
        dict,
    ):
        fail(
            f"Invalid required-input schema for {node_name}."
        )

    spec = required.get(
        input_name
    )

    if not isinstance(
        spec,
        list,
    ) or not spec:

        fail(
            f"Live schema missing:\n"
            f"{node_name}.{input_name}"
        )

    options = spec[0]

    if not isinstance(
        options,
        list,
    ):

        fail(
            f"Live schema does not expose choices:\n"
            f"{node_name}.{input_name}"
        )

    return [
        str(value)
        for value in options
    ]


def require_choice(
    object_info: dict,
    node_name: str,
    input_name: str,
    expected: str,
) -> None:

    available = choices(
        object_info,
        node_name,
        input_name,
    )

    if expected not in available:

        fail(
            "Locked model is not discoverable:\n"
            f"Node: {node_name}\n"
            f"Input: {input_name}\n"
            f"Expected: {expected}\n"
            f"Available: {available}"
        )

    print(
        f"OK   {node_name}.{input_name}: {expected}"
    )


def verify_nodes(
    object_info: dict,
) -> None:

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

    available = set(
        object_info
    )

    missing = (
        required_nodes
        - available
    )

    if missing:

        fail(
            "Live ComfyUI is missing required nodes:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    print(
        f"OK   required live nodes: "
        f"{len(required_nodes)}"
    )


def verify_models(
    object_info: dict,
    lock: dict,
) -> None:

    models = lock[
        "models"
    ]

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


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Verify live LTX-13B ComfyUI runtime."
        )
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=(
            "ComfyUI URL. "
            "Default: http://127.0.0.1:8188"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
    )

    args = parser.parse_args()

    lock = load_lock()

    print("=" * 80)
    print("LTX-13B LIVE COMFYUI RUNTIME VERIFICATION")
    print("=" * 80)

    print(
        f"URL: {args.url}"
    )

    wait_for_comfy(
        args.url,
        args.timeout,
    )

    object_info = get_json(
        args.url,
        "/object_info",
        30,
    )

    verify_nodes(
        object_info
    )

    verify_models(
        object_info,
        lock,
    )

    print()
    print("=" * 80)
    print("✅ LIVE COMFYUI RUNTIME VERIFIED")
    print("=" * 80)

    print(
        "ComfyUI endpoint:",
        args.url,
    )

    print(
        "Locked models: discoverable"
    )

    print(
        "Modern latent upscaler: discoverable"
    )


if __name__ == "__main__":
    main()
