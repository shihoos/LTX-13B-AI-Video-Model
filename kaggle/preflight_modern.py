from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT = Path(
    os.getenv(
        "LTX_PROJECT_ROOT",
        "/kaggle/working/LTX-13B-AI-Video-Model",
    )
)

LOCK_FILE = (
    PROJECT
    / "kaggle"
    / "compatibility_lock.yaml"
)


def load_lock():

    try:
        import yaml

    except ImportError as error:

        raise RuntimeError(
            "PyYAML is required to read "
            "compatibility_lock.yaml."
        ) from error

    if not LOCK_FILE.exists():

        raise FileNotFoundError(
            "Compatibility lock not found:\n"
            f"{LOCK_FILE}"
        )

    with LOCK_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):

        raise RuntimeError(
            "compatibility_lock.yaml does not contain "
            "a valid mapping."
        )

    return data


def verify_model_sources(
    lock,
):

    models = lock[
        "models"
    ]

    for name, spec in (
        models.items()
    ):

        dataset = Path(
            spec[
                "dataset"
            ]
        )

        filename = spec[
            "filename"
        ]

        path = (
            dataset
            / filename
        )

        if not path.exists():

            raise FileNotFoundError(
                f"{name} model missing:\n"
                f"{path}"
            )

        print(
            f"✅ {name}: {filename}"
        )


def verify_gpu(
    lock,
):

    import torch

    runtime = lock[
        "python_runtime"
    ]

    expected_torch = runtime[
        "torch"
    ]

    # Convert PyTorch build tag such as:
    #
    #     2.10.0+cu128
    #
    # into:
    #
    #     12.8
    #
    # so it can be compared with:
    #
    #     torch.version.cuda -> "12.8"

    expected_cuda = None

    if "+cu" in expected_torch:

        cuda_tag = (
            expected_torch
            .split(
                "+cu",
                1,
            )[1]
        )

        if len(cuda_tag) >= 3:

            expected_cuda = (
                f"{cuda_tag[:-1]}."
                f"{cuda_tag[-1]}"
            )

    print(
        f"✅ PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"✅ CUDA build: "
        f"{torch.version.cuda}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "GPU is OFF. "
            "Turn the Kaggle GPU ON."
        )

    if (
        torch.__version__
        != expected_torch
    ):

        raise RuntimeError(
            "PyTorch version mismatch.\n"
            f"Expected from lock: "
            f"{expected_torch}\n"
            f"Actual: "
            f"{torch.__version__}"
        )

    if (
        expected_cuda is not None
        and torch.version.cuda is not None
        and not torch.version.cuda.startswith(
            expected_cuda
        )
    ):

        raise RuntimeError(
            "CUDA runtime mismatch.\n"
            f"Expected family: "
            f"CUDA {expected_cuda}\n"
            f"Actual: CUDA "
            f"{torch.version.cuda}"
        )

    expected_torchvision = runtime[
        "torchvision"
    ]

    try:

        import torchvision

    except Exception as error:

        raise RuntimeError(
            "Could not import torchvision.\n"
            f"{error}"
        ) from error

    if (
        torchvision.__version__
        != expected_torchvision
    ):

        raise RuntimeError(
            "torchvision version mismatch.\n"
            f"Expected from lock: "
            f"{expected_torchvision}\n"
            f"Actual: "
            f"{torchvision.__version__}"
        )

    print(
        f"✅ torchvision: "
        f"{torchvision.__version__}"
    )

    gpu_count = (
        torch.cuda.device_count()
    )

    if gpu_count == 0:

        raise RuntimeError(
            "No CUDA GPU detected."
        )

    for index in range(
        gpu_count
    ):

        props = (
            torch.cuda
            .get_device_properties(
                index
            )
        )

        print(
            f"GPU {index}: "
            f"{props.name} | "
            f"{props.total_memory / (1024**3):.2f} GB"
        )


def verify_project():

    if not PROJECT.exists():

        raise FileNotFoundError(
            f"Project not found:\n"
            f"{PROJECT}"
        )

    print(
        f"✅ Project: {PROJECT}"
    )


def main():

    print(
        "=" * 80
    )

    print(
        "LTX-13B MODERN STACK PREFLIGHT"
    )

    print(
        "=" * 80
    )

    lock = load_lock()

    verify_project()

    print(
        f"✅ Python: "
        f"{sys.version.split()[0]}"
    )

    # Pre-bootstrap checks only.
    #
    # These must validate the environment that already exists
    # before bootstrap.py installs the exact locked packages.

    verify_gpu(
        lock
    )

    verify_model_sources(
        lock
    )

    try:

        import psutil

        ram_gb = (
            psutil.virtual_memory()
            .total
            / (1024**3)
        )

        print(
            f"✅ System RAM: "
            f"{ram_gb:.2f} GB"
        )

        if ram_gb < 32:

            print(
                "⚠️ RAM below 32 GB. "
                "The tested detailer used approximately "
                "28–29 GB."
            )

    except Exception:

        print(
            "ℹ️ RAM information unavailable."
        )

    print()

    print(
        "✅ PREFLIGHT PASSED"
    )

    print(
        "Pre-bootstrap GPU/runtime checks passed."
    )

    print(
        "Exact package installation and "
        "post-install verification are handled "
        "by bootstrap.py."
    )


if __name__ == "__main__":

    main()
