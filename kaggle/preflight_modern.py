from __future__ import annotations

import importlib.metadata
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

    expected_cuda = None

    if "+cu" in expected_torch:
        cuda_tag = expected_torch.split(
            "+cu", 1
        )[1]

        if len(cuda_tag) >= 3:
            expected_cuda = (
                f"{cuda_tag[:-1]}."
                f"{cuda_tag[-1]}"
            )

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
            "GPU is OFF. "
            "Turn the Kaggle GPU ON."
        )

    if (
        torch.__version__
        != expected_torch
    ):

        raise RuntimeError(
            "PyTorch version mismatch.\n"
            f"Expected from lock: {expected_torch}\n"
            f"Actual: {torch.__version__}"
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
            f"Expected family: CUDA {expected_cuda}\n"
            f"Actual: CUDA {torch.version.cuda}"
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

    for index in range(
        torch.cuda.device_count()
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

    if torch.cuda.device_count() == 0:

        raise RuntimeError(
            "No CUDA GPU detected."
        )


def verify_locked_packages(
    lock,
):

    comfy = lock[
        "comfyui"
    ]

    runtime = lock[
        "python_runtime"
    ]

    expected = {

        comfy[
            "frontend"
        ][
            "package"
        ]:
            comfy[
                "frontend"
            ][
                "version"
            ],

        comfy[
            "workflow_templates"
        ][
            "package"
        ]:
            comfy[
                "workflow_templates"
            ][
                "version"
            ],

        comfy[
            "embedded_docs"
        ][
            "package"
        ]:
            comfy[
                "embedded_docs"
            ][
                "version"
            ],

        comfy[
            "comfy_kitchen"
        ][
            "package"
        ]:
            comfy[
                "comfy_kitchen"
            ][
                "version"
            ],

        comfy[
            "comfy_aimdo"
        ][
            "package"
        ]:
            comfy[
                "comfy_aimdo"
            ][
                "version"
            ],

        "torchsde":
            runtime[
                "torchsde"
            ],

        "spandrel":
            runtime[
                "spandrel"
            ],

        "av":
            runtime[
                "av"
            ],

        "gguf":
            runtime[
                "gguf"
            ],
    }

    for (
        package,
        expected_version,
    ) in expected.items():

        try:

            actual = (
                importlib.metadata
                .version(
                    package
                )
            )

        except importlib.metadata.PackageNotFoundError:

            raise RuntimeError(
                "Missing locked package:\n"
                f"{package}=={expected_version}"
            )

        if (
            actual
            != expected_version
        ):

            raise RuntimeError(
                f"{package} version mismatch.\n"
                f"Expected: {expected_version}\n"
                f"Actual:   {actual}"
            )

        print(
            f"✅ {package}=={actual}"
        )


def verify_project(
):

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

    verify_gpu(
        lock
    )

    verify_locked_packages(
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
        "All runtime versions are sourced "
        "from compatibility_lock.yaml."
    )

    print(
        "No duplicated ComfyUI/frontend version "
        "pins are used by this preflight."
    )


if __name__ == "__main__":

    main()
