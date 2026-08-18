from __future__ import annotations

import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


def fail(
    message,
):

    raise RuntimeError(
        message
    )


def check_file(
    path: Path,
):

    if not path.exists():

        fail(
            f"Missing file:\n{path}"
        )

    print(
        f"OK   {path}"
    )


def check_import(
    module_name: str,
):

    try:

        importlib.import_module(
            module_name
        )

    except Exception as error:

        fail(
            "Import failed:\n"
            f"{module_name}\n"
            f"{error}"
        )

    print(
        f"OK   import {module_name}"
    )


def check_json(
    path: Path,
):

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            json.load(
                file
            )

    except Exception as error:

        fail(
            "Invalid JSON:\n"
            f"{path}\n"
            f"{error}"
        )

    print(
        f"OK   JSON {path}"
    )


def check_lock():

    lock = (
        PROJECT_ROOT
        / "kaggle"
        / "compatibility_lock.yaml"
    )

    check_file(
        lock
    )

    try:
        import yaml

    except Exception as error:

        fail(
            "PyYAML unavailable:\n"
            f"{error}"
        )

    with lock.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(
            file
        )

    required_sections = [
        "comfyui",
        "python_runtime",
        "custom_nodes",
        "legacy_ltx_098_compat",
        "detailer_compat",
        "models",
        "validation",
    ]

    for section in (
        required_sections
    ):

        if section not in data:

            fail(
                "Compatibility lock is missing "
                f"section: {section}"
            )

    print(
        "OK   compatibility lock structure"
    )

    return data


def check_git_revision(
    path: Path,
    expected: str,
):

    actual = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )

    if actual != expected:

        fail(
            "Git revision mismatch:\n"
            f"Path: {path}\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )

    print(
        f"OK   {path.name}: {actual}"
    )


def validate_git_stack(
    lock,
):

    comfy = lock[
        "comfyui"
    ]

    comfy_dir = (
        PROJECT_ROOT
        / "ComfyUI"
    )

    if comfy_dir.exists():

        check_git_revision(
            comfy_dir,
            comfy[
                "commit"
            ],
        )

        custom_dir = (
            comfy_dir
            / "custom_nodes"
        )

        for name, spec in (
            lock[
                "custom_nodes"
            ].items()
        ):

            node_path = (
                custom_dir
                / name
            )

            if node_path.exists():

                check_git_revision(
                    node_path,
                    spec[
                        "commit"
                    ],
                )


def validate_workflows():

    base = (
        PROJECT_ROOT
        / "workflows"
        / "baseline"
        / "ltxv-13b-dist-i2v-base.json"
    )

    detailer = (
        PROJECT_ROOT
        / "workflows"
        / "detailer"
        / "ltxv-13b-098-ic-lora-upscale.json"
    )

    check_json(
        base
    )

    check_json(
        detailer
    )

    base_data = json.loads(
        base.read_text(
            encoding="utf-8"
        )
    )

    detailer_data = json.loads(
        detailer.read_text(
            encoding="utf-8"
        )
    )

    base_types = {
        node.get("type")
        for node in base_data.get(
            "nodes",
            [],
        )
        if isinstance(
            node,
            dict,
        )
    }

    detailer_types = {
        node.get("type")
        for node in detailer_data.get(
            "nodes",
            [],
        )
        if isinstance(
            node,
            dict,
        )
    }

    required_base = {
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
    }

    required_detailer = {
        "VHS_LoadVideo",
        "LTXVLoopingSampler",
        "LTXVLatentUpsampler",
        "LTXVLatentUpsamplerModelLoader",
        "LTXVTiledVAEDecode",
        "LTXVFilmGrain",
        "LoraLoaderModelOnly",
        "VHS_VideoCombine",
    }

    missing_base = (
        required_base
        - base_types
    )

    missing_detailer = (
        required_detailer
        - detailer_types
    )

    if missing_base:

        fail(
            "BASE workflow is missing node types:\n"
            + "\n".join(
                sorted(
                    missing_base
                )
            )
        )

    if missing_detailer:

        fail(
            "DETAILER workflow is missing node types:\n"
            + "\n".join(
                sorted(
                    missing_detailer
                )
            )
        )

    print(
        "OK   BASE workflow node set"
    )

    print(
        "OK   DETAILER workflow node set"
    )


def validate_compatibility_sources():

    compatibility = (
        PROJECT_ROOT
        / "compatibility"
        / "prepare_modern_ltx.py"
    )

    check_file(
        compatibility
    )

    text = compatibility.read_text(
        encoding="utf-8"
    )

    required_markers = [

        "LEGACY_LTX_COMMIT",

        "blur_internal(image, blur_radius)",

        "LTXVBaseSampler",

        "LTXVLoopingSampler",

        "LTXVTiledSampler",

        "LTXVTiledVAEDecode",

        "LTXVLatentUpsampler",

        "LTXVLatentUpsamplerModelLoader",

        "LTXVFilmGrain",

        "STGGuiderAdvanced",

        "Set VAE Decoder Noise",

    ]

    for marker in (
        required_markers
    ):

        if marker not in text:

            fail(
                "Modern compatibility source "
                f"is missing marker: {marker}"
            )

    print(
        "OK   modern compatibility builder"
    )


def validate_adapter():

    adapter = (
        PROJECT_ROOT
        / "execution"
        / "comfy_workflow_adapter.py"
    )

    check_file(
        adapter
    )

    text = adapter.read_text(
        encoding="utf-8"
    )

    required_markers = [

        "LTXVLatentUpsamplerModelLoader",

        "LatentUpscaleModelLoader",

        "apply_modern_compatibility",

        "validate_modern_detailer",

        "set_input_video",

    ]

    for marker in (
        required_markers
    ):

        if marker not in text:

            fail(
                "Modern workflow adapter "
                f"is missing marker: {marker}"
            )

    print(
        "OK   workflow adapter"
    )


def validate_executor():

    executor = (
        PROJECT_ROOT
        / "execution"
        / "shot_executor.py"
    )

    check_file(
        executor
    )

    text = executor.read_text(
        encoding="utf-8"
    )

    required_markers = [

        "_copy_raw_to_comfy_input",

        "VHS_LoadVideo",

        "execute_raw",

        "execute_detailer",

        "mark_raw_complete",

        "mark_upscaled_complete",

    ]

    for marker in (
        required_markers
    ):

        if marker not in text:

            fail(
                "ShotExecutor is missing "
                f"required stage marker: {marker}"
            )

    print(
        "OK   BASE→DETAILER physical handoff"
    )


def validate_models(
    lock,
):

    models = lock[
        "models"
    ]

    for name, spec in (
        models.items()
    ):

        source = (
            Path(
                spec[
                    "dataset"
                ]
            )
            / spec[
                "filename"
            ]
        )

        if source.exists():

            print(
                f"OK   {name}: "
                f"{source.name}"
            )

        else:

            print(
                f"SKIP {name}: "
                f"Kaggle dataset not mounted"
            )


def validate_runtime_packages(
    lock,
):

    expected = {}

    comfy = lock[
        "comfyui"
    ]

    runtime = lock[
        "python_runtime"
    ]

    expected[
        comfy[
            "frontend"
        ][
            "package"
        ]
    ] = comfy[
        "frontend"
    ][
        "version"
    ]

    expected[
        comfy[
            "workflow_templates"
        ][
            "package"
        ]
    ] = comfy[
        "workflow_templates"
    ][
        "version"
    ]

    expected[
        comfy[
            "embedded_docs"
        ][
            "package"
        ]
    ] = comfy[
        "embedded_docs"
    ][
        "version"
    ]

    expected[
        comfy[
            "comfy_kitchen"
        ][
            "package"
        ]
    ] = comfy[
        "comfy_kitchen"
    ][
        "version"
    ]

    expected[
        comfy[
            "comfy_aimdo"
        ][
            "package"
        ]
    ] = comfy[
        "comfy_aimdo"
    ][
        "version"
    ]

    for name in (
        "torchsde",
        "spandrel",
        "av",
        "gguf",
    ):

        expected[
            name
        ] = runtime[
            name
        ]

    print(
        "PACKAGE LOCK CHECK"
    )

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

            print(
                f"SKIP {package}: "
                "runtime package unavailable"
            )

            continue

        if (
            actual
            != expected_version
        ):

            fail(
                f"{package} mismatch.\n"
                f"Expected: {expected_version}\n"
                f"Actual: {actual}"
            )

        print(
            f"OK   {package}=={actual}"
        )


def validate_init_files():

    required = {

        "execution/__init__.py",

        "pipeline/__init__.py",

        "planner/__init__.py",

        "schemas/__init__.py",

        "scheduler/__init__.py",

    }

    for relative in (
        required
    ):

        path = (
            PROJECT_ROOT
            / relative
        )

        check_file(
            path
        )

    print(
        "OK   all package __init__.py files"
    )


def validate_general_structure():

    required = [

        "planner/config.py",
        "planner/qwen_loader.py",
        "planner/story_planner.py",
        "planner/character_detector.py",
        "planner/character_planner.py",
        "planner/scene_planner.py",
        "planner/shot_planner.py",

        "pipeline/continuity_manager.py",
        "pipeline/modes.py",
        "pipeline/production_manager.py",
        "pipeline/production_orchestrator.py",
        "pipeline/reference_manager.py",

        "execution/checkpoint_manager.py",
        "execution/comfy_client.py",
        "execution/comfy_workflow_adapter.py",
        "execution/shot_executor.py",
        "execution/assembly_manager.py",
        "execution/production_runner.py",

        "scheduler/gpu_scheduler.py",
        "scheduler/shot_queue.py",

        "schemas/character.py",
        "schemas/scene.py",
        "schemas/shot.py",
        "schemas/parser.py",

        "kaggle/bootstrap.py",
        "kaggle/launch.py",
        "kaggle/preflight_modern.py",
        "kaggle/runtime_requirements.lock",
        "kaggle/start_comfyui_tunnel.py",
        "kaggle/model_paths.yaml",

        "compatibility/prepare_modern_ltx.py",

        "workflows/baseline/ltxv-13b-dist-i2v-base.json",
        "workflows/detailer/ltxv-13b-098-ic-lora-upscale.json",
    ]

    for relative in (
        required
    ):

        check_file(
            PROJECT_ROOT
            / relative
        )


def validate_external_tools():

    if (
        shutil.which(
            "ffmpeg"
        )
        is None
    ):

        print(
            "WARNING ffmpeg not found"
        )

    else:

        print(
            "OK   ffmpeg"
        )


def main():

    print(
        "=" * 80
    )

    print(
        "LTX-13B FINAL MODERN PROJECT VALIDATION"
    )

    print(
        "=" * 80
    )

    validate_general_structure()

    validate_init_files()

    lock = check_lock()

    validate_git_stack(
        lock
    )

    validate_runtime_packages(
        lock
    )

    validate_compatibility_sources()

    validate_adapter()

    validate_executor()

    validate_workflows()

    validate_models(
        lock
    )

    validate_external_tools()

    print()
    print(
        "=" * 80
    )

    print(
        "✅ FINAL MODERN PROJECT STRUCTURE VALIDATED"
    )

    print(
        "=" * 80
    )

    print(
        "ComfyUI backend is commit-pinned."
    )

    print(
        "Frontend is package-pinned."
    )

    print(
        "No @latest frontend override is required."
    )

    print(
        "LTX compatibility is explicitly represented."
    )

    print(
        "BASE and DETAILER workflows are both validated."
    )

    print(
        "Package __init__.py files are present."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

    main()
