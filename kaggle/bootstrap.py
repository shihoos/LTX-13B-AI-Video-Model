from pathlib import Path
import subprocess
import sys
import os

PROJECT = Path("/kaggle/working/LTX-13B-AI-Video-Model")
COMFY = PROJECT / "ComfyUI"
CUSTOM = COMFY / "custom_nodes"

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"

NODES = {
    "ComfyUI-GGUF":
        "https://github.com/city96/ComfyUI-GGUF.git",

    "ComfyUI-LTXVideo":
        "https://github.com/Lightricks/ComfyUI-LTXVideo.git",

    "ComfyUI-VideoHelperSuite":
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",

    "rgthree-comfy":
        "https://github.com/rgthree/rgthree-comfy.git",
}

def run(cmd):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def clone_repo(url: str, destination: Path) -> None:
    if destination.exists():
        print("Already exists:", destination)
        return

    run([
        "git",
        "clone",
        "--depth",
        "1",
        url,
        str(destination),
    ])


def main():
    PROJECT.mkdir(parents=True, exist_ok=True)
    CUSTOM.mkdir(parents=True, exist_ok=True)

    # ComfyUI
    clone_repo(COMFY_REPO, COMFY)

    # Custom nodes
    for name, url in NODES.items():
        clone_repo(url, CUSTOM / name)

    # ComfyUI dependencies
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        str(COMFY / "requirements.txt"),
    ])

    # Node requirements
    for requirements in CUSTOM.glob("*/requirements.txt"):
        print("Installing:", requirements)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-r",
                str(requirements),
            ],
            check=False,
        )

    print("\n✅ Bootstrap completed.")


if __name__ == "__main__":
    main()
