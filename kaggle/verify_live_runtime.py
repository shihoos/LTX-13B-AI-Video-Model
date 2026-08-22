from __future__ import annotations

import json
import urllib.request


BASE_URL = "http://127.0.0.1:8188"


def get(path: str):
    with urllib.request.urlopen(
        BASE_URL + path,
        timeout=60,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    stats = get("/system_stats")
    objects = get("/object_info")

    required = [
        "H3ModelLoaderAny",
        "H3ClipLoaderAny",
        "H3MultishotMemorySampler",
        "H3ReferenceAudio",
    ]

    missing = [
        name for name in required
        if name not in objects
    ]

    print("ComfyUI healthy:", bool(stats))
    print("Required H3 nodes:")
    for name in required:
        print(
            ("OK   " if name not in missing else "FAIL "),
            name,
        )

    if missing:
        raise SystemExit(
            "H3 runtime is incomplete."
        )

    print("\nH3 Ref2VA runtime is live.")


if __name__ == "__main__":
    main()
