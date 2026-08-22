from __future__ import annotations

import json
import urllib.request


def get(
    base_url: str,
    endpoint: str,
):

    with urllib.request.urlopen(
        base_url + endpoint,
        timeout=60,
    ) as response:

        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


REQUIRED = [
    "H3ModelLoaderAny",
    "H3ClipLoaderAny",
    "MiniMaxH3ReferenceToVideo",
    "H3FreeTextEncoder",
    "H3MultishotMemorySampler",
    "VAEDecode",
    "VAEDecodeAudio",
    "CreateVideo",
    "SaveVideo",
]


def main():

    for port in (
        8188,
        8189,
    ):

        base = (
            f"http://127.0.0.1:{port}"
        )

        print(
            f"\n=== WORKER {port} ==="
        )

        objects = get(
            base,
            "/object_info",
        )

        missing = [
            name
            for name in REQUIRED
            if name not in objects
        ]

        for name in REQUIRED:

            print(
                (
                    "OK   "
                    if name not in missing
                    else "FAIL "
                ),
                name,
            )

        if missing:
            print(
                "Missing:",
                ", ".join(missing),
            )
        else:
            print(
                "Worker ready."
            )


if __name__ == "__main__":
    main()
