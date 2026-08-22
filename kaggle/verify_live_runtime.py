from __future__ import annotations

import json
import urllib.request


BASE_URL = (
    "http://127.0.0.1:8188"
)


REQUIRED_NODES = [
    "H3ModelLoaderAny",
    "H3ClipLoaderAny",
    "MiniMaxH3ReferenceToVideo",
    "H3FreeTextEncoder",
    "H3ReferenceAudio",
    "H3MultishotMemorySampler",
    "VAEDecode",
    "VAEDecodeAudio",
    "SamplerCustomAdvanced",
    "CreateVideo",
    "SaveVideo",
]


def get(
    path: str,
):

    with urllib.request.urlopen(
        BASE_URL + path,
        timeout=60,
    ) as response:

        return json.loads(
            response
            .read()
            .decode("utf-8")
        )


def main():

    stats = get(
        "/system_stats"
    )

    objects = get(
        "/object_info"
    )

    missing = [
        name
        for name
        in REQUIRED_NODES
        if name not in objects
    ]

    print(
        "ComfyUI healthy:",
        bool(stats),
    )

    print(
        "\nH3 runtime nodes:"
    )

    for name in REQUIRED_NODES:

        print(
            (
                "OK   "
                if name not in missing
                else "FAIL "
            ),
            name,
        )

    if missing:

        raise SystemExit(
            "Missing H3 runtime nodes:\n"
            + "\n".join(
                missing
            )
        )

    print(
        "\nH3 Ref2VA runtime is ready."
    )


if __name__ == "__main__":
    main()
