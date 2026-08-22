from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import (
    Request,
    urlopen,
)

from planner.config import (
    H3_API_BASE,
    H3_API_KEY,
    H3_REGENERATE_ENDPOINT,
)


class H3Regenerate2K:

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str = H3_API_BASE,
        endpoint: str = H3_REGENERATE_ENDPOINT,
    ):

        self.api_key = (
            api_key
            or H3_API_KEY
        )

        self.api_base = (
            api_base.rstrip("/")
        )

        self.endpoint = endpoint

    def _request(
        self,
        payload,
    ):

        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY is not configured."
            )

        data = json.dumps(
            payload
        ).encode(
            "utf-8"
        )

        request = Request(
            self.api_base
            + self.endpoint,
            data=data,
            method="POST",
            headers={
                "Content-Type":
                    "application/json",

                "Authorization":
                    f"Bearer {self.api_key}",
            },
        )

        try:
            with urlopen(
                request,
                timeout=300,
            ) as response:

                body = response.read()

                return json.loads(
                    body.decode(
                        "utf-8"
                    )
                )

        except HTTPError as error:

            body = (
                error.read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )

            raise RuntimeError(
                "H3 Regenerate-2K API failed: "
                f"HTTP {error.code}: {body}"
            ) from error

    @staticmethod
    def _data_url(
        path: Path,
    ):

        mime = "video/mp4"

        encoded = base64.b64encode(
            path.read_bytes()
        ).decode(
            "ascii"
        )

        return (
            f"data:{mime};base64,"
            f"{encoded}"
        )

    def regenerate(
        self,
        source_video: Path,
        destination: Path,
        prompt: str | None = None,
    ) -> Path:

        source_video = Path(
            source_video
        )

        destination = Path(
            destination
        )

        if not source_video.is_file():
            raise FileNotFoundError(
                source_video
            )

        payload = {
            "base_video": self._data_url(
                source_video
            ),
        }

        if prompt:
            payload[
                "prompt"
            ] = prompt

        result = self._request(
            payload
        )

        # API response schemas can change.
        # Keep extraction deliberately defensive.
        video_url = (
            result.get("video_url")
            or result.get("url")
            or result.get("output_url")
        )

        if not video_url:

            data = result.get(
                "data"
            )

            if isinstance(
                data,
                dict,
            ):
                video_url = (
                    data.get(
                        "video_url"
                    )
                    or data.get(
                        "url"
                    )
                )

        if not video_url:
            raise RuntimeError(
                "H3 Regenerate-2K did not "
                "return a video URL:\n"
                + json.dumps(
                    result,
                    indent=2,
                )
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        request = Request(
            video_url,
            method="GET",
        )

        with urlopen(
            request,
            timeout=600,
        ) as response:

            destination.write_bytes(
                response.read()
            )

        if (
            not destination.is_file()
            or destination.stat().st_size <= 0
        ):
            raise RuntimeError(
                "H3 2K result is empty."
            )

        return destination
