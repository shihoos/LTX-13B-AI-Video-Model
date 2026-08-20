from __future__ import annotations

import json
import time
import uuid

from pathlib import Path
from urllib.error import (
    HTTPError,
    URLError,
)
from urllib.parse import (
    urlencode,
)
from urllib.request import (
    Request,
    urlopen,
)


# ============================================================================
# TRANSIENT HTTP ERRORS
# ============================================================================

TRANSIENT_HTTP_STATUS_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}


# ============================================================================
# COMFYUI CLIENT
# ============================================================================

class ComfyClient:

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        request_retries: int = 3,
    ):

        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout = timeout

        self.request_retries = max(
            0,
            request_retries,
        )

    # ========================================================================
    # HTTP REQUEST
    # ========================================================================

    def _request(
        self,
        method: str,
        path: str,
        payload=None,
        retry: bool = False,
    ):

        url = (
            self.base_url
            + path
        )

        data = None

        headers = {}

        if payload is not None:

            data = json.dumps(
                payload
            ).encode(
                "utf-8"
            )

            headers[
                "Content-Type"
            ] = "application/json"

        attempts = (
            self.request_retries + 1
            if retry
            else 1
        )

        for attempt in range(
            attempts
        ):

            request = Request(
                url=url,
                method=method,
                data=data,
                headers=headers,
            )

            try:

                with urlopen(
                    request,
                    timeout=self.timeout,
                ) as response:

                    body = (
                        response.read()
                    )

                    if not body:

                        return None

                    content_type = (
                        response.headers.get(
                            "Content-Type",
                            "",
                        )
                    )

                    if (
                        "application/json"
                        in content_type
                    ):

                        return json.loads(
                            body.decode(
                                "utf-8"
                            )
                        )

                    return body

            except HTTPError as error:

                body = error.read().decode(
                    "utf-8",
                    errors="replace",
                )

                if (
                    retry
                    and
                    error.code
                    in TRANSIENT_HTTP_STATUS_CODES
                    and
                    attempt
                    < attempts - 1
                ):

                    self._sleep_before_retry(
                        attempt
                    )

                    continue

                raise RuntimeError(
                    f"ComfyUI HTTP "
                    f"{error.code}: "
                    f"{body}"
                ) from error

            except URLError as error:

                if (
                    retry
                    and
                    attempt
                    < attempts - 1
                ):

                    self._sleep_before_retry(
                        attempt
                    )

                    continue

                raise RuntimeError(
                    "Cannot connect to ComfyUI "
                    f"at {self.base_url}: "
                    f"{error}"
                ) from error

            except TimeoutError as error:

                if (
                    retry
                    and
                    attempt
                    < attempts - 1
                ):

                    self._sleep_before_retry(
                        attempt
                    )

                    continue

                raise RuntimeError(
                    "ComfyUI request timed out "
                    f"at {self.base_url}: "
                    f"{error}"
                ) from error

    # ========================================================================
    # RETRY BACKOFF
    # ========================================================================

    @staticmethod
    def _sleep_before_retry(
        attempt: int,
    ) -> None:

        delay = min(
            1.0 * (2 ** attempt),
            10.0,
        )

        time.sleep(
            delay
        )

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    def health_check(
        self,
    ) -> bool:

        try:

            self._request(
                "GET",
                "/system_stats",
                retry=True,
            )

            return True

        except Exception:

            return False

    # ========================================================================
    # QUEUE PROMPT
    # ========================================================================

    def queue_prompt(
        self,
        workflow: dict,
        client_id: str | None = None,
    ) -> str:

        if client_id is None:

            client_id = str(
                uuid.uuid4()
            )

        payload = {
            "prompt": workflow,
            "client_id": client_id,
        }

        # IMPORTANT:
        #
        # Do NOT retry POST /prompt automatically.
        #
        # A retry could submit the same generation twice if ComfyUI
        # accepted the request but the connection failed before the
        # response reached this client.

        result = self._request(
            "POST",
            "/prompt",
            payload,
            retry=False,
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "ComfyUI returned an invalid "
                "response to /prompt."
            )

        prompt_id = (
            result.get(
                "prompt_id"
            )
        )

        if not prompt_id:

            raise RuntimeError(
                "ComfyUI did not return "
                "a prompt_id."
            )

        return prompt_id

    # ========================================================================
    # HISTORY
    # ========================================================================

    def get_history(
        self,
        prompt_id: str,
    ) -> dict:

        result = self._request(
            "GET",
            f"/history/{prompt_id}",
            retry=True,
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "ComfyUI returned an invalid "
                "history response."
            )

        return result

    # ========================================================================
    # WAIT FOR PROMPT
    # ========================================================================

    def wait_for_prompt(
        self,
        prompt_id: str,
        poll_interval: float = 2.0,
        timeout: float = 3600.0,
    ) -> dict:

        started = time.monotonic()

        current_interval = max(
            0.5,
            poll_interval,
        )

        max_poll_interval = 10.0

        while True:

            elapsed = (
                time.monotonic()
                - started
            )

            if elapsed > timeout:

                raise TimeoutError(
                    f"ComfyUI prompt "
                    f"{prompt_id} timed out."
                )

            history = (
                self.get_history(
                    prompt_id
                )
            )

            if prompt_id in history:

                result = history[
                    prompt_id
                ]

                status = result.get(
                    "status",
                    {},
                )

                if status.get(
                    "status_str"
                ) == "error":

                    raise RuntimeError(
                        "ComfyUI generation "
                        f"failed for {prompt_id}: "
                        f"{status}"
                    )

                if result.get(
                    "outputs"
                ):

                    return result

            time.sleep(
                current_interval
            )

            current_interval = min(
                current_interval * 1.5,
                max_poll_interval,
            )

    # ========================================================================
    # DOWNLOAD FILE
    # ========================================================================

    def download_file(
        self,
        filename: str,
        subfolder: str = "",
        file_type: str = "output",
        destination: Path | None = None,
    ) -> Path:

        query = urlencode(
            {
                "filename": filename,
                "subfolder": subfolder,
                "type": file_type,
            }
        )

        data = self._request(
            "GET",
            f"/view?{query}",
            retry=True,
        )

        if not isinstance(
            data,
            bytes,
        ):

            raise RuntimeError(
                "ComfyUI /view did not "
                "return binary data."
            )

        if destination is None:

            destination = Path(
                filename
            )

        destination = Path(
            destination
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_bytes(
            data
        )

        if (
            not destination.is_file()
            or
            destination.stat().st_size <= 0
        ):

            raise RuntimeError(
                "Downloaded ComfyUI output "
                "is missing or empty:\n"
                f"{destination}"
            )

        return destination

    # ========================================================================
    # FIND VIDEO OUTPUTS
    # ========================================================================

    @staticmethod
    def find_video_outputs(
        history: dict,
    ) -> list:

        results = []

        outputs = (
            history.get(
                "outputs",
                {},
            )
        )

        if not isinstance(
            outputs,
            dict,
        ):

            return results

        for node_output in (
            outputs.values()
        ):

            if not isinstance(
                node_output,
                dict,
            ):

                continue

            for key, value in (
                node_output.items()
            ):

                if not isinstance(
                    value,
                    list,
                ):

                    continue

                for item in value:

                    if not isinstance(
                        item,
                        dict,
                    ):

                        continue

                    filename = (
                        item.get(
                            "filename"
                        )
                    )

                    if not filename:

                        continue

                    extension = (
                        Path(
                            filename
                        )
                        .suffix
                        .lower()
                    )

                    if extension in {
                        ".mp4",
                        ".mov",
                        ".webm",
                        ".mkv",
                        ".gif",
                    }:

                        results.append(
                            {
                                "filename": (
                                    filename
                                ),
                                "subfolder": (
                                    item.get(
                                        "subfolder",
                                        "",
                                    )
                                ),
                                "type": (
                                    item.get(
                                        "type",
                                        "output",
                                    )
                                ),
                            }
                        )

        return results
