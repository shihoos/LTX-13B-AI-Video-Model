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


TRANSIENT_HTTP_STATUS_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}


class ComfyClient:

    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        request_retries: int = 3,
    ):
        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout = max(
            1,
            int(timeout),
        )

        self.request_retries = max(
            0,
            int(request_retries),
        )

    def _request(
        self,
        method: str,
        path: str,
        payload=None,
        retry: bool = False,
        timeout: int | None = None,
    ):
        url = (
            self.base_url
            + path
        )

        data = None
        headers = {}

        if payload is not None:
            data = json.dumps(
                payload,
                ensure_ascii=False,
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

        request_timeout = (
            self.timeout
            if timeout is None
            else max(1, int(timeout))
        )

        last_error = None

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
                    timeout=request_timeout,
                ) as response:

                    body = response.read()

                    if not body:
                        return None

                    content_type = (
                        response.headers.get(
                            "Content-Type",
                            "",
                        )
                    ).lower()

                    if (
                        "json" in content_type
                        or body[:1] in (
                            b"{",
                            b"[",
                        )
                    ):
                        try:
                            return json.loads(
                                body.decode(
                                    "utf-8"
                                )
                            )
                        except json.JSONDecodeError:
                            pass

                    return body

            except HTTPError as error:
                last_error = error

                body = (
                    error.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )

                if (
                    retry
                    and error.code
                    in TRANSIENT_HTTP_STATUS_CODES
                    and attempt < attempts - 1
                ):
                    self._sleep_before_retry(
                        attempt
                    )
                    continue

                raise RuntimeError(
                    f"ComfyUI HTTP {error.code}: "
                    f"{body}"
                ) from error

            except (
                URLError,
                TimeoutError,
            ) as error:
                last_error = error

                if (
                    retry
                    and attempt < attempts - 1
                ):
                    self._sleep_before_retry(
                        attempt
                    )
                    continue

                raise RuntimeError(
                    "Cannot connect to ComfyUI "
                    f"at {self.base_url}: {error}"
                ) from error

        raise RuntimeError(
            f"ComfyUI request failed: {last_error}"
        )

    @staticmethod
    def _sleep_before_retry(
        attempt: int,
    ) -> None:
        time.sleep(
            min(
                1.0 * (2 ** attempt),
                10.0,
            )
        )

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

    def get_object_info(
        self,
    ) -> dict:
        result = self._request(
            "GET",
            "/object_info",
            retry=True,
        )

        if not isinstance(
            result,
            dict,
        ):
            raise RuntimeError(
                "Invalid /object_info response."
            )

        return result

    def convert_workflow(
        self,
        workflow: dict,
    ) -> dict:
        """
        Convert a full ComfyUI UI/save-format workflow
        into ComfyUI API format using the server-side converter.

        This deliberately does NOT attempt to recreate ComfyUI's
        frontend conversion logic locally.
        """

        result = self._request(
            "POST",
            "/workflow/convert",
            payload=workflow,
            retry=False,
            timeout=120,
        )

        if not isinstance(
            result,
            dict,
        ):
            raise RuntimeError(
                "ComfyUI workflow converter returned "
                "an invalid response."
            )

        if (
            "success" in result
            and result.get("success") is False
        ):
            raise RuntimeError(
                "ComfyUI workflow conversion failed: "
                + str(
                    result.get(
                        "error",
                        result,
                    )
                )
            )

        # The converter returns the API workflow itself.
        if all(
            isinstance(value, dict)
            for value in result.values()
        ):
            return result

        # Some converter versions may wrap it.
        for key in (
            "workflow",
            "data",
            "prompt",
        ):
            candidate = result.get(
                key
            )

            if (
                isinstance(candidate, dict)
                and all(
                    isinstance(value, dict)
                    for value in candidate.values()
                )
            ):
                return candidate

        raise RuntimeError(
            "Could not locate API workflow in "
            "converter response."
        )

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

        # NEVER retry /prompt automatically.
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
                "Invalid /prompt response."
            )

        if result.get(
            "error"
        ):
            raise RuntimeError(
                "ComfyUI rejected prompt: "
                + str(
                    result["error"]
                )
            )

        prompt_id = result.get(
            "prompt_id"
        )

        if not prompt_id:
            raise RuntimeError(
                "ComfyUI did not return prompt_id: "
                + str(result)
            )

        return str(
            prompt_id
        )

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
                "Invalid history response."
            )

        return result

    def wait_for_prompt(
        self,
        prompt_id: str,
        poll_interval: float = 2.0,
        timeout: float = 7200.0,
    ) -> dict:

        started = time.monotonic()
        current_interval = max(
            0.5,
            float(poll_interval),
        )

        while True:
            elapsed = (
                time.monotonic()
                - started
            )

            if elapsed > timeout:
                raise TimeoutError(
                    f"ComfyUI prompt {prompt_id} "
                    "timed out."
                )

            history = self.get_history(
                prompt_id
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
                        "ComfyUI generation failed "
                        f"for {prompt_id}: "
                        f"{status}"
                    )

                if status.get(
                    "status_str"
                ) == "success":
                    return result

                if result.get(
                    "outputs"
                ):
                    return result

            time.sleep(
                current_interval
            )

            current_interval = min(
                current_interval * 1.5,
                10.0,
            )

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
                "ComfyUI /view did not return "
                "binary data."
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
            or destination.stat().st_size <= 0
        ):
            raise RuntimeError(
                "Downloaded output is missing "
                "or empty: "
                f"{destination}"
            )

        return destination

    @staticmethod
    def _is_video_filename(
        filename: str,
    ) -> bool:
        return Path(
            filename
        ).suffix.lower() in {
            ".mp4",
            ".mov",
            ".webm",
            ".mkv",
            ".gif",
        }

    @staticmethod
    def _is_image_filename(
        filename: str,
    ) -> bool:
        return Path(
            filename
        ).suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }

    @classmethod
    def find_video_outputs(
        cls,
        history: dict,
    ) -> list[dict]:

        results = []

        outputs = history.get(
            "outputs",
            {},
        )

        if not isinstance(
            outputs,
            dict,
        ):
            return results

        for node_output in outputs.values():

            if not isinstance(
                node_output,
                dict,
            ):
                continue

            for value in node_output.values():

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

                    filename = item.get(
                        "filename"
                    )

                    if not filename:
                        continue

                    if not cls._is_video_filename(
                        filename
                    ):
                        continue

                    results.append(
                        {
                            "filename": filename,
                            "subfolder": item.get(
                                "subfolder",
                                "",
                            ),
                            "type": item.get(
                                "type",
                                "output",
                            ),
                        }
                    )

        return results

    @classmethod
    def find_image_outputs(
        cls,
        history: dict,
    ) -> list[dict]:

        results = []

        outputs = history.get(
            "outputs",
            {},
        )

        if not isinstance(
            outputs,
            dict,
        ):
            return results

        for node_output in outputs.values():

            if not isinstance(
                node_output,
                dict,
            ):
                continue

            for value in node_output.values():

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

                    filename = item.get(
                        "filename"
                    )

                    if (
                        not filename
                        or not cls._is_image_filename(
                            filename
                        )
                    ):
                        continue

                    results.append(
                        {
                            "filename": filename,
                            "subfolder": item.get(
                                "subfolder",
                                "",
                            ),
                            "type": item.get(
                                "type",
                                "output",
                            ),
                        }
                    )

        return results
