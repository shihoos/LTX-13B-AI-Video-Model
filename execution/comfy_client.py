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


class ComfyClient:

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
    ):

        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        payload=None,
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

            raise RuntimeError(
                f"ComfyUI HTTP {error.code}: "
                f"{body}"
            ) from error

        except URLError as error:

            raise RuntimeError(
                f"Cannot connect to ComfyUI "
                f"at {self.base_url}: "
                f"{error}"
            ) from error

    def health_check(self) -> bool:

        try:

            self._request(
                "GET",
                "/system_stats",
            )

            return True

        except Exception:

            return False

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

        result = self._request(
            "POST",
            "/prompt",
            payload,
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

    def get_history(
        self,
        prompt_id: str,
    ) -> dict:

        return self._request(
            "GET",
            f"/history/{prompt_id}",
        )

    def wait_for_prompt(
        self,
        prompt_id: str,
        poll_interval: float = 2.0,
        timeout: float = 3600.0,
    ) -> dict:

        started = time.time()

        while True:

            if (
                time.time()
                - started
                > timeout
            ):

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
                poll_interval
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

            destination = (
                Path(filename)
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

        return destination

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

        for node_output in (
            outputs.values()
        ):

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
