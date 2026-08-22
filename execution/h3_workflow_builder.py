from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class H3WorkflowBuilder:
    """
    Loads complete upstream ComfyUI UI-workflow JSON files and converts them
    to ComfyUI API prompt format after applying shot-specific values.

    This deliberately does NOT rebuild the H3 graph from scratch. The
    upstream JSON remains the authoritative graph.
    """

    MODEL = "minimax_h3_ref2va_pruned-Q4_K_M.gguf"
    CLIP = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    MM_PROJ = "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf"
    VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
    AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

    def __init__(
        self,
        project_root: Path,
        object_info: dict[str, Any],
        workflow_root: Path | None = None,
    ):
        self.project_root = Path(project_root)
        self.object_info = object_info
        self.workflow_root = (
            Path(workflow_root)
            if workflow_root
            else self.project_root / "workflows" / "MiniMax-H3"
        )

    def _require_node(self, node_type: str) -> None:
        if node_type not in self.object_info:
            raise RuntimeError(
                f"Required H3 node is not installed in ComfyUI: {node_type}"
            )

    def validate_runtime(self) -> None:
        required = (
            "H3ModelLoaderAny",
            "H3ClipLoaderAny",
            "MiniMaxH3ReferenceToVideo",
            "H3FreeTextEncoder",
            "VAELoader",
            "VAEDecode",
            "VAEDecodeAudio",
            "RandomNoise",
            "BasicGuider",
            "KSamplerSelect",
            "BasicScheduler",
            "SamplerCustomAdvanced",
            "CreateVideo",
            "SaveVideo",
        )
        for node_type in required:
            self._require_node(node_type)

    def _resolve_workflow(self, name: str) -> Path:
        candidates = [
            self.workflow_root / name,
            self.workflow_root / "base" / name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"H3 workflow not found. Tried:\n"
            + "\n".join(str(path) for path in candidates)
        )

    def load_workflow(self, name: str) -> dict[str, Any]:
        path = self._resolve_workflow(name)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a JSON object")
        if not isinstance(data.get("nodes"), list):
            raise ValueError(f"{path} is not a ComfyUI UI workflow")

        return data

    @staticmethod
    def _node_matches(node: dict[str, Any], *terms: str) -> bool:
        haystack = " ".join(
            [
                str(node.get("type", "")),
                str(node.get("title", "")),
                str(node.get("properties", {}).get("Node name for S&R", "")),
            ]
        ).lower()
        return all(term.lower() in haystack for term in terms)

    @staticmethod
    def _widget_values(node: dict[str, Any]) -> list[Any]:
        values = node.get("widgets_values")
        if isinstance(values, list):
            return values
        return []

    @staticmethod
    def _set_widget(node: dict[str, Any], index: int, value: Any) -> None:
        values = node.setdefault("widgets_values", [])
        while len(values) <= index:
            values.append(None)
        values[index] = value

    @staticmethod
    def _set_first_widget(node: dict[str, Any], value: Any) -> None:
        H3WorkflowBuilder._set_widget(node, 0, value)

    @staticmethod
    def _find_nodes(workflow: dict[str, Any], predicate):
        return [
            node
            for node in workflow.get("nodes", [])
            if isinstance(node, dict) and predicate(node)
        ]

    def _patch_model_and_encoder(self, workflow: dict[str, Any]) -> None:
        for node in workflow["nodes"]:
            if not isinstance(node, dict):
                continue

            if node.get("type") == "H3ModelLoaderAny":
                title = str(node.get("title", "")).lower()
                values = self._widget_values(node)

                if "ref2va" in title or "ref2va" in " ".join(map(str, values)).lower():
                    self._set_first_widget(node, self.MODEL)

                if "fl2va" in title and "ref2va" not in title:
                    # Keep FL2VA workflows untouched. They belong to a separate
                    # checkpoint family.
                    continue

            elif node.get("type") == "H3ClipLoaderAny":
                values = self._widget_values(node)
                self._set_first_widget(node, self.CLIP)

            elif node.get("type") == "VAELoader":
                title = str(node.get("title", "")).lower()
                if "audio" in title:
                    self._set_first_widget(node, self.AUDIO_VAE)
                elif "video" in title:
                    self._set_first_widget(node, self.VIDEO_VAE)

    def _patch_common_values(
        self,
        workflow: dict[str, Any],
        *,
        prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        duration_seconds: float | None = None,
        output_prefix: str | None = None,
    ) -> None:
        for node in workflow["nodes"]:
            if not isinstance(node, dict):
                continue

            node_type = str(node.get("type", ""))
            title = str(node.get("title", ""))

            if prompt is not None and (
                "PrimitiveStringMultiline" in node_type
                or "PrimitiveString" in node_type
            ):
                low = title.lower()
                if any(
                    key in low
                    for key in (
                        "prompt",
                        "script",
                        "binding",
                        "input text",
                    )
                ):
                    self._set_first_widget(node, prompt)

            if seed is not None and node_type == "RandomNoise":
                self._set_first_widget(node, int(seed))

            if steps is not None and node_type == "BasicScheduler":
                values = self._widget_values(node)
                if len(values) >= 2:
                    values[1] = int(steps)

            if duration_seconds is not None and node_type == "PrimitiveFloat":
                if "duration" in title.lower():
                    self._set_first_widget(
                        node,
                        float(duration_seconds),
                    )

            if node_type == "MiniMaxH3ReferenceToVideo":
                if width is not None:
                    # widget_values: prompt, width, height, length, ref_image_size
                    self._set_widget(node, 1, int(width))
                if height is not None:
                    self._set_widget(node, 2, int(height))
                if frames is not None:
                    self._set_widget(node, 3, int(frames))

            if node_type == "H3Keyframes":
                if width is not None:
                    # H3Keyframes widget_values:
                    # prompt, width, height, length, positions
                    self._set_widget(node, 1, int(width))
                if height is not None:
                    self._set_widget(node, 2, int(height))
                if frames is not None:
                    self._set_widget(node, 3, int(frames))
                if prompt is not None:
                    self._set_first_widget(node, prompt)

            if output_prefix is not None and node_type == "SaveVideo":
                # Usually first widget is filename_prefix.
                self._set_first_widget(node, output_prefix)

    @staticmethod
    def _set_load_image_slots(
        workflow: dict[str, Any],
        filenames: list[str],
    ) -> int:
        loaders = [
            node
            for node in workflow["nodes"]
            if isinstance(node, dict)
            and node.get("type") == "LoadImage"
            and (
                "reference" in str(node.get("title", "")).lower()
                or "picture" in str(node.get("title", "")).lower()
            )
        ]

        count = min(len(loaders), len(filenames))

        for index in range(count):
            H3WorkflowBuilder._set_first_widget(
                loaders[index],
                filenames[index],
            )

        return count

    @staticmethod
    def _set_load_audio_slots(
        workflow: dict[str, Any],
        filenames: list[str],
    ) -> int:
        loaders = [
            node
            for node in workflow["nodes"]
            if isinstance(node, dict)
            and node.get("type") == "LoadAudio"
        ]

        count = min(len(loaders), len(filenames))

        for index in range(count):
            H3WorkflowBuilder._set_first_widget(
                loaders[index],
                filenames[index],
            )
            loaders[index]["mode"] = 0

        return count

    @staticmethod
    def _set_load_video_slots(
        workflow: dict[str, Any],
        filenames: list[str],
    ) -> int:
        loaders = [
            node
            for node in workflow["nodes"]
            if isinstance(node, dict)
            and node.get("type") in {
                "VHS_LoadVideo",
                "VHS_LoadVideoPath",
                "VHS_LoadVideoFFmpeg",
            }
        ]

        count = min(len(loaders), len(filenames))

        for index in range(count):
            H3WorkflowBuilder._set_first_widget(
                loaders[index],
                filenames[index],
            )

        return count

    @staticmethod
    def _ui_input_value_map(
        workflow: dict[str, Any],
    ) -> dict[int, tuple[int, int]]:
        result: dict[int, tuple[int, int]] = {}
        for link in workflow.get("links", []):
            if not isinstance(link, list) or len(link) < 5:
                continue
            link_id = int(link[0])
            src_node = int(link[1])
            src_slot = int(link[2])
            result[link_id] = (src_node, src_slot)
        return result

    @classmethod
    def ui_to_api_prompt(
        cls,
        workflow: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """
        Convert a ComfyUI UI workflow export into the API /prompt graph.

        This is intentionally template-driven: the complete upstream JSON
        remains intact; only its widget values are patched before conversion.
        """

        links = cls._ui_input_value_map(workflow)
        api: dict[str, dict[str, Any]] = {}

        for node in workflow.get("nodes", []):
            if not isinstance(node, dict):
                continue

            node_id = str(node["id"])
            class_type = node.get("type")
            if not class_type:
                continue

            inputs: dict[str, Any] = {}
            widget_values = cls._widget_values(node)
            widget_index = 0

            for input_def in node.get("inputs", []):
                if not isinstance(input_def, dict):
                    continue

                name = input_def.get("name")
                if not name:
                    continue

                link_id = input_def.get("link")

                if link_id is not None and int(link_id) in links:
                    src_node, src_slot = links[int(link_id)]
                    inputs[str(name)] = [str(src_node), src_slot]
                    continue

                widget = input_def.get("widget")
                if widget is not None:
                    if widget_index < len(widget_values):
                        inputs[str(name)] = copy.deepcopy(
                            widget_values[widget_index]
                        )
                    widget_index += 1

            api[node_id] = {
                "class_type": class_type,
                "inputs": inputs,
            }

        return api

    def build_ref2va(
        self,
        *,
        prompt: str,
        image_files: list[str],
        video_files: list[str],
        audio_files: list[str],
        width: int,
        height: int,
        frames: int,
        steps: int,
        seed: int,
        output_prefix: str,
    ) -> dict[str, dict[str, Any]]:
        self.validate_runtime()

        workflow = self.load_workflow("H3_HardMode_R2V.json")
        self._patch_model_and_encoder(workflow)

        self._set_load_image_slots(workflow, image_files[:9])
        self._set_load_video_slots(workflow, video_files[:3])
        self._set_load_audio_slots(workflow, audio_files[:3])

        self._patch_common_values(
            workflow,
            prompt=prompt,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            seed=seed,
            duration_seconds=frames / 24.0,
            output_prefix=output_prefix,
        )

        return self.ui_to_api_prompt(workflow)

    def build_chained(
        self,
        *,
        script_with_bindings: str,
        image_files: list[str],
        audio_files: list[str],
        width: int,
        height: int,
        frames: int,
        steps: int,
        seed: int,
        output_prefix: str,
    ) -> dict[str, dict[str, Any]]:
        self.validate_runtime()

        workflow = self.load_workflow("H3_HardMode_Chained.json")
        self._patch_model_and_encoder(workflow)

        self._set_load_image_slots(workflow, image_files[:9])
        self._set_load_audio_slots(workflow, audio_files[:3])

        # The chained graph accepts the entire script in its multiline
        # script/bindings control. Keep the separators exactly as expected by
        # the upstream workflow.
        self._patch_common_values(
            workflow,
            prompt=script_with_bindings,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            seed=seed,
            duration_seconds=frames / 24.0,
            output_prefix=output_prefix,
        )

        return self.ui_to_api_prompt(workflow)
