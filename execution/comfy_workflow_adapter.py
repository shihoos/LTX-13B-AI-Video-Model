from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ComfyWorkflowAdapter:
    """
    Converts a saved ComfyUI graph JSON into a ComfyUI API prompt.

    Supports runtime replacement of:
        - positive prompt
        - negative prompt
        - seed / noise_seed
        - filename_prefix
        - LoadImage.image
        - VHS_LoadVideo.video
    """

    # UI/documentation nodes that must not be sent to the ComfyUI API.
    UI_ONLY_NODE_TYPES = {
        "Note",
    }

    def __init__(self, workflow_path: str | Path):
        self.workflow_path = Path(workflow_path)

        if not self.workflow_path.exists():
            raise FileNotFoundError(
                f"Workflow does not exist: {self.workflow_path}"
            )

        with self.workflow_path.open("r", encoding="utf-8") as file:
            self.workflow_json = json.load(file)

    # ============================================================
    # WORKFLOW -> COMFYUI API PROMPT
    # ============================================================

    def to_api_workflow(self) -> dict[str, Any]:
        """
        Convert a normal ComfyUI graph JSON into the API prompt format.

        Normal workflow format:
            {
                "nodes": [...],
                "links": [...]
            }

        API prompt format:
            {
                "node_id": {
                    "class_type": "...",
                    "inputs": {...}
                }
            }
        """

        workflow = self.workflow_json

        if self._is_api_workflow(workflow):
            return copy.deepcopy(workflow)

        if not isinstance(workflow, dict):
            raise RuntimeError(
                f"Workflow must be a dictionary: {self.workflow_path}"
            )

        nodes = workflow.get("nodes")

        if not isinstance(nodes, list):
            raise RuntimeError(
                f"Workflow is neither a valid ComfyUI graph nor "
                f"an API workflow: {self.workflow_path}"
            )

        # --------------------------------------------------------
        # Build:
        #
        # link_id -> [source_node_id, source_output_index]
        # --------------------------------------------------------

        link_lookup: dict[int, list[Any]] = {}

        for link in workflow.get("links", []):

            if not isinstance(link, list) or len(link) < 5:
                continue

            link_id = link[0]
            source_node_id = str(link[1])
            source_output_index = link[2]

            link_lookup[link_id] = [
                source_node_id,
                source_output_index,
            ]

        api_workflow: dict[str, Any] = {}

        for node in nodes:

            if not isinstance(node, dict):
                continue

            node_id_value = node.get("id")
            class_type = node.get("type")

            if node_id_value is None:
                raise RuntimeError(
                    "Workflow contains a node without an ID."
                )

            if not class_type:
                raise RuntimeError(
                    f"Node {node_id_value} has no type."
                )

            # Skip UI/documentation-only nodes.
            if class_type in self.UI_ONLY_NODE_TYPES:
                continue

            node_id = str(node_id_value)
            api_inputs: dict[str, Any] = {}

            # ----------------------------------------------------
            # Named widgets
            #
            # Examples:
            #
            # text
            # noise_seed
            # filename_prefix
            # image
            # video
            # ----------------------------------------------------

            named_widgets = node.get("widgets_values_named", {})

            if isinstance(named_widgets, dict):
                for name, value in named_widgets.items():
                    api_inputs[name] = copy.deepcopy(value)

            # ----------------------------------------------------
            # Graph connections
            # ----------------------------------------------------

            node_inputs = node.get("inputs", [])

            if isinstance(node_inputs, list):

                for input_def in node_inputs:

                    if not isinstance(input_def, dict):
                        continue

                    input_name = input_def.get("name")

                    if not input_name:
                        continue

                    link_id = input_def.get("link")

                    if link_id is not None:

                        source = link_lookup.get(link_id)

                        if source is None:
                            raise RuntimeError(
                                f"Node {node_id} input "
                                f"'{input_name}' references "
                                f"missing link {link_id}."
                            )

                        source_node_id = source[0]

                        # A graph connection cannot originate from
                        # a skipped UI-only node.
                        if source_node_id not in {
                            str(n.get("id"))
                            for n in nodes
                            if isinstance(n, dict)
                            and n.get("type")
                            not in self.UI_ONLY_NODE_TYPES
                        }:
                            raise RuntimeError(
                                f"Node {node_id} input '{input_name}' "
                                f"references skipped/non-executable "
                                f"node {source_node_id}."
                            )

                        api_inputs[input_name] = copy.deepcopy(source)

                    elif "value" in input_def:
                        api_inputs[input_name] = copy.deepcopy(
                            input_def["value"]
                        )

            api_workflow[node_id] = {
                "class_type": class_type,
                "inputs": api_inputs,
            }

        if not api_workflow:
            raise RuntimeError(
                f"No executable nodes found in: {self.workflow_path}"
            )

        return api_workflow

    @staticmethod
    def _is_api_workflow(workflow: Any) -> bool:
        """
        Return True only when the dictionary already looks like
        a ComfyUI API prompt.
        """

        if not isinstance(workflow, dict) or not workflow:
            return False

        if "nodes" in workflow:
            return False

        return all(
            isinstance(value, dict)
            and "class_type" in value
            and "inputs" in value
            for value in workflow.values()
        )

    # ============================================================
    # APPLY SHOT
    # ============================================================

    def apply_shot(
        self,
        workflow: dict[str, Any],
        shot: Any,
    ) -> dict[str, Any]:
        """
        Apply shot-specific prompt values to a copy of the workflow.
        """

        result = copy.deepcopy(workflow)

        positive_prompt = self._get_shot_value(
            shot,
            [
                "prompt",
                "positive_prompt",
                "generation_prompt",
                "description",
            ],
        )

        negative_prompt = self._get_shot_value(
            shot,
            [
                "negative_prompt",
                "negative",
            ],
        )

        if positive_prompt:
            self.set_prompt(result, positive_prompt)

        if negative_prompt:
            self.set_negative_prompt(result, negative_prompt)

        return result

    @staticmethod
    def _get_shot_value(
        shot: Any,
        field_names: list[str],
    ) -> str | None:

        if isinstance(shot, dict):
            for field in field_names:
                value = shot.get(field)

                if isinstance(value, str) and value.strip():
                    return value.strip()

            return None

        for field in field_names:
            value = getattr(shot, field, None)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    # ============================================================
    # PROMPTS
    # ============================================================

    @staticmethod
    def _get_clip_text_nodes(
        workflow: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:

        result = []

        for node_id, node in workflow.items():

            if node.get("class_type") != "CLIPTextEncode":
                continue

            inputs = node.get("inputs", {})

            if "text" in inputs:
                result.append((node_id, node))

        return result

    @staticmethod
    def set_prompt(
        workflow: dict[str, Any],
        prompt: str,
    ) -> None:

        candidates = ComfyWorkflowAdapter._get_clip_text_nodes(
            workflow
        )

        if not candidates:
            raise RuntimeError(
                "No CLIPTextEncode node with a text input was found."
            )

        # Prefer an empty prompt node.
        for _, node in candidates:

            current = str(
                node["inputs"].get("text", "")
            ).strip()

            if not current:
                node["inputs"]["text"] = prompt
                return

        # Otherwise select the first node that does not look negative.
        negative_markers = (
            "negative",
            "worst quality",
            "low quality",
            "blurry",
            "deformed",
            "distorted",
            "bad anatomy",
        )

        for _, node in candidates:

            current = str(
                node["inputs"].get("text", "")
            ).lower()

            if not any(marker in current for marker in negative_markers):
                node["inputs"]["text"] = prompt
                return

        # Final fallback.
        candidates[0][1]["inputs"]["text"] = prompt

    @staticmethod
    def set_negative_prompt(
        workflow: dict[str, Any],
        negative_prompt: str,
    ) -> None:

        candidates = ComfyWorkflowAdapter._get_clip_text_nodes(
            workflow
        )

        if not candidates:
            raise RuntimeError(
                "No CLIPTextEncode node with a text input was found."
            )

        negative_markers = (
            "negative",
            "worst quality",
            "low quality",
            "blurry",
            "deformed",
            "distorted",
            "bad anatomy",
        )

        for _, node in candidates:

            current = str(
                node["inputs"].get("text", "")
            ).lower()

            if any(marker in current for marker in negative_markers):
                node["inputs"]["text"] = negative_prompt
                return

        # If there are at least two text nodes, the second is the
        # safest fallback for the negative conditioning path.
        if len(candidates) >= 2:
            candidates[1][1]["inputs"]["text"] = negative_prompt
            return

        raise RuntimeError(
            "Could not identify a negative CLIPTextEncode node."
        )

    # ============================================================
    # SEED
    # ============================================================

    @staticmethod
    def set_seed(
        workflow: dict[str, Any],
        seed: int,
    ) -> None:

        seed = int(seed)
        updated = False

        for node in workflow.values():

            inputs = node.get("inputs", {})

            for seed_name in ("noise_seed", "seed"):

                if seed_name in inputs:
                    inputs[seed_name] = seed
                    updated = True

        if not updated:
            raise RuntimeError(
                "No seed or noise_seed input was found."
            )

    # ============================================================
    # OUTPUT PREFIX
    # ============================================================

    @staticmethod
    def set_filename_prefix(
        workflow: dict[str, Any],
        filename_prefix: str,
    ) -> None:

        updated = False

        for node in workflow.values():

            inputs = node.get("inputs", {})

            if "filename_prefix" in inputs:
                inputs["filename_prefix"] = filename_prefix
                updated = True

        if not updated:
            raise RuntimeError(
                "No filename_prefix input was found."
            )

    # ============================================================
    # INPUT IMAGE
    # ============================================================

    @staticmethod
    def set_input_image(
        workflow: dict[str, Any],
        filename: str,
    ) -> None:

        for node in workflow.values():

            if node.get("class_type") == "LoadImage":
                node.setdefault("inputs", {})["image"] = filename
                return

        raise RuntimeError(
            "No LoadImage node was found."
        )

    # ============================================================
    # INPUT VIDEO
    # ============================================================

    @staticmethod
    def set_input_video(
        workflow: dict[str, Any],
        filename: str,
    ) -> None:

        for node in workflow.values():

            if node.get("class_type") == "VHS_LoadVideo":
                node.setdefault("inputs", {})["video"] = filename
                return

        raise RuntimeError(
            "No VHS_LoadVideo node was found."
        )
