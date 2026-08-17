from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ComfyWorkflowAdapter:
    """
    Adapter for converting a ComfyUI workflow JSON into an API prompt
    and dynamically injecting shot-specific values.

    Supported runtime changes:

        - prompt replacement
        - negative prompt replacement
        - seed replacement
        - filename prefix replacement
        - reference image replacement
        - input video replacement
    """

    def __init__(self, workflow_path: str | Path):
        self.workflow_path = Path(workflow_path)

        if not self.workflow_path.exists():
            raise FileNotFoundError(
                f"Workflow file does not exist: "
                f"{self.workflow_path}"
            )

        with self.workflow_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.workflow_json = json.load(file)

    # ============================================================
    # GRAPH JSON -> API WORKFLOW
    # ============================================================

    def to_api_workflow(self) -> dict[str, Any]:
        """
        Convert a normal ComfyUI graph/workflow JSON into the API
        prompt format expected by POST /prompt.

        If the JSON is already API-format, return a deep copy.
        """

        workflow = self.workflow_json

        # --------------------------------------------------------
        # Already API format
        # --------------------------------------------------------

        if isinstance(workflow, dict) and all(
            isinstance(value, dict)
            and "class_type" in value
            for value in workflow.values()
        ):
            return copy.deepcopy(workflow)

        # --------------------------------------------------------
        # Standard ComfyUI graph format
        # --------------------------------------------------------

        if not isinstance(workflow, dict):
            raise RuntimeError(
                "Workflow JSON must be a dictionary."
            )

        nodes = workflow.get("nodes")

        if not isinstance(nodes, list):
            raise RuntimeError(
                f"Unable to convert workflow to API format: "
                f"{self.workflow_path} does not contain an API "
                f"workflow or a standard 'nodes' list."
            )

        api_workflow: dict[str, Any] = {}

        for node in nodes:

            node_id = str(node.get("id"))

            if not node_id or node_id == "None":
                raise RuntimeError(
                    "Workflow contains a node without a valid ID."
                )

            class_type = node.get("type")

            if not class_type:
                raise RuntimeError(
                    f"Workflow node {node_id} has no type."
                )

            inputs: dict[str, Any] = {}

            widgets_values = node.get(
                "widgets_values",
                [],
            )

            if isinstance(widgets_values, list):

                for index, value in enumerate(
                    widgets_values
                ):
                    inputs[f"widget_{index}"] = value

            api_workflow[node_id] = {
                "class_type": class_type,
                "inputs": inputs,
            }

        links = workflow.get(
            "links",
            [],
        )

        if isinstance(links, list):

            node_map = {
                str(node.get("id")): node
                for node in nodes
            }

            for link in links:

                if not isinstance(link, list):
                    continue

                if len(link) < 5:
                    continue

                origin_node_id = str(link[1])
                origin_slot = link[2]
                target_node_id = str(link[3])
                target_slot = link[4]

                target_node = node_map.get(
                    target_node_id
                )

                if target_node is None:
                    continue

                inputs_list = target_node.get(
                    "inputs",
                    [],
                )

                if (
                    not isinstance(
                        inputs_list,
                        list,
                    )
                    or target_slot >= len(inputs_list)
                ):
                    continue

                target_input = inputs_list[
                    target_slot
                ]

                if not isinstance(
                    target_input,
                    dict,
                ):
                    continue

                input_name = target_input.get(
                    "name"
                )

                if not input_name:
                    continue

                if target_node_id not in api_workflow:
                    continue

                api_workflow[
                    target_node_id
                ]["inputs"][input_name] = [
                    origin_node_id,
                    origin_slot,
                ]

        return api_workflow

    # ============================================================
    # SHOT APPLICATION
    # ============================================================

    def apply_shot(
        self,
        workflow: dict[str, Any],
        shot: Any,
    ) -> dict[str, Any]:
        """
        Create an independent copy of the API workflow and apply
        shot-specific prompt information.

        The method intentionally works with flexible Shot objects
        so the planner/schema can evolve without breaking the
        execution layer.
        """

        result = copy.deepcopy(workflow)

        prompt = self._get_shot_prompt(
            shot
        )

        negative_prompt = self._get_shot_negative_prompt(
            shot
        )

        if prompt:
            self.set_prompt(
                result,
                prompt,
            )

        if negative_prompt:
            self.set_negative_prompt(
                result,
                negative_prompt,
            )

        return result

    # ============================================================
    # SHOT PROMPT EXTRACTION
    # ============================================================

    @staticmethod
    def _get_shot_prompt(
        shot: Any,
    ) -> str | None:

        candidate_fields = [
            "prompt",
            "positive_prompt",
            "generation_prompt",
            "description",
        ]

        for field in candidate_fields:

            value = getattr(
                shot,
                field,
                None,
            )

            if isinstance(
                value,
                str,
            ) and value.strip():

                return value.strip()

        return None

    @staticmethod
    def _get_shot_negative_prompt(
        shot: Any,
    ) -> str | None:

        candidate_fields = [
            "negative_prompt",
            "negative",
        ]

        for field in candidate_fields:

            value = getattr(
                shot,
                field,
                None,
            )

            if isinstance(
                value,
                str,
            ) and value.strip():

                return value.strip()

        return None

    # ============================================================
    # PROMPT REPLACEMENT
    # ============================================================

    @staticmethod
    def set_prompt(
        workflow: dict[str, Any],
        prompt: str,
    ) -> None:

        candidates = []

        for node_id, node in workflow.items():

            class_type = node.get(
                "class_type",
                ""
            )

            inputs = node.setdefault(
                "inputs",
                {},
            )

            if "CLIPTextEncode" not in class_type:
                continue

            if "text" not in inputs:
                continue

            existing_text = inputs.get(
                "text"
            )

            candidates.append(
                (
                    node_id,
                    existing_text,
                )
            )

        if not candidates:

            raise RuntimeError(
                "Workflow has no CLIPTextEncode node "
                "with a writable 'text' input."
            )

        # Prefer a node whose existing text does not look like
        # a negative prompt.

        for node_id, existing_text in candidates:

            text = str(
                existing_text
            ).lower()

            if not any(
                word in text
                for word in [
                    "negative",
                    "worst quality",
                    "low quality",
                    "blurry",
                ]
            ):

                workflow[node_id][
                    "inputs"
                ]["text"] = prompt

                return

        # Fallback to first candidate.

        node_id = candidates[0][0]

        workflow[node_id][
            "inputs"
        ]["text"] = prompt

    # ============================================================
    # NEGATIVE PROMPT REPLACEMENT
    # ============================================================

    @staticmethod
    def set_negative_prompt(
        workflow: dict[str, Any],
        negative_prompt: str,
    ) -> None:

        candidates = []

        for node_id, node in workflow.items():

            class_type = node.get(
                "class_type",
                ""
            )

            inputs = node.setdefault(
                "inputs",
                {},
            )

            if "CLIPTextEncode" not in class_type:
                continue

            if "text" not in inputs:
                continue

            existing_text = str(
                inputs.get(
                    "text",
                    ""
                )
            ).lower()

            if any(
                word in existing_text
                for word in [
                    "negative",
                    "worst quality",
                    "low quality",
                    "blurry",
                ]
            ):
                candidates.append(
                    node_id
                )

        if candidates:

            workflow[
                candidates[0]
            ]["inputs"][
                "text"
            ] = negative_prompt

    # ============================================================
    # SEED REPLACEMENT
    # ============================================================

    @staticmethod
    def set_seed(
        workflow: dict[str, Any],
        seed: int,
    ) -> None:

        updated = False

        for node in workflow.values():

            inputs = node.setdefault(
                "inputs",
                {},
            )

            if "seed" in inputs:

                inputs["seed"] = int(seed)

                updated = True

        if not updated:

            raise RuntimeError(
                "Workflow has no writable 'seed' input."
            )

    # ============================================================
    # OUTPUT FILENAME PREFIX
    # ============================================================

    @staticmethod
    def set_filename_prefix(
        workflow: dict[str, Any],
        filename_prefix: str,
    ) -> None:

        updated = False

        for node in workflow.values():

            inputs = node.setdefault(
                "inputs",
                {},
            )

            if "filename_prefix" in inputs:

                inputs[
                    "filename_prefix"
                ] = filename_prefix

                updated = True

        if not updated:

            raise RuntimeError(
                "Workflow has no writable "
                "'filename_prefix' input."
            )

    # ============================================================
    # REFERENCE IMAGE
    # ============================================================

    @staticmethod
    def set_input_image(
        workflow: dict[str, Any],
        filename: str,
    ) -> None:

        for node in workflow.values():

            if node.get(
                "class_type"
            ) != "LoadImage":

                continue

            inputs = node.setdefault(
                "inputs",
                {},
            )

            if "image" in inputs:

                inputs["image"] = filename

                return

        raise RuntimeError(
            "Workflow has no LoadImage node "
            "with a writable 'image' input."
        )

    # ============================================================
    # RAW VIDEO FOR DETAILER
    # ============================================================

    @staticmethod
    def set_input_video(
        workflow: dict[str, Any],
        filename: str,
    ) -> None:

        for node in workflow.values():

            if node.get(
                "class_type"
            ) != "VHS_LoadVideo":

                continue

            inputs = node.setdefault(
                "inputs",
                {},
            )

            # VHS_LoadVideo normally uses "video".
            if "video" in inputs:

                inputs["video"] = filename

                return

        raise RuntimeError(
            "Workflow has no VHS_LoadVideo node "
            "with a writable 'video' input."
        )
