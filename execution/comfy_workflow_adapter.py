from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ComfyWorkflowAdapter:
    """
    Converts a saved ComfyUI graph workflow into the API prompt format
    and applies runtime values for each shot.

    Supports:
        - positive prompt
        - negative prompt
        - seed / noise_seed
        - filename_prefix
        - LoadImage.image
        - VHS_LoadVideo.video
    """

    def __init__(self, workflow_path: str | Path):
        self.workflow_path = Path(workflow_path)

        if not self.workflow_path.exists():
            raise FileNotFoundError(
                f"Workflow does not exist: {self.workflow_path}"
            )

        with self.workflow_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.workflow_json = json.load(file)

    # ============================================================
    # WORKFLOW -> COMFYUI API PROMPT
    # ============================================================

    def to_api_workflow(self) -> dict[str, Any]:
        """
        Convert a normal ComfyUI workflow JSON into the API prompt
        structure expected by POST /prompt.

        If the file is already in API format, return a copy.
        """

        workflow = self.workflow_json

        # --------------------------------------------------------
        # Already API prompt format
        # --------------------------------------------------------

        if (
            isinstance(workflow, dict)
            and workflow
            and all(
                isinstance(value, dict)
                and "class_type" in value
                for value in workflow.values()
            )
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
                f"Workflow is neither API format nor a normal "
                f"ComfyUI graph: {self.workflow_path}"
            )

        # --------------------------------------------------------
        # Build lookup:
        #
        # link_id ->
        #     [origin_node_id, origin_output_slot]
        #
        # Example:
        #
        # target input:
        #     "link": 5162
        #
        # API prompt:
        #     ["1206", 0]
        # --------------------------------------------------------

        link_lookup: dict[int, list[Any]] = {}

        for link in workflow.get("links", []):

            if not isinstance(link, list):
                continue

            if len(link) < 5:
                continue

            link_id = link[0]
            origin_node_id = str(link[1])
            origin_output_slot = link[2]

            link_lookup[link_id] = [
                origin_node_id,
                origin_output_slot,
            ]

        # --------------------------------------------------------
        # Convert nodes
        # --------------------------------------------------------

        api_workflow: dict[str, Any] = {}

        for node in nodes:

            node_id = str(node.get("id"))

            class_type = node.get("type")

            if not node_id or node_id == "None":
                raise RuntimeError(
                    "Workflow contains a node without a valid ID."
                )

            if not class_type:
                raise RuntimeError(
                    f"Node {node_id} has no type."
                )

            api_inputs: dict[str, Any] = {}

            # ----------------------------------------------------
            # 1. Named widget values
            #
            # This is the important part for your workflows.
            #
            # Example:
            #
            # "widgets_values_named": {
            #     "text": "...",
            #     "noise_seed": 1256,
            #     "filename_prefix": "ltxv-base"
            # }
            # ----------------------------------------------------

            named_widgets = node.get(
                "widgets_values_named",
                {},
            )

            if isinstance(named_widgets, dict):

                for name, value in named_widgets.items():

                    api_inputs[name] = copy.deepcopy(
                        value
                    )

            # ----------------------------------------------------
            # 2. Graph inputs
            #
            # Convert ComfyUI link references:
            #
            # {
            #     "name": "clip",
            #     "link": 5771
            # }
            #
            # into:
            #
            # "clip": ["2010", 0]
            # ----------------------------------------------------

            node_inputs = node.get(
                "inputs",
                [],
            )

            if isinstance(node_inputs, list):

                for input_def in node_inputs:

                    if not isinstance(
                        input_def,
                        dict,
                    ):
                        continue

                    input_name = input_def.get(
                        "name"
                    )

                    if not input_name:
                        continue

                    link_id = input_def.get(
                        "link"
                    )

                    if link_id is not None:

                        source = link_lookup.get(
                            link_id
                        )

                        if source is None:
                            raise RuntimeError(
                                f"Node {node_id} input "
                                f"'{input_name}' references "
                                f"missing link {link_id}."
                            )

                        api_inputs[
                            input_name
                        ] = copy.deepcopy(
                            source
                        )

                    # ------------------------------------------------
                    # Some graph JSON versions can contain a literal
                    # value directly in the input definition.
                    # ------------------------------------------------

                    elif "value" in input_def:

                        api_inputs[
                            input_name
                        ] = copy.deepcopy(
                            input_def["value"]
                        )

            api_workflow[node_id] = {
                "class_type": class_type,
                "inputs": api_inputs,
            }

        return api_workflow

    # ============================================================
    # APPLY SHOT
    # ============================================================

    def apply_shot(
        self,
        workflow: dict[str, Any],
        shot: Any,
    ) -> dict[str, Any]:
        """
        Return an independent workflow copy containing shot-specific
        prompt values.
        """

        result = copy.deepcopy(workflow)

        positive_prompt = self._get_positive_prompt(
            shot
        )

        negative_prompt = self._get_negative_prompt(
            shot
        )

        if positive_prompt:
            self.set_prompt(
                result,
                positive_prompt,
            )

        if negative_prompt:
            self.set_negative_prompt(
                result,
                negative_prompt,
            )

        return result

    # ============================================================
    # SHOT VALUE EXTRACTION
    # ============================================================

    @staticmethod
    def _get_positive_prompt(
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

            if (
                isinstance(value, str)
                and value.strip()
            ):
                return value.strip()

        return None

    @staticmethod
    def _get_negative_prompt(
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

            if (
                isinstance(value, str)
                and value.strip()
            ):
                return value.strip()

        return None

    # ============================================================
    # POSITIVE PROMPT
    # ============================================================

    @staticmethod
    def set_prompt(
        workflow: dict[str, Any],
        prompt: str,
    ) -> None:

        # Prefer nodes explicitly titled as positive prompt.

        fallback_nodes = []

        for node_id, node in workflow.items():

            if node.get("class_type") != "CLIPTextEncode":
                continue

            inputs = node.get("inputs", {})

            if "text" not in inputs:
                continue

            fallback_nodes.append(node_id)

        if not fallback_nodes:
            raise RuntimeError(
                "No CLIPTextEncode node with "
                "a 'text' input was found."
            )

        # Your current workflows use:
        #
        # positive prompt node = text initially empty
        #
        # negative prompt node = existing negative text
        #
        # So prefer an empty text node.

        for node_id in fallback_nodes:

            text = workflow[node_id][
                "inputs"
            ].get(
                "text",
                "",
            )

            if not str(text).strip():

                workflow[node_id][
                    "inputs"
                ]["text"] = prompt

                return

        # If no empty node exists, choose the first node that does
        # not look like a negative prompt.

        negative_keywords = [
            "negative",
            "worst quality",
            "low quality",
            "blurry",
            "deformed",
            "distorted",
        ]

        for node_id in fallback_nodes:

            existing_text = str(
                workflow[node_id][
                    "inputs"
                ].get(
                    "text",
                    "",
                )
            ).lower()

            if not any(
                keyword in existing_text
                for keyword in negative_keywords
            ):
                workflow[node_id][
                    "inputs"
                ]["text"] = prompt

                return

        # Final fallback.

        workflow[
            fallback_nodes[0]
        ]["inputs"]["text"] = prompt

    # ============================================================
    # NEGATIVE PROMPT
    # ============================================================

    @staticmethod
    def set_negative_prompt(
        workflow: dict[str, Any],
        negative_prompt: str,
    ) -> None:

        negative_keywords = [
            "negative",
            "worst quality",
            "low quality",
            "blurry",
            "deformed",
            "distorted",
            "bad anatomy",
        ]

        candidates = []

        for node_id, node in workflow.items():

            if node.get("class_type") != "CLIPTextEncode":
                continue

            inputs = node.get("inputs", {})

            if "text" not in inputs:
                continue

            existing_text = str(
                inputs.get(
                    "text",
                    "",
                )
            ).lower()

            if any(
                keyword in existing_text
                for keyword in negative_keywords
            ):
                candidates.append(node_id)

        if not candidates:
            return

        workflow[
            candidates[0]
        ]["inputs"][
            "text"
        ] = negative_prompt

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

        # Your workflows use RandomNoise.noise_seed.
        # Other ComfyUI nodes may use simply "seed".

        seed_names = {
            "seed",
            "noise_seed",
        }

        for node in workflow.values():

            inputs = node.get(
                "inputs",
                {},
            )

            for input_name in seed_names:

                if input_name in inputs:

                    inputs[input_name] = seed
                    updated = True

        if not updated:
            raise RuntimeError(
                "No writable seed or noise_seed input "
                "was found in the workflow."
            )

    # ============================================================
    # OUTPUT FILENAME
    # ============================================================

    @staticmethod
    def set_filename_prefix(
        workflow: dict[str, Any],
        filename_prefix: str,
    ) -> None:

        updated = False

        for node in workflow.values():

            inputs = node.get(
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
                "No writable filename_prefix input "
                "was found in the workflow."
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

            if node.get("class_type") != "LoadImage":
                continue

            inputs = node.setdefault(
                "inputs",
                {},
            )

            inputs["image"] = filename

            return

        raise RuntimeError(
            "No LoadImage node was found."
        )

    # ============================================================
    # RAW VIDEO INPUT
    # ============================================================

    @staticmethod
    def set_input_video(
        workflow: dict[str, Any],
        filename: str,
    ) -> None:

        for node in workflow.values():

            if node.get("class_type") != "VHS_LoadVideo":
                continue

            inputs = node.setdefault(
                "inputs",
                {},
            )

            inputs["video"] = filename

            return

        raise RuntimeError(
            "No VHS_LoadVideo node was found."
        )
