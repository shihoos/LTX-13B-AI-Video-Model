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

    Handles non-executable graph nodes:
        - Note
        - Reroute

    Reroute nodes are resolved so downstream executable nodes
    connect directly to the original executable source node.
    """

    # ------------------------------------------------------------
    # UI / documentation nodes that are never sent to the
    # ComfyUI execution API.
    # ------------------------------------------------------------

    UI_ONLY_NODE_TYPES = {
        "Note",
    }

    # ------------------------------------------------------------
    # Graph helper nodes that should not appear in the API prompt.
    #
    # Unlike Note nodes, Reroute nodes can sit between executable
    # nodes, so their connections must be resolved before removal.
    # ------------------------------------------------------------

    REROUTE_NODE_TYPES = {
        "Reroute",
    }

    # ------------------------------------------------------------
    # UI-only widget values that should not be copied into the
    # execution API prompt.
    # ------------------------------------------------------------

    UI_ONLY_WIDGET_NAMES = {
        "upload",
        "choose video to upload",
        "videopreview",
        "control_after_generate",
    }

    def __init__(
        self,
        workflow_path: str | Path,
    ):
        self.workflow_path = Path(
            workflow_path
        )

        if not self.workflow_path.exists():
            raise FileNotFoundError(
                f"Workflow does not exist: "
                f"{self.workflow_path}"
            )

        with self.workflow_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.workflow_json = json.load(
                file
            )

    # ============================================================
    # WORKFLOW → COMFYUI API PROMPT
    # ============================================================

    def to_api_workflow(
        self,
    ) -> dict[str, Any]:
        """
        Convert a normal ComfyUI graph JSON into
        the ComfyUI API prompt format.

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

        # --------------------------------------------------------
        # Already API-format.
        # --------------------------------------------------------

        if self._is_api_workflow(
            workflow
        ):
            return copy.deepcopy(
                workflow
            )

        if not isinstance(
            workflow,
            dict,
        ):
            raise RuntimeError(
                "Workflow must be a dictionary: "
                f"{self.workflow_path}"
            )

        nodes = workflow.get(
            "nodes"
        )

        if not isinstance(
            nodes,
            list,
        ):
            raise RuntimeError(
                "Workflow is neither a valid "
                "ComfyUI graph nor an API workflow: "
                f"{self.workflow_path}"
            )

        # --------------------------------------------------------
        # Build node lookup.
        #
        # node_id -> original graph node
        # --------------------------------------------------------

        node_lookup: dict[
            str,
            dict[str, Any],
        ] = {}

        for node in nodes:

            if not isinstance(
                node,
                dict,
            ):
                continue

            node_id_value = node.get(
                "id"
            )

            if node_id_value is None:
                continue

            node_lookup[
                str(node_id_value)
            ] = node

        # --------------------------------------------------------
        # Build:
        #
        # link_id -> [source_node_id, source_output_index]
        #
        # Example:
        #
        # 5162 -> ["1206", 0]
        # --------------------------------------------------------

        link_lookup: dict[
            int,
            list[Any],
        ] = {}

        for link in workflow.get(
            "links",
            [],
        ):

            if (
                not isinstance(
                    link,
                    list,
                )
                or len(link) < 5
            ):
                continue

            link_id = link[0]

            source_node_id = str(
                link[1]
            )

            source_output_index = link[2]

            link_lookup[
                link_id
            ] = [
                source_node_id,
                source_output_index,
            ]

        # --------------------------------------------------------
        # Executable nodes.
        #
        # Note and Reroute nodes are intentionally excluded.
        # --------------------------------------------------------

        executable_node_ids = {
            str(
                node.get("id")
            )
            for node in nodes
            if (
                isinstance(
                    node,
                    dict,
                )
                and node.get("id") is not None
                and node.get("type")
                not in self.UI_ONLY_NODE_TYPES
                and node.get("type")
                not in self.REROUTE_NODE_TYPES
            )
        }

        # --------------------------------------------------------
        # Resolve a source connection.
        #
        # If:
        #
        # Executable A
        #      ↓
        # Reroute
        #      ↓
        # Reroute
        #      ↓
        # Executable B
        #
        # then Executable B receives:
        #
        # ["A", output_index]
        #
        # instead of referencing a Reroute.
        # --------------------------------------------------------

        def resolve_source(
            source_node_id: str,
            source_output_index: int,
        ) -> list[Any]:

            visited: set[str] = set()

            current_node_id = str(
                source_node_id
            )

            current_output_index = (
                source_output_index
            )

            while True:

                # Prevent accidental infinite loops.
                if current_node_id in visited:

                    raise RuntimeError(
                        "Reroute cycle detected while "
                        f"resolving node {source_node_id}."
                    )

                visited.add(
                    current_node_id
                )

                source_node = node_lookup.get(
                    current_node_id
                )

                if source_node is None:

                    raise RuntimeError(
                        "Graph references missing "
                        f"source node {current_node_id}."
                    )

                source_type = source_node.get(
                    "type"
                )

                # ------------------------------------------------
                # Real executable node reached.
                # ------------------------------------------------

                if (
                    source_type
                    not in self.REROUTE_NODE_TYPES
                ):

                    if (
                        current_node_id
                        not in executable_node_ids
                    ):

                        raise RuntimeError(
                            "Graph connection resolves to "
                            "a skipped/non-executable node "
                            f"{current_node_id} "
                            f"({source_type})."
                        )

                    return [
                        current_node_id,
                        current_output_index,
                    ]

                # ------------------------------------------------
                # Current node is a Reroute.
                #
                # Find its incoming graph connection and continue
                # walking upstream.
                # ------------------------------------------------

                reroute_inputs = source_node.get(
                    "inputs",
                    [],
                )

                reroute_link_id = None

                if isinstance(
                    reroute_inputs,
                    list,
                ):

                    # Prefer the first actual linked input.
                    for reroute_input in reroute_inputs:

                        if not isinstance(
                            reroute_input,
                            dict,
                        ):
                            continue

                        link_id = reroute_input.get(
                            "link"
                        )

                        if link_id is not None:

                            reroute_link_id = link_id
                            break

                if reroute_link_id is None:

                    raise RuntimeError(
                        "Reroute node "
                        f"{current_node_id} "
                        "has no incoming connection."
                    )

                upstream_source = (
                    link_lookup.get(
                        reroute_link_id
                    )
                )

                if upstream_source is None:

                    raise RuntimeError(
                        "Reroute node "
                        f"{current_node_id} "
                        "references missing link "
                        f"{reroute_link_id}."
                    )

                current_node_id = str(
                    upstream_source[0]
                )

                current_output_index = (
                    upstream_source[1]
                )

        # --------------------------------------------------------
        # Convert executable nodes into API prompt format.
        # --------------------------------------------------------

        api_workflow: dict[
            str,
            Any,
        ] = {}

        for node in nodes:

            if not isinstance(
                node,
                dict,
            ):
                continue

            node_id_value = node.get(
                "id"
            )

            class_type = node.get(
                "type"
            )

            if node_id_value is None:

                raise RuntimeError(
                    "Workflow contains a node "
                    "without an ID."
                )

            if not class_type:

                raise RuntimeError(
                    f"Node {node_id_value} "
                    "has no type."
                )

            # ----------------------------------------------------
            # Skip Note / UI-only nodes.
            # ----------------------------------------------------

            if (
                class_type
                in self.UI_ONLY_NODE_TYPES
            ):
                continue

            # ----------------------------------------------------
            # Skip Reroute nodes.
            #
            # Their downstream references are resolved directly
            # to the upstream executable node.
            # ----------------------------------------------------

            if (
                class_type
                in self.REROUTE_NODE_TYPES
            ):
                continue

            node_id = str(
                node_id_value
            )

            api_inputs: dict[
                str,
                Any,
            ] = {}

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

            named_widgets = node.get(
                "widgets_values_named",
                {},
            )

            if isinstance(
                named_widgets,
                dict,
            ):

                for name, value in (
                    named_widgets.items()
                ):

                    normalized_name = (
                        str(name)
                        .strip()
                        .lower()
                    )

                    if (
                        normalized_name
                        in self.UI_ONLY_WIDGET_NAMES
                    ):
                        continue

                    api_inputs[
                        name
                    ] = copy.deepcopy(
                        value
                    )

            # ----------------------------------------------------
            # Graph connections
            # ----------------------------------------------------

            node_inputs = node.get(
                "inputs",
                [],
            )

            if isinstance(
                node_inputs,
                list,
            ):

                for input_def in (
                    node_inputs
                ):

                    if not isinstance(
                        input_def,
                        dict,
                    ):
                        continue

                    input_name = (
                        input_def.get(
                            "name"
                        )
                    )

                    if not input_name:
                        continue

                    link_id = (
                        input_def.get(
                            "link"
                        )
                    )

                    # ------------------------------------------------
                    # Connected graph input.
                    # ------------------------------------------------

                    if link_id is not None:

                        source = (
                            link_lookup.get(
                                link_id
                            )
                        )

                        if source is None:

                            raise RuntimeError(
                                f"Node {node_id} "
                                f"input '{input_name}' "
                                f"references missing "
                                f"link {link_id}."
                            )

                        resolved_source = (
                            resolve_source(
                                str(source[0]),
                                source[1],
                            )
                        )

                        api_inputs[
                            input_name
                        ] = copy.deepcopy(
                            resolved_source
                        )

                    # ------------------------------------------------
                    # Literal/default value.
                    # ------------------------------------------------

                    elif (
                        "value"
                        in input_def
                    ):

                        api_inputs[
                            input_name
                        ] = copy.deepcopy(
                            input_def[
                                "value"
                            ]
                        )

            api_workflow[
                node_id
            ] = {
                "class_type": class_type,
                "inputs": api_inputs,
            }

        if not api_workflow:

            raise RuntimeError(
                "No executable nodes found in: "
                f"{self.workflow_path}"
            )

        return api_workflow

    # ============================================================
    # CHECK API WORKFLOW
    # ============================================================

    @staticmethod
    def _is_api_workflow(
        workflow: Any,
    ) -> bool:
        """
        Return True only when the dictionary already
        looks like a ComfyUI API prompt.
        """

        if (
            not isinstance(
                workflow,
                dict,
            )
            or not workflow
        ):
            return False

        if "nodes" in workflow:
            return False

        return all(
            isinstance(
                value,
                dict,
            )
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
        Apply shot-specific prompt values to
        a copy of the API workflow.
        """

        result = copy.deepcopy(
            workflow
        )

        positive_prompt = (
            self._get_shot_value(
                shot,
                [
                    "prompt",
                    "positive_prompt",
                    "generation_prompt",
                    "description",
                    "visual_prompt",
                ],
            )
        )

        negative_prompt = (
            self._get_shot_value(
                shot,
                [
                    "negative_prompt",
                    "negative",
                ],
            )
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

    @staticmethod
    def _get_shot_value(
        shot: Any,
        field_names: list[str],
    ) -> str | None:

        if isinstance(
            shot,
            dict,
        ):

            for field in field_names:

                value = shot.get(
                    field
                )

                if (
                    isinstance(
                        value,
                        str,
                    )
                    and value.strip()
                ):

                    return value.strip()

            return None

        for field in field_names:

            value = getattr(
                shot,
                field,
                None,
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):

                return value.strip()

        return None

    # ============================================================
    # PROMPTS
    # ============================================================

    @staticmethod
    def _get_clip_text_nodes(
        workflow: dict[str, Any],
    ) -> list[
        tuple[
            str,
            dict[str, Any],
        ]
    ]:

        result = []

        for node_id, node in (
            workflow.items()
        ):

            if (
                node.get(
                    "class_type"
                )
                != "CLIPTextEncode"
            ):
                continue

            inputs = node.get(
                "inputs",
                {},
            )

            if "text" in inputs:

                result.append(
                    (
                        node_id,
                        node,
                    )
                )

        return result

    @staticmethod
    def set_prompt(
        workflow: dict[str, Any],
        prompt: str,
    ) -> None:

        candidates = (
            ComfyWorkflowAdapter
            ._get_clip_text_nodes(
                workflow
            )
        )

        if not candidates:

            raise RuntimeError(
                "No CLIPTextEncode node "
                "with a text input was found."
            )

        # Prefer an empty prompt node.
        for _, node in candidates:

            current = str(
                node["inputs"].get(
                    "text",
                    "",
                )
            ).strip()

            if not current:

                node[
                    "inputs"
                ][
                    "text"
                ] = prompt

                return

        # Otherwise select the first node
        # that does not look negative.
        negative_markers = (
            "negative",
            "worst quality",
            "low quality",
            "blurry",
            "deformed",
            "distorted",
            "bad anatomy",
            "watermark",
            "ugly",
            "fused fingers",
        )

        for _, node in candidates:

            current = str(
                node["inputs"].get(
                    "text",
                    "",
                )
            ).lower()

            if not any(
                marker in current
                for marker in negative_markers
            ):

                node[
                    "inputs"
                ][
                    "text"
                ] = prompt

                return

        # Final fallback.
        candidates[0][1][
            "inputs"
        ][
            "text"
        ] = prompt

    @staticmethod
    def set_negative_prompt(
        workflow: dict[str, Any],
        negative_prompt: str,
    ) -> None:

        candidates = (
            ComfyWorkflowAdapter
            ._get_clip_text_nodes(
                workflow
            )
        )

        if not candidates:

            raise RuntimeError(
                "No CLIPTextEncode node "
                "with a text input was found."
            )

        negative_markers = (
            "negative",
            "worst quality",
            "low quality",
            "blurry",
            "deformed",
            "distorted",
            "bad anatomy",
            "watermark",
            "ugly",
            "fused fingers",
        )

        for _, node in candidates:

            current = str(
                node["inputs"].get(
                    "text",
                    "",
                )
            ).lower()

            if any(
                marker in current
                for marker in negative_markers
            ):

                node[
                    "inputs"
                ][
                    "text"
                ] = negative_prompt

                return

        # If there are at least two text nodes,
        # the second is the safest fallback.
        if len(candidates) >= 2:

            candidates[1][1][
                "inputs"
            ][
                "text"
            ] = negative_prompt

            return

        raise RuntimeError(
            "Could not identify a negative "
            "CLIPTextEncode node."
        )

    # ============================================================
    # SEED
    # ============================================================

    @staticmethod
    def set_seed(
        workflow: dict[str, Any],
        seed: int,
    ) -> None:

        seed = int(
            seed
        )

        updated = False

        for node in workflow.values():

            inputs = node.get(
                "inputs",
                {},
            )

            for seed_name in (
                "noise_seed",
                "seed",
            ):

                if seed_name in inputs:

                    inputs[
                        seed_name
                    ] = seed

                    updated = True

        if not updated:

            raise RuntimeError(
                "No seed or noise_seed input "
                "was found."
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

            inputs = node.get(
                "inputs",
                {},
            )

            if (
                "filename_prefix"
                in inputs
            ):

                inputs[
                    "filename_prefix"
                ] = filename_prefix

                updated = True

        if not updated:

            raise RuntimeError(
                "No filename_prefix input "
                "was found."
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

            if (
                node.get(
                    "class_type"
                )
                != "LoadImage"
            ):
                continue

            inputs = node.setdefault(
                "inputs",
                {},
            )

            inputs[
                "image"
            ] = filename

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

            if (
                node.get(
                    "class_type"
                )
                != "VHS_LoadVideo"
            ):
                continue

            inputs = node.setdefault(
                "inputs",
                {},
            )

            inputs[
                "video"
            ] = filename

            return

        raise RuntimeError(
            "No VHS_LoadVideo node was found."
        )
