import copy
import json

from pathlib import Path


class ComfyWorkflowAdapter:

    """
    Converts a ComfyUI graph-format workflow into
    an API-format prompt and applies shot-specific values.
    """

    def __init__(
        self,
        workflow_path: Path,
    ):

        self.workflow_path = (
            Path(workflow_path)
        )

        with self.workflow_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.workflow = json.load(
                file
            )

    def _link_map(self):

        result = {}

        for node in self.workflow.get(
            "nodes",
            [],
        ):

            node_id = str(
                node["id"]
            )

            for input_data in node.get(
                "inputs",
                [],
            ):

                link = (
                    input_data.get(
                        "link"
                    )
                )

                if link is None:
                    continue

                result[
                    (
                        node_id,
                        input_data["name"],
                    )
                ] = link

        links = {}

        for link in self.workflow.get(
            "links",
            [],
        ):

            if len(link) < 6:
                continue

            link_id = link[0]

            source_node = link[1]
            source_slot = link[2]

            target_node = link[3]
            target_slot = link[4]

            links[link_id] = {
                "source_node": str(
                    source_node
                ),
                "source_slot": (
                    source_slot
                ),
                "target_node": str(
                    target_node
                ),
                "target_slot": (
                    target_slot
                ),
            }

        resolved = {}

        for key, link_id in result.items():

            link_info = links.get(
                link_id
            )

            if link_info:

                resolved[key] = (
                    link_info
                )

        return resolved

    def _node_api_inputs(
        self,
        node: dict,
        link_map: dict,
    ) -> dict:

        api_inputs = {}

        node_id = str(
            node["id"]
        )

        widgets_named = node.get(
            "widgets_values_named",
            {},
        )

        if isinstance(
            widgets_named,
            dict,
        ):

            api_inputs.update(
                copy.deepcopy(
                    widgets_named
                )
            )

        for input_data in node.get(
            "inputs",
            [],
        ):

            name = input_data[
                "name"
            ]

            key = (
                node_id,
                name,
            )

            connection = (
                link_map.get(
                    key
                )
            )

            if connection:

                api_inputs[name] = [
                    connection[
                        "source_node"
                    ],
                    connection[
                        "source_slot"
                    ],
                ]

        return api_inputs

    def to_api_workflow(self):

        link_map = (
            self._link_map()
        )

        api = {}

        for node in self.workflow.get(
            "nodes",
            [],
        ):

            node_id = str(
                node["id"]
            )

            api[node_id] = {
                "class_type": node[
                    "type"
                ],
                "inputs": (
                    self._node_api_inputs(
                        node,
                        link_map,
                    )
                ),
            }

        return api

    @staticmethod
    def _find_text_nodes(
        workflow: dict,
    ):

        matches = []

        for node_id, node in (
            workflow.items()
        ):

            if node.get(
                "class_type"
            ) != "CLIPTextEncode":

                continue

            matches.append(
                (
                    node_id,
                    node,
                )
            )

        return matches

    def apply_shot(
        self,
        api_workflow: dict,
        shot,
    ) -> dict:

        workflow = copy.deepcopy(
            api_workflow
        )

        text_nodes = (
            self._find_text_nodes(
                workflow
            )
        )

        for node_id, node in (
            text_nodes
        ):

            existing = str(
                node[
                    "inputs"
                ].get(
                    "text",
                    "",
                )
            )

            lowered = (
                existing.lower()
            )

            if (
                "low quality"
                in lowered
                or "worst quality"
                in lowered
                or "watermark"
                in lowered
            ):

                negative = (
                    shot.negative_prompt
                    or existing
                )

                node[
                    "inputs"
                ][
                    "text"
                ] = negative

            else:

                node[
                    "inputs"
                ][
                    "text"
                ] = shot.visual_prompt

        return workflow

    @staticmethod
    def set_filename_prefix(
        workflow: dict,
        prefix: str,
    ):

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
                ] = prefix

    @staticmethod
    def set_seed(
        workflow: dict,
        seed: int,
    ):

        for node in workflow.values():

            inputs = node.get(
                "inputs",
                {},
            )

            for key in (
                "seed",
                "noise_seed",
            ):

                if key in inputs:

                    inputs[
                        key
                    ] = seed
