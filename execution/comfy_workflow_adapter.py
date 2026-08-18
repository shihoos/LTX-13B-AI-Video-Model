from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ComfyWorkflowAdapter:

    UI_ONLY_NODE_TYPES = {
        "Note",
    }

    REROUTE_NODE_TYPES = {
        "Reroute",
    }

    UI_ONLY_WIDGET_NAMES = {
        "upload",
        "choose video to upload",
        "videopreview",
        "control_after_generate",
    }

    LEGACY_LATENT_LOADER = (
        "LTXVLatentUpsamplerModelLoader"
    )

    MODERN_LATENT_LOADER = (
        "LatentUpscaleModelLoader"
    )

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

        self.workflow_json = json.loads(
            self.workflow_path.read_text(
                encoding="utf-8"
            )
        )

    def to_api_workflow(self):
        workflow = self.workflow_json

        if self._is_api_workflow(
            workflow
        ):

            result = copy.deepcopy(
                workflow
            )

            self.apply_modern_compatibility(
                result
            )

            return result

        nodes = workflow.get(
            "nodes"
        )

        if not isinstance(
            nodes,
            list,
        ):

            raise RuntimeError(
                f"Invalid ComfyUI workflow: "
                f"{self.workflow_path}"
            )

        node_lookup = {}

        for node in nodes:

            if (
                isinstance(node, dict)
                and node.get("id") is not None
            ):

                node_lookup[
                    str(node["id"])
                ] = node

        link_lookup = {}

        for link in workflow.get(
            "links",
            [],
        ):

            if (
                isinstance(link, list)
                and len(link) >= 5
            ):

                link_lookup[
                    link[0]
                ] = [
                    str(link[1]),
                    link[2],
                ]

        executable_ids = {
            str(node["id"])
            for node in nodes
            if (
                isinstance(node, dict)
                and node.get("id") is not None
                and node.get("type")
                not in self.UI_ONLY_NODE_TYPES
                and node.get("type")
                not in self.REROUTE_NODE_TYPES
            )
        }

        def resolve_source(
            source_node_id,
            output_index,
        ):

            visited = set()

            current_id = str(
                source_node_id
            )

            current_output = (
                output_index
            )

            while True:

                if current_id in visited:

                    raise RuntimeError(
                        "Reroute cycle detected: "
                        f"{current_id}"
                    )

                visited.add(
                    current_id
                )

                source = node_lookup.get(
                    current_id
                )

                if source is None:

                    raise RuntimeError(
                        f"Missing source node: "
                        f"{current_id}"
                    )

                source_type = source.get(
                    "type"
                )

                if (
                    source_type
                    not in self.REROUTE_NODE_TYPES
                ):

                    if (
                        current_id
                        not in executable_ids
                    ):

                        raise RuntimeError(
                            "Resolved source is not executable: "
                            f"{current_id}"
                        )

                    return [
                        current_id,
                        current_output,
                    ]

                incoming = None

                for item in source.get(
                    "inputs",
                    [],
                ):

                    if (
                        isinstance(
                            item,
                            dict,
                        )
                        and item.get(
                            "link"
                        )
                        is not None
                    ):

                        incoming = (
                            item["link"]
                        )

                        break

                if incoming is None:

                    raise RuntimeError(
                        "Reroute has no input link: "
                        f"{current_id}"
                    )

                upstream = (
                    link_lookup.get(
                        incoming
                    )
                )

                if upstream is None:

                    raise RuntimeError(
                        "Missing reroute link: "
                        f"{incoming}"
                    )

                current_id = str(
                    upstream[0]
                )

                current_output = (
                    upstream[1]
                )

        api = {}

        for node in nodes:

            if not isinstance(
                node,
                dict,
            ):
                continue

            node_id = node.get(
                "id"
            )

            class_type = node.get(
                "type"
            )

            if (
                node_id is None
                or not class_type
            ):

                continue

            if (
                class_type
                in self.UI_ONLY_NODE_TYPES
            ):

                continue

            if (
                class_type
                in self.REROUTE_NODE_TYPES
            ):

                continue

            api_id = str(
                node_id
            )

            inputs = {}

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

                    if (
                        str(name)
                        .strip()
                        .lower()
                        in self.UI_ONLY_WIDGET_NAMES
                    ):

                        continue

                    inputs[name] = (
                        copy.deepcopy(
                            value
                        )
                    )

            for input_def in node.get(
                "inputs",
                [],
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

                if link_id is not None:

                    source = (
                        link_lookup.get(
                            link_id
                        )
                    )

                    if source is None:

                        raise RuntimeError(
                            f"Input {input_name} "
                            f"references missing link "
                            f"{link_id}"
                        )

                    inputs[
                        input_name
                    ] = resolve_source(
                        str(source[0]),
                        source[1],
                    )

                elif (
                    "value"
                    in input_def
                ):

                    inputs[
                        input_name
                    ] = copy.deepcopy(
                        input_def[
                            "value"
                        ]
                    )

            api[
                api_id
            ] = {
                "class_type":
                    class_type,

                "inputs":
                    inputs,
            }

        if not api:

            raise RuntimeError(
                "No executable nodes found."
            )

        self.apply_modern_compatibility(
            api
        )

        return api

    @classmethod
    def apply_modern_compatibility(
        cls,
        workflow,
    ):

        for node in workflow.values():

            if not isinstance(
                node,
                dict,
            ):

                continue

            if (
                node.get(
                    "class_type"
                )
                != cls.LEGACY_LATENT_LOADER
            ):

                continue

            old_inputs = node.get(
                "inputs",
                {},
            )

            model_name = (
                old_inputs.get(
                    "upscale_model"
                )
                or old_inputs.get(
                    "model_name"
                )
            )

            if not model_name:

                raise RuntimeError(
                    "Legacy spatial loader "
                    "has no model filename."
                )

            node[
                "class_type"
            ] = (
                cls.MODERN_LATENT_LOADER
            )

            node[
                "inputs"
            ] = {
                "model_name":
                    model_name,
            }

        return workflow

    @staticmethod
    def set_prompt(
        workflow,
        text,
    ):

        nodes = [
            node
            for node in workflow.values()
            if (
                node.get(
                    "class_type"
                )
                == "CLIPTextEncode"
            )
        ]

        if not nodes:

            raise RuntimeError(
                "No CLIPTextEncode nodes."
            )

        nodes[0].setdefault(
            "inputs",
            {},
        )["text"] = text

    @staticmethod
    def set_negative_prompt(
        workflow,
        text,
    ):

        nodes = [
            node
            for node in workflow.values()
            if (
                node.get(
                    "class_type"
                )
                == "CLIPTextEncode"
            )
        ]

        if len(nodes) < 2:

            raise RuntimeError(
                "Positive/negative CLIPTextEncode "
                "pair not found."
            )

        nodes[1].setdefault(
            "inputs",
            {},
        )["text"] = text

    @staticmethod
    def set_seed(
        workflow,
        seed,
    ):

        found = False

        for node in workflow.values():

            class_type = node.get(
                "class_type"
            )

            inputs = node.setdefault(
                "inputs",
                {},
            )

            if (
                class_type
                == "RandomNoise"
            ):

                inputs[
                    "noise_seed"
                ] = int(seed)

                found = True

            elif (
                class_type
                == "Set VAE Decoder Noise"
                and "seed"
                in inputs
            ):

                inputs[
                    "seed"
                ] = int(seed)

                found = True

        if not found:

            raise RuntimeError(
                "No supported seed input."
            )

    @staticmethod
    def set_filename_prefix(
        workflow,
        prefix,
    ):

        nodes = [
            node
            for node in workflow.values()
            if (
                node.get(
                    "class_type"
                )
                == "VHS_VideoCombine"
            )
        ]

        if not nodes:

            raise RuntimeError(
                "No VHS_VideoCombine node."
            )

        for node in nodes:

            node.setdefault(
                "inputs",
                {},
            )[
                "filename_prefix"
            ] = prefix

    set_output_prefix = (
        set_filename_prefix
    )

    @staticmethod
    def set_input_image(
        workflow,
        filename,
    ):

        nodes = [
            node
            for node in workflow.values()
            if (
                node.get(
                    "class_type"
                )
                == "LoadImage"
            )
        ]

        if not nodes:

            raise RuntimeError(
                "No LoadImage node."
            )

        for node in nodes:

            node.setdefault(
                "inputs",
                {},
            )[
                "image"
            ] = filename

    @staticmethod
    def set_input_video(
        workflow,
        filename,
    ):

        nodes = [
            node
            for node in workflow.values()
            if (
                node.get(
                    "class_type"
                )
                == "VHS_LoadVideo"
            )
        ]

        if not nodes:

            raise RuntimeError(
                "No VHS_LoadVideo node."
            )

        for node in nodes:

            node.setdefault(
                "inputs",
                {},
            )[
                "video"
            ] = filename

    @classmethod
    def validate_modern_detailer(
        cls,
        workflow,
    ):

        for node in workflow.values():

            if (
                isinstance(
                    node,
                    dict,
                )
                and node.get(
                    "class_type"
                )
                == cls.LEGACY_LATENT_LOADER
            ):

                raise RuntimeError(
                    "Legacy spatial loader "
                    "survived conversion."
                )

        if not any(
            isinstance(
                node,
                dict,
            )
            and node.get(
                "class_type"
            )
            == cls.MODERN_LATENT_LOADER
            for node in workflow.values()
        ):

            raise RuntimeError(
                "Modern detailer has no "
                "LatentUpscaleModelLoader."
            )

    def apply_shot(
        self,
        workflow,
        shot,
    ):

        result = copy.deepcopy(
            workflow
        )

        positive = (
            self._get_shot_value(
                shot,
                (
                    "visual_prompt",
                    "prompt",
                    "positive_prompt",
                    "generation_prompt",
                    "description",
                ),
            )
        )

        negative = (
            self._get_shot_value(
                shot,
                (
                    "negative_prompt",
                    "negative",
                ),
            )
        )

        if positive:

            self.set_prompt(
                result,
                positive,
            )

        if negative:

            self.set_negative_prompt(
                result,
                negative,
            )

        return result

    @staticmethod
    def _get_shot_value(
        shot,
        fields,
    ):

        if isinstance(
            shot,
            dict,
        ):

            for field in fields:

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

        for field in fields:

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

    @staticmethod
    def _is_api_workflow(
        workflow,
    ):

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
            and "class_type"
            in value
            and "inputs"
            in value
            for value in workflow.values()
        )
