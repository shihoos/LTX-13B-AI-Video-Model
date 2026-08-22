from __future__ import annotations

import copy
import json
import re
from pathlib import Path


class H3WorkflowBuilder:

    MAX_IMAGES = 9
    MAX_VIDEOS = 3
    MAX_AUDIO = 3

    DEFAULT_WIDTH = 960
    DEFAULT_HEIGHT = 544
    DEFAULT_FRAMES = 124
    DEFAULT_FPS = 24
    DEFAULT_STEPS = 14
    DEFAULT_SCHEDULER = "beta"
    DEFAULT_SAMPLER = "res_multistep"

    REF2VA_MODEL = (
        "minimax_h3_ref2va_pruned-Q4_K_M.gguf"
    )

    TEXT_ENCODER = (
        "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    )

    VIDEO_VAE = (
        "minimax_h3_video_vae_fp16.safetensors"
    )

    AUDIO_VAE = (
        "minimax_h3_audio_vae_fp32.safetensors"
    )

    def __init__(
        self,
        project_root: Path,
        object_info: dict | None = None,
    ):
        self.project_root = Path(
            project_root
        )

        self.object_info = (
            object_info or {}
        )

        self.workflow_root = (
            self.project_root
            / "workflows"
            / "MiniMax-H3"
        )

        self.base_root = (
            self.workflow_root
            / "base"
        )

    # ------------------------------------------------------------------
    # WORKFLOW FILES
    # ------------------------------------------------------------------

    def _load_ui(
        self,
        filename: str,
    ) -> dict:

        path = (
            self.base_root
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"H3 workflow not found: {path}"
            )

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    # ------------------------------------------------------------------
    # BASIC NODE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _nodes(
        workflow: dict,
    ) -> list[dict]:

        nodes = workflow.get(
            "nodes",
            []
        )

        if not isinstance(
            nodes,
            list,
        ):
            raise RuntimeError(
                "UI workflow does not contain "
                "a valid nodes array."
            )

        return nodes

    @staticmethod
    def _title(
        node: dict,
    ) -> str:
        return str(
            node.get(
                "title",
                ""
            )
        ).lower()

    @staticmethod
    def _class_name(
        node: dict,
    ) -> str:
        return str(
            node.get(
                "type",
                ""
            )
        )

    @staticmethod
    def _numeric_tag(
        title: str,
        tag: str,
    ) -> int:

        match = re.search(
            rf"{re.escape(tag)}\s*(\d+)",
            title,
            flags=re.IGNORECASE,
        )

        if match:
            return int(
                match.group(1)
            )

        return 9999

    @classmethod
    def _find_ui_nodes(
        cls,
        workflow: dict,
        node_type: str,
    ) -> list[dict]:

        return [
            node
            for node in cls._nodes(
                workflow
            )
            if node.get("type") == node_type
        ]

    @classmethod
    def _find_ui_reference_images(
        cls,
        workflow: dict,
    ) -> list[dict]:

        nodes = [
            node
            for node in cls._nodes(
                workflow
            )
            if node.get("type") == "LoadImage"
            and (
                "<Picture"
                in node.get(
                    "title",
                    ""
                )
            )
        ]

        return sorted(
            nodes,
            key=lambda node: cls._numeric_tag(
                node.get(
                    "title",
                    ""
                ),
                "<Picture",
            ),
        )

    # ------------------------------------------------------------------
    # UI PATCHING
    # ------------------------------------------------------------------

    @staticmethod
    def _set_widget(
        node: dict,
        index: int,
        value,
    ) -> None:

        widgets = node.get(
            "widgets_values"
        )

        if not isinstance(
            widgets,
            list,
        ):
            raise RuntimeError(
                "Cannot patch node without "
                "widgets_values: "
                f"{node.get('type')}"
            )

        while len(widgets) <= index:
            widgets.append(
                None
            )

        widgets[index] = value

    @classmethod
    def _patch_common_ui_values(
        cls,
        workflow: dict,
    ) -> None:

        for node in cls._nodes(
            workflow
        ):
            node_type = node.get(
                "type"
            )

            title = node.get(
                "title",
                ""
            ).lower()

            if node_type == "H3ModelLoaderAny":
                cls._set_widget(
                    node,
                    0,
                    cls.REF2VA_MODEL,
                )

            elif node_type == "H3ClipLoaderAny":
                cls._set_widget(
                    node,
                    0,
                    cls.TEXT_ENCODER,
                )

                cls._set_widget(
                    node,
                    1,
                    "minimax",
                )

            elif node_type == "VAELoader":
                if "audio" in title:
                    cls._set_widget(
                        node,
                        0,
                        cls.AUDIO_VAE,
                    )

                elif "video" in title:
                    cls._set_widget(
                        node,
                        0,
                        cls.VIDEO_VAE,
                    )

            elif node_type == "BasicScheduler":
                cls._set_widget(
                    node,
                    0,
                    cls.DEFAULT_SCHEDULER,
                )

                cls._set_widget(
                    node,
                    1,
                    cls.DEFAULT_STEPS,
                )

                cls._set_widget(
                    node,
                    2,
                    1.0,
                )

            elif node_type == "KSamplerSelect":
                cls._set_widget(
                    node,
                    0,
                    cls.DEFAULT_SAMPLER,
                )

            elif node_type == "CreateVideo":
                cls._set_widget(
                    node,
                    0,
                    cls.DEFAULT_FPS,
                )

    @classmethod
    def _patch_duration_ui(
        cls,
        workflow: dict,
        duration_seconds: float,
    ) -> None:

        for node in cls._nodes(
            workflow
        ):
            if (
                node.get("type")
                == "PrimitiveFloat"
                and
                "duration"
                in node.get(
                    "title",
                    ""
                ).lower()
            ):
                cls._set_widget(
                    node,
                    0,
                    float(duration_seconds),
                )

    @classmethod
    def _patch_reference_images_ui(
        cls,
        workflow: dict,
        image_files: list[str],
    ) -> None:

        nodes = cls._find_ui_reference_images(
            workflow
        )

        for index, filename in enumerate(
            image_files[: cls.MAX_IMAGES]
        ):
            if index >= len(nodes):
                break

            cls._set_widget(
                nodes[index],
                0,
                filename,
            )

    @classmethod
    def _patch_standalone_audio_ui(
        cls,
        workflow: dict,
        audio_files: list[str],
    ) -> None:

        if not audio_files:
            return

        candidates = []

        for node in cls._nodes(
            workflow
        ):
            if node.get(
                "type"
            ) != "LoadAudio":
                continue

            title = node.get(
                "title",
                ""
            ).lower()

            if (
                "voice reference"
                in title
                or
                "reference audio"
                in title
                or
                "audio"
                in title
            ):
                candidates.append(
                    node
                )

        candidates = sorted(
            candidates,
            key=lambda node: (
                0
                if "voice reference"
                in node.get(
                    "title",
                    ""
                ).lower()
                else 1
            )
        )

        for index, filename in enumerate(
            audio_files[: cls.MAX_AUDIO]
        ):
            if index >= len(candidates):
                break

            cls._set_widget(
                candidates[index],
                0,
                filename,
            )

    @classmethod
    def _patch_prompt_ui(
        cls,
        workflow: dict,
        prompt: str,
        chained: bool = False,
    ) -> None:

        nodes = cls._nodes(
            workflow
        )

        if chained:
            candidates = [
                node
                for node in nodes
                if (
                    node.get("type")
                    == "PrimitiveStringMultiline"
                    and
                    (
                        "script + bindings"
                        in node.get(
                            "title",
                            ""
                        ).lower()
                        or
                        (
                            "script"
                            in node.get(
                                "title",
                                ""
                            ).lower()
                            and
                            "prompt"
                            not in node.get(
                                "title",
                                ""
                            ).lower()
                        )
                    )
                )
            ]
        else:
            candidates = [
                node
                for node in nodes
                if (
                    node.get("type")
                    == "PrimitiveStringMultiline"
                    and
                    (
                        "prompt"
                        in node.get(
                            "title",
                            ""
                        ).lower()
                    )
                )
            ]

        if not candidates:
            raise RuntimeError(
                "Could not find H3 prompt input "
                "in workflow."
            )

        cls._set_widget(
            candidates[0],
            0,
            prompt,
        )

    # ------------------------------------------------------------------
    # API GRAPH HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _api_nodes(
        workflow: dict,
    ) -> dict:

        if not isinstance(
            workflow,
            dict,
        ):
            raise RuntimeError(
                "API workflow is not a dictionary."
            )

        return workflow

    @classmethod
    def _find_api_nodes(
        cls,
        workflow: dict,
        class_type: str,
    ) -> list[tuple[str, dict]]:

        result = []

        for node_id, node in workflow.items():
            if (
                isinstance(node, dict)
                and node.get(
                    "class_type"
                ) == class_type
            ):
                result.append(
                    (
                        str(node_id),
                        node,
                    )
                )

        return result

    @classmethod
    def _find_first_api_node(
        cls,
        workflow: dict,
        class_type: str,
    ) -> tuple[str, dict]:

        nodes = cls._find_api_nodes(
            workflow,
            class_type,
        )

        if not nodes:
            raise RuntimeError(
                f"Required H3 node missing: "
                f"{class_type}"
            )

        return nodes[0]

    @classmethod
    def _next_id(
        cls,
        workflow: dict,
    ) -> str:

        numeric = []

        for node_id in workflow:
            try:
                numeric.append(
                    int(node_id)
                )
            except (
                ValueError,
                TypeError,
            ):
                pass

        if not numeric:
            return "1"

        return str(
            max(numeric) + 1
        )

    @classmethod
    def _set_api_input(
        cls,
        node: dict,
        key: str,
        value,
    ) -> None:

        inputs = node.setdefault(
            "inputs",
            {}
        )

        inputs[key] = value

    @classmethod
    def _set_api_inputs_if_present(
        cls,
        node: dict,
        values: dict,
    ) -> None:

        inputs = node.setdefault(
            "inputs",
            {}
        )

        for key, value in values.items():
            if (
                key in inputs
                or
                key in {
                    "width",
                    "height",
                    "length",
                    "prompt",
                    "seed",
                    "steps",
                    "scheduler",
                    "sampler_name",
                    "frames_per_shot",
                    "shot_count",
                    "script",
                }
            ):
                inputs[key] = value

    @classmethod
    def _find_api_ref2va(
        cls,
        workflow: dict,
    ) -> tuple[str, dict]:

        return cls._find_first_api_node(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )

    # ------------------------------------------------------------------
    # REFERENCE LOADERS
    # ------------------------------------------------------------------

    @classmethod
    def _add_load_image(
        cls,
        workflow: dict,
        filename: str,
    ) -> str:

        node_id = cls._next_id(
            workflow
        )

        workflow[node_id] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": filename,
            },
            "_meta": {
                "title": (
                    "H3 generated reference image"
                ),
            },
        }

        return node_id

    @classmethod
    def _add_vhs_video(
        cls,
        workflow: dict,
        filename: str,
    ) -> str:

        node_id = cls._next_id(
            workflow
        )

        workflow[node_id] = {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": filename,
                "force_rate": 24,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
            "_meta": {
                "title": (
                    "H3 reference video"
                ),
            },
        }

        return node_id

    # ------------------------------------------------------------------
    # REFERENCE GRAPH PATCHING
    # ------------------------------------------------------------------

    @classmethod
    def _patch_api_references(
        cls,
        workflow: dict,
        image_files: list[str],
        video_files: list[str],
        audio_files: list[str],
    ) -> None:

        _, ref_node = cls._find_api_ref2va(
            workflow
        )

        inputs = ref_node.setdefault(
            "inputs",
            {}
        )

        # --------------------------------------------------------------
        # Images
        # --------------------------------------------------------------

        for index, filename in enumerate(
            image_files[: cls.MAX_IMAGES]
        ):
            key = (
                f"ref_images.ref_image_{index}"
            )

            existing = inputs.get(
                key
            )

            if existing is not None:
                continue

            loader_id = cls._add_load_image(
                workflow,
                filename,
            )

            inputs[key] = [
                loader_id,
                0,
            ]

        # --------------------------------------------------------------
        # Videos
        # --------------------------------------------------------------

        for index, filename in enumerate(
            video_files[: cls.MAX_VIDEOS]
        ):
            video_key = (
                f"ref_videos.ref_video_{index}"
            )

            audio_key = (
                f"ref_video_audios."
                f"ref_video_audio_{index}"
            )

            video_id = cls._add_vhs_video(
                workflow,
                filename,
            )

            inputs[video_key] = [
                video_id,
                0,
            ]

            # VHS_LoadVideo output 2 is its audio output.
            inputs[audio_key] = [
                video_id,
                2,
            ]

        # --------------------------------------------------------------
        # Standalone audio
        # --------------------------------------------------------------

        # Existing workflow usually has a stereo guard.
        # We leave the existing first audio path intact if present.
        existing_audio_slots = [
            key
            for key in inputs
            if key.startswith(
                "ref_audios.ref_audio_"
            )
        ]

        for index, filename in enumerate(
            audio_files[: cls.MAX_AUDIO]
        ):
            key = (
                f"ref_audios.ref_audio_{index}"
            )

            if key in inputs:
                # Existing workflow audio loader is already connected.
                # Its filename is patched separately in the UI workflow.
                continue

            node_id = cls._add_load_audio(
                workflow,
                filename,
            )

            inputs[key] = [
                node_id,
                0,
            ]

    @classmethod
    def _add_load_audio(
        cls,
        workflow: dict,
        filename: str,
    ) -> str:

        node_id = cls._next_id(
            workflow
        )

        workflow[node_id] = {
            "class_type": "LoadAudio",
            "inputs": {
                "audio_file": filename,
            },
            "_meta": {
                "title": (
                    "H3 standalone reference audio"
                ),
            },
        }

        return node_id

    # ------------------------------------------------------------------
    # COMMON API PATCH
    # ------------------------------------------------------------------

    @classmethod
    def _patch_api_common(
        cls,
        workflow: dict,
        prompt: str,
        width: int,
        height: int,
        frames: int,
        steps: int,
        seed: int,
    ) -> None:

        _, ref_node = cls._find_api_ref2va(
            workflow
        )

        cls._set_api_inputs_if_present(
            ref_node,
            {
                "prompt": prompt,
                "width": int(width),
                "height": int(height),
                "length": int(frames),
                "ref_image_size": "match",
            },
        )

        for _, node in cls._find_api_nodes(
            workflow,
            "BasicScheduler",
        ):
            cls._set_api_inputs_if_present(
                node,
                {
                    "scheduler": cls.DEFAULT_SCHEDULER,
                    "steps": int(steps),
                    "denoise": 1.0,
                },
            )

        for _, node in cls._find_api_nodes(
            workflow,
            "KSamplerSelect",
        ):
            cls._set_api_inputs_if_present(
                node,
                {
                    "sampler_name": cls.DEFAULT_SAMPLER,
                },
            )

        for _, node in cls._find_api_nodes(
            workflow,
            "RandomNoise",
        ):
            cls._set_api_inputs_if_present(
                node,
                {
                    "noise_seed": int(seed),
                },
            )

        for _, node in cls._find_api_nodes(
            workflow,
            "CreateVideo",
        ):
            cls._set_api_inputs_if_present(
                node,
                {
                    "frame_rate": cls.DEFAULT_FPS,
                },
            )

    @classmethod
    def _patch_api_models(
        cls,
        workflow: dict,
    ) -> None:

        for _, node in cls._find_api_nodes(
            workflow,
            "H3ModelLoaderAny",
        ):
            node.setdefault(
                "inputs",
                {}
            )[
                "model_name"
            ] = cls.REF2VA_MODEL

        for _, node in cls._find_api_nodes(
            workflow,
            "H3ClipLoaderAny",
        ):
            inputs = node.setdefault(
                "inputs",
                {}
            )

            inputs[
                "clip_name"
            ] = cls.TEXT_ENCODER

            inputs[
                "type"
            ] = "minimax"

        for _, node in cls._find_api_nodes(
            workflow,
            "VAELoader",
        ):
            title = (
                node.get(
                    "_meta",
                    {}
                ).get(
                    "title",
                    ""
                ).lower()
            )

            if "audio" in title:
                node.setdefault(
                    "inputs",
                    {}
                )[
                    "vae_name"
                ] = cls.AUDIO_VAE

            elif "video" in title:
                node.setdefault(
                    "inputs",
                    {}
                )[
                    "vae_name"
                ] = cls.VIDEO_VAE

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    @classmethod
    def _patch_save_prefix(
        cls,
        workflow: dict,
        prefix: str,
    ) -> None:

        for _, node in cls._find_api_nodes(
            workflow,
            "SaveVideo",
        ):
            inputs = node.setdefault(
                "inputs",
                {}
            )

            if "filename_prefix" in inputs:
                inputs[
                    "filename_prefix"
                ] = prefix

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_nodes(
        self,
        workflow: dict,
        required: set[str],
    ) -> None:

        present = {
            node.get(
                "class_type"
            )
            for node in workflow.values()
            if isinstance(
                node,
                dict,
            )
        }

        missing = sorted(
            required - present
        )

        if missing:
            raise RuntimeError(
                "Converted H3 workflow is missing "
                "required nodes:\n- "
                + "\n- ".join(missing)
            )

        if self.object_info:
            unknown = sorted(
                node_type
                for node_type in required
                if node_type not in self.object_info
            )

            if unknown:
                raise RuntimeError(
                    "Live ComfyUI /object_info does not "
                    "contain required H3 nodes:\n- "
                    + "\n- ".join(unknown)
                )

    # ------------------------------------------------------------------
    # PUBLIC BUILDERS
    # ------------------------------------------------------------------

    def build_native_ref2va(
        self,
        client,
        prompt: str,
        image_files: list[str],
        video_files: list[str],
        audio_files: list[str],
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        frames: int = DEFAULT_FRAMES,
        steps: int = DEFAULT_STEPS,
        seed: int = 0,
        output_prefix: str = "h3/ref2va",
    ) -> dict:

        ui = self._load_ui(
            "H3_HardMode_R2V.json"
        )

        self._patch_common_ui_values(
            ui
        )

        self._patch_duration_ui(
            ui,
            frames / self.DEFAULT_FPS,
        )

        self._patch_reference_images_ui(
            ui,
            image_files,
        )

        self._patch_standalone_audio_ui(
            ui,
            audio_files,
        )

        self._patch_prompt_ui(
            ui,
            prompt,
            chained=False,
        )

        api = client.convert_workflow(
            ui
        )

        api = copy.deepcopy(
            api
        )

        self._patch_api_models(
            api
        )

        self._patch_api_common(
            api,
            prompt=prompt,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            seed=seed,
        )

        self._patch_api_references(
            api,
            image_files=image_files,
            video_files=video_files,
            audio_files=audio_files,
        )

        self._patch_save_prefix(
            api,
            output_prefix,
        )

        self._validate_nodes(
            api,
            {
                "H3ModelLoaderAny",
                "H3ClipLoaderAny",
                "MiniMaxH3ReferenceToVideo",
                "H3FreeTextEncoder",
                "H3ConditionStrength",
                "VAEDecode",
                "VAEDecodeAudio",
                "CreateVideo",
                "SaveVideo",
            },
        )

        return api

    def build_hardmode_chained(
        self,
        client,
        shots: list[dict],
        image_files: list[str],
        video_files: list[str],
        audio_files: list[str],
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        frames: int = DEFAULT_FRAMES,
        steps: int = DEFAULT_STEPS,
        seed: int = 0,
        output_prefix: str = "h3/chained",
    ) -> dict:

        if len(shots) < 2:
            raise ValueError(
                "Hard Mode Chained requires at least "
                "two shots."
            )

        if len(shots) > 9:
            raise ValueError(
                "A single H3 chained production supports "
                "at most 9 planned shot prompts."
            )

        ui = self._load_ui(
            "H3_HardMode_Chained.json"
        )

        self._patch_common_ui_values(
            ui
        )

        self._patch_duration_ui(
            ui,
            frames / self.DEFAULT_FPS,
        )

        self._patch_reference_images_ui(
            ui,
            image_files,
        )

        self._patch_standalone_audio_ui(
            ui,
            audio_files,
        )

        prompts = []

        for index, shot in enumerate(
            shots
        ):
            prompt = (
                str(
                    shot.get(
                        "visual_prompt",
                        "",
                    )
                ).strip()
            )

            if not prompt:
                prompt = (
                    str(
                        shot.get(
                            "action",
                            "",
                        )
                    ).strip()
                )

            if index == 0:
                bindings = shot.get(
                    "reference_bindings",
                    [],
                )

                if bindings:
                    prompt = (
                        "\n".join(
                            str(item)
                            for item in bindings
                            if str(item).strip()
                        )
                        + "\n"
                        + prompt
                    )

            prompts.append(
                prompt
            )

        script = (
            "\n---\n".join(
                prompts
            )
        )

        self._patch_prompt_ui(
            ui,
            script,
            chained=True,
        )

        api = client.convert_workflow(
            ui
        )

        api = copy.deepcopy(
            api
        )

        self._patch_api_models(
            api
        )

        self._patch_api_references(
            api,
            image_files=image_files,
            video_files=video_files,
            audio_files=audio_files,
        )

        # First-stage reference-to-video settings.
        for _, node in self._find_api_nodes(
            api,
            "MiniMaxH3ReferenceToVideo",
        ):
            self._set_api_inputs_if_present(
                node,
                {
                    "width": int(width),
                    "height": int(height),
                    "length": int(frames),
                    "ref_image_size": "match",
                },
            )

        # Later chained shots.
        for _, node in self._find_api_nodes(
            api,
            "H3MultishotSampler",
        ):
            self._set_api_inputs_if_present(
                node,
                {
                    "width": int(width),
                    "height": int(height),
                    "frames_per_shot": int(frames),
                    "steps": int(steps),
                    "seed": int(seed),
                    "sampler_name": (
                        self.DEFAULT_SAMPLER
                    ),
                    "scheduler": (
                        self.DEFAULT_SCHEDULER
                    ),
                },
            )

        for _, node in self._find_api_nodes(
            api,
            "BasicScheduler",
        ):
            self._set_api_inputs_if_present(
                node,
                {
                    "scheduler": self.DEFAULT_SCHEDULER,
                    "steps": int(steps),
                },
            )

        for _, node in self._find_api_nodes(
            api,
            "RandomNoise",
        ):
            self._set_api_inputs_if_present(
                node,
                {
                    "noise_seed": int(seed),
                },
            )

        for _, node in self._find_api_nodes(
            api,
            "CreateVideo",
        ):
            self._set_api_inputs_if_present(
                node,
                {
                    "frame_rate": self.DEFAULT_FPS,
                },
            )

        self._patch_save_prefix(
            api,
            output_prefix,
        )

        self._validate_nodes(
            api,
            {
                "H3ModelLoaderAny",
                "H3ClipLoaderAny",
                "MiniMaxH3ReferenceToVideo",
                "H3MultishotSampler",
                "H3LastFrame",
                "H3ConcatAV",
                "VAEDecode",
                "VAEDecodeAudio",
                "SaveVideo",
            },
        )

        return api
