from __future__ import annotations

import copy
import json
from pathlib import Path

from planner.config import (
    H3_AUDIO_VAE,
    H3_FPS,
    H3_REF2VA_MODEL,
    H3_REF_IMAGE_SIZE,
    H3_TEXT_ENCODER,
    H3_VIDEO_VAE,
)


class H3WorkflowBuilder:

    MAX_IMAGES = 9
    MAX_VIDEOS = 3
    MAX_AUDIO = 3
    MAX_TOTAL_FILES = 12

    def __init__(
        self,
        project_root: Path,
    ):

        self.project_root = Path(
            project_root
        )

        self.workflow_path = (
            self.project_root
            / "workflows"
            / "MiniMax-H3"
            / "base"
            / "H3_Ref2VA_Memory_API.json"
        )

    def _load_template(self):

        if not self.workflow_path.is_file():
            raise FileNotFoundError(
                f"H3 API workflow missing: "
                f"{self.workflow_path}"
            )

        return json.loads(
            self.workflow_path.read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def _find(
        workflow,
        class_type,
    ):

        for node_id, node in workflow.items():

            if (
                isinstance(node, dict)
                and node.get(
                    "class_type"
                ) == class_type
            ):
                return (
                    str(node_id),
                    node,
                )

        raise RuntimeError(
            f"Required node missing: "
            f"{class_type}"
        )

    @staticmethod
    def _find_all(
        workflow,
        class_type,
    ):

        return [
            (
                str(node_id),
                node,
            )
            for node_id, node
            in workflow.items()
            if (
                isinstance(node, dict)
                and node.get(
                    "class_type"
                ) == class_type
            )
        ]

    @staticmethod
    def _next_id(workflow):

        numeric = []

        for node_id in workflow:

            try:
                numeric.append(
                    int(node_id)
                )
            except Exception:
                pass

        return str(
            max(numeric or [0])
            + 1
        )

    @classmethod
    def _add_node(
        cls,
        workflow,
        class_type,
        inputs,
        title,
    ):

        node_id = cls._next_id(
            workflow
        )

        workflow[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
            "_meta": {
                "title": title
            },
        }

        return node_id

    @classmethod
    def _add_image(
        cls,
        workflow,
        filename,
    ):

        return cls._add_node(
            workflow,
            "LoadImage",
            {
                "image": filename,
            },
            "H3 reference image",
        )

    @classmethod
    def _add_video(
        cls,
        workflow,
        filename,
    ):

        return cls._add_node(
            workflow,
            "VHS_LoadVideo",
            {
                "video": filename,

                "force_rate": 24,

                "force_size": "Disabled",

                "custom_width": 0,
                "custom_height": 0,

                "frame_load_cap": 0,

                "skip_first_frames": 0,

                "select_every_nth": 1,
            },
            "H3 reference video",
        )

    @classmethod
    def _add_audio(
        cls,
        workflow,
        filename,
    ):

        return cls._add_node(
            workflow,
            "LoadAudio",
            {
                "audio_file": filename,
            },
            "H3 reference audio",
        )

    @classmethod
    def _set_input(
        cls,
        node,
        key,
        value,
    ):

        node.setdefault(
            "inputs",
            {}
        )[key] = value

    @classmethod
    def _patch_model_nodes(
        cls,
        workflow,
    ):

        for _, node in cls._find_all(
            workflow,
            "H3ModelLoaderAny",
        ):

            cls._set_input(
                node,
                "model_name",
                H3_REF2VA_MODEL,
            )

        for _, node in cls._find_all(
            workflow,
            "H3ClipLoaderAny",
        ):

            cls._set_input(
                node,
                "model_name",
                H3_TEXT_ENCODER,
            )

            cls._set_input(
                node,
                "clip_name",
                H3_TEXT_ENCODER,
            )

            cls._set_input(
                node,
                "type",
                "minimax",
            )

        for _, node in cls._find_all(
            workflow,
            "VAELoader",
        ):

            title = str(
                node.get(
                    "_meta",
                    {}
                ).get(
                    "title",
                    ""
                )
            ).lower()

            if "audio" in title:
                cls._set_input(
                    node,
                    "vae_name",
                    H3_AUDIO_VAE,
                )

            elif "video" in title:
                cls._set_input(
                    node,
                    "vae_name",
                    H3_VIDEO_VAE,
                )

    @classmethod
    def _patch_sampling(
        cls,
        workflow,
        steps,
        seed,
    ):

        for _, node in cls._find_all(
            workflow,
            "BasicScheduler",
        ):

            cls._set_input(
                node,
                "steps",
                int(steps),
            )

            cls._set_input(
                node,
                "denoise",
                1.0,
            )

        for _, node in cls._find_all(
            workflow,
            "RandomNoise",
        ):

            cls._set_input(
                node,
                "noise_seed",
                int(seed),
            )

    @classmethod
    def _patch_video(
        cls,
        workflow,
        width,
        height,
        frames,
        prompt,
    ):

        _, ref = cls._find(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )

        cls._set_input(
            ref,
            "width",
            int(width),
        )

        cls._set_input(
            ref,
            "height",
            int(height),
        )

        cls._set_input(
            ref,
            "length",
            int(frames),
        )

        cls._set_input(
            ref,
            "prompt",
            prompt,
        )

        cls._set_input(
            ref,
            "ref_image_size",
            H3_REF_IMAGE_SIZE,
        )

        for _, node in cls._find_all(
            workflow,
            "CreateVideo",
        ):

            cls._set_input(
                node,
                "frame_rate",
                H3_FPS,
            )

    @classmethod
    def _patch_references(
        cls,
        workflow,
        image_files,
        video_files,
        audio_files,
        video_audio_files=None,
    ):

        video_audio_files = (
            video_audio_files or []
        )

        total = (
            len(image_files)
            + len(video_files)
            + len(video_audio_files)
            + len(audio_files)
        )

        if total > cls.MAX_TOTAL_FILES:
            raise ValueError(
                "H3 Ref2VA allows at most "
                f"{cls.MAX_TOTAL_FILES} "
                "reference files total."
            )

        if len(image_files) > cls.MAX_IMAGES:
            raise ValueError(
                "Too many reference images."
            )

        if len(video_files) > cls.MAX_VIDEOS:
            raise ValueError(
                "Too many reference videos."
            )

        if len(audio_files) > cls.MAX_AUDIO:
            raise ValueError(
                "Too many standalone audio files."
            )

        _, ref = cls._find(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )

        # Remove old dynamic refs.
        inputs = ref.setdefault(
            "inputs",
            {}
        )

        for key in list(
            inputs.keys()
        ):

            if (
                key.startswith(
                    "ref_images."
                )
                or key.startswith(
                    "ref_videos."
                )
                or key.startswith(
                    "ref_video_audios."
                )
                or key.startswith(
                    "ref_audios."
                )
            ):
                del inputs[key]

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        for index, filename in enumerate(
            image_files
        ):

            node_id = cls._add_image(
                workflow,
                filename,
            )

            inputs[
                f"ref_images.ref_image_{index}"
            ] = [
                node_id,
                0,
            ]

        # ----------------------------------------------------
        # Videos
        # ----------------------------------------------------

        for index, filename in enumerate(
            video_files
        ):

            node_id = cls._add_video(
                workflow,
                filename,
            )

            inputs[
                f"ref_videos.ref_video_{index}"
            ] = [
                node_id,
                0,
            ]

            if index < len(
                video_audio_files
            ):

                audio_node = cls._add_audio(
                    workflow,
                    video_audio_files[index],
                )

                inputs[
                    "ref_video_audios."
                    f"ref_video_audio_{index}"
                ] = [
                    audio_node,
                    0,
                ]

        # ----------------------------------------------------
        # Standalone audio
        # ----------------------------------------------------

        for index, filename in enumerate(
            audio_files
        ):

            audio_node = cls._add_audio(
                workflow,
                filename,
            )

            inputs[
                f"ref_audios.ref_audio_{index}"
            ] = [
                audio_node,
                0,
            ]

    def build_ref2va(
        self,
        prompt,
        image_files,
        video_files,
        audio_files,
        video_audio_files=None,
        width=1344,
        height=768,
        frames=124,
        steps=14,
        seed=0,
        output_prefix="h3/ref2va",
    ):

        workflow = copy.deepcopy(
            self._load_template()
        )

        self._patch_model_nodes(
            workflow
        )

        self._patch_sampling(
            workflow,
            steps=steps,
            seed=seed,
        )

        self._patch_video(
            workflow,
            width=width,
            height=height,
            frames=frames,
            prompt=prompt,
        )

        self._patch_references(
            workflow,
            image_files=image_files,
            video_files=video_files,
            audio_files=audio_files,
            video_audio_files=(
                video_audio_files
            ),
        )

        for _, node in self._find_all(
            workflow,
            "SaveVideo",
        ):

            self._set_input(
                node,
                "filename_prefix",
                output_prefix,
            )

        return workflow
