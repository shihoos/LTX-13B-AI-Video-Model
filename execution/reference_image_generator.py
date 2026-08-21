from __future__ import annotations

import hashlib
import re

from pathlib import Path

from execution.comfy_client import (
    ComfyClient,
)

from planner.config import (
    GENERATED_CHARACTERS_DIR,
    REFERENCE_IMAGE_CFG,
    REFERENCE_IMAGE_CHECKPOINT,
    REFERENCE_IMAGE_HEIGHT,
    REFERENCE_IMAGE_HOST,
    REFERENCE_IMAGE_PORT,
    REFERENCE_IMAGE_STEPS,
    REFERENCE_IMAGE_WIDTH,
)


class ReferenceImageGenerator:

    """
    Generates one persistent reference image for a character.

    Existing user-provided references are handled before this
    class is called.

    Generated files are stored under:

        data/characters/generated/
    """

    def __init__(
        self,
        client=None,
    ):

        if client is None:

            client = ComfyClient(
                base_url=(
                    "http://"
                    f"{REFERENCE_IMAGE_HOST}:"
                    f"{REFERENCE_IMAGE_PORT}"
                )
            )

        self.client = client

        GENERATED_CHARACTERS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:

        normalized = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            name.strip().lower(),
        )

        normalized = normalized.strip(
            "_"
        )

        if not normalized:

            raise ValueError(
                "Character name cannot be empty."
            )

        return normalized

    def output_path(
        self,
        character_name: str,
    ) -> Path:

        return (
            GENERATED_CHARACTERS_DIR
            / (
                f"{self._normalize_name(character_name)}"
                ".png"
            )
        )

    @staticmethod
    def _stable_seed(
        character_name: str,
    ) -> int:

        digest = hashlib.sha256(
            character_name
            .strip()
            .lower()
            .encode(
                "utf-8"
            )
        ).digest()

        return int.from_bytes(
            digest[:4],
            "big",
        )

    def _available_checkpoint(
        self,
    ) -> str:

        object_info = (
            self.client
            .get_object_info()
        )

        loader = object_info.get(
            "CheckpointLoaderSimple"
        )

        if not isinstance(
            loader,
            dict,
        ):

            raise RuntimeError(
                "ComfyUI does not provide "
                "CheckpointLoaderSimple. "
                "A still-image checkpoint is "
                "required for automatic character "
                "reference generation."
            )

        inputs = loader.get(
            "input",
            {},
        )

        required = inputs.get(
            "required",
            {},
        )

        checkpoint_spec = required.get(
            "ckpt_name"
        )

        if not (
            isinstance(
                checkpoint_spec,
                list,
            )
            and checkpoint_spec
        ):

            raise RuntimeError(
                "No CheckpointLoaderSimple "
                "checkpoint choices were exposed "
                "by ComfyUI."
            )

        choices = checkpoint_spec[0]

        if not isinstance(
            choices,
            list,
        ) or not choices:

            raise RuntimeError(
                "No still-image checkpoint is "
                "available in ComfyUI."
            )

        if REFERENCE_IMAGE_CHECKPOINT:

            if (
                REFERENCE_IMAGE_CHECKPOINT
                not in choices
            ):

                raise RuntimeError(
                    "Configured reference-image "
                    "checkpoint was not found:\n"
                    f"{REFERENCE_IMAGE_CHECKPOINT}"
                )

            return (
                REFERENCE_IMAGE_CHECKPOINT
            )

        for checkpoint in choices:

            name = str(
                checkpoint
            ).lower()

            if (
                "sdxl"
                in name
                or "sd1.5"
                in name
                or "sd15"
                in name
            ):

                return str(
                    checkpoint
                )

        return str(
            choices[0]
        )

    def _build_workflow(
        self,
        character_name: str,
        description: str,
        personality: str,
        appearance: dict,
        clothing: dict,
        distinctive_features: list,
        character_state: dict,
    ) -> dict:

        checkpoint = (
            self._available_checkpoint()
        )

        appearance_text = ", ".join(
            f"{key}: {value}"
            for key, value
            in appearance.items()
        )

        clothing_text = ", ".join(
            f"{key}: {value}"
            for key, value
            in clothing.items()
        )

        features_text = ", ".join(
            str(item)
            for item
            in distinctive_features
        )

        state_text = ", ".join(
            f"{key}: {value}"
            for key, value
            in character_state.items()
        )

        positive = (
            "cinematic character reference, "
            "single person, full body, "
            "clear face, realistic human proportions, "
            "stable visual identity, "
            f"character name: {character_name}, "
            f"description: {description}, "
            f"personality: {personality}, "
            f"appearance: {appearance_text}, "
            f"clothing: {clothing_text}, "
            f"distinctive features: {features_text}, "
            f"current state: {state_text}, "
            "neutral uncluttered background, "
            "clean silhouette, "
            "realistic materials, "
            "high detail, "
            "natural lighting"
        )

        negative = (
            "multiple people, "
            "duplicate person, "
            "extra limbs, "
            "extra fingers, "
            "deformed hands, "
            "distorted face, "
            "cropped head, "
            "cropped feet, "
            "text, watermark, logo"
        )

        seed = (
            self._stable_seed(
                character_name
            )
        )

        prefix = (
            "character_references/"
            + self._normalize_name(
                character_name
            )
        )

        return {
            "1": {
                "class_type":
                    "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name":
                        checkpoint,
                },
            },

            "2": {
                "class_type":
                    "CLIPTextEncode",
                "inputs": {
                    "text":
                        positive,
                    "clip": [
                        "1",
                        1,
                    ],
                },
            },

            "3": {
                "class_type":
                    "CLIPTextEncode",
                "inputs": {
                    "text":
                        negative,
                    "clip": [
                        "1",
                        1,
                    ],
                },
            },

            "4": {
                "class_type":
                    "EmptyLatentImage",
                "inputs": {
                    "width":
                        REFERENCE_IMAGE_WIDTH,
                    "height":
                        REFERENCE_IMAGE_HEIGHT,
                    "batch_size":
                        1,
                },
            },

            "5": {
                "class_type":
                    "KSampler",
                "inputs": {
                    "seed":
                        seed,
                    "steps":
                        REFERENCE_IMAGE_STEPS,
                    "cfg":
                        REFERENCE_IMAGE_CFG,
                    "sampler_name":
                        "euler",
                    "scheduler":
                        "normal",
                    "denoise":
                        1.0,
                    "model": [
                        "1",
                        0,
                    ],
                    "positive": [
                        "2",
                        0,
                    ],
                    "negative": [
                        "3",
                        0,
                    ],
                    "latent_image": [
                        "4",
                        0,
                    ],
                },
            },

            "6": {
                "class_type":
                    "VAEDecode",
                "inputs": {
                    "samples": [
                        "5",
                        0,
                    ],
                    "vae": [
                        "1",
                        2,
                    ],
                },
            },

            "7": {
                "class_type":
                    "SaveImage",
                "inputs": {
                    "filename_prefix":
                        prefix,
                    "images": [
                        "6",
                        0,
                    ],
                },
            },
        }

    def generate(
        self,
        character_name: str,
        description: str,
        personality: str,
        appearance: dict,
        clothing: dict,
        distinctive_features: list,
        character_state: dict,
    ) -> Path:

        destination = self.output_path(
            character_name
        )

        if (
            destination.is_file()
            and
            destination.stat().st_size > 0
        ):

            return destination

        if not self.client.health_check():

            raise RuntimeError(
                "ComfyUI is unavailable for "
                "character reference generation:\n"
                f"{self.client.base_url}"
            )

        workflow = self._build_workflow(
            character_name=character_name,
            description=description,
            personality=personality,
            appearance=appearance,
            clothing=clothing,
            distinctive_features=(
                distinctive_features
            ),
            character_state=(
                character_state
            ),
        )

        prompt_id = (
            self.client.queue_prompt(
                workflow
            )
        )

        history = (
            self.client.wait_for_prompt(
                prompt_id
            )
        )

        image_outputs = (
            self.client.find_image_outputs(
                history
            )
        )

        if not image_outputs:

            raise RuntimeError(
                "Reference-image generation "
                "completed without an image output "
                f"for '{character_name}'."
            )

        image = (
            image_outputs[0]
        )

        self.client.download_file(
            filename=image[
                "filename"
            ],
            subfolder=image[
                "subfolder"
            ],
            file_type=image[
                "type"
            ],
            destination=destination,
        )

        if (
            not destination.is_file()
            or
            destination.stat().st_size <= 0
        ):

            raise RuntimeError(
                "Generated character reference "
                "was not saved correctly:\n"
                f"{destination}"
            )

        return destination
