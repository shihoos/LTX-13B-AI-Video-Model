from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

from PIL import Image


class ShotExecutor:

    """
    Executes one shot through the complete pipeline:

        Base workflow
            ↓
        raw MP4
            ↓
        IC-LoRA/detail/upscale workflow
            ↓
        detailed high-resolution MP4

    Character references:

        0 references
            → existing no-reference behavior

        1 reference
            → use the original image directly

        2+ references
            → create one dynamic composite reference image
              containing every reference image

    Character identity masks:

        0 references
            → no identity mask

        1 reference
            → use the original identity mask directly

        2+ references
            → create one dynamic composite identity mask
              matching the reference-image layout

    There is no hardcoded character/reference-count limit.
    """

    def __init__(
        self,
        comfy_client,
        workflow_adapter,
        detailer_workflow_adapter,
        checkpoint_manager,
        project_root: Path,
        comfy_input_dir: Path,
    ):

        self.client = comfy_client

        self.workflow_adapter = workflow_adapter

        self.detailer_workflow_adapter = (
            detailer_workflow_adapter
        )

        self.checkpoints = checkpoint_manager

        self.project_root = Path(
            project_root
        )

        self.comfy_input_dir = Path(
            comfy_input_dir
        )

        self.comfy_input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.api_workflow = (
            self.workflow_adapter
            .to_api_workflow()
        )

        self.api_detailer_workflow = (
            self.detailer_workflow_adapter
            .to_api_workflow()
        )

    # ========================================================
    # REFERENCE IMAGE
    # ========================================================

    @staticmethod
    def _validate_reference_paths(
        reference_paths: list[str],
        shot_id: str,
    ) -> list[Path]:

        paths = []

        for reference_path in (
            reference_paths
        ):

            source = Path(
                reference_path
            )

            if not source.is_file():

                raise FileNotFoundError(
                    f"[{shot_id}] "
                    "Reference image does not exist:\n"
                    f"{source}"
                )

            if (
                source.stat().st_size
                <= 0
            ):

                raise RuntimeError(
                    f"[{shot_id}] "
                    "Reference image is empty:\n"
                    f"{source}"
                )

            paths.append(
                source
            )

        return paths

    def _copy_reference(
        self,
        reference_path: str,
        shot_id: str,
    ) -> str:

        source = (
            self._validate_reference_paths(
                [reference_path],
                shot_id,
            )[0]
        )

        destination = (
            self.comfy_input_dir
            / (
                f"{shot_id}_"
                f"{source.name}"
            )
        )

        shutil.copy2(
            source,
            destination,
        )

        return destination.name

    # ========================================================
    # IDENTITY MASK
    # ========================================================

    @staticmethod
    def _validate_mask_paths(
        mask_paths: list[str],
        shot_id: str,
    ) -> list[Path]:

        paths = []

        for mask_path in (
            mask_paths
        ):

            source = Path(
                mask_path
            )

            if not source.is_file():

                raise FileNotFoundError(
                    f"[{shot_id}] "
                    "Reference mask does not exist:\n"
                    f"{source}"
                )

            if (
                source.stat().st_size
                <= 0
            ):

                raise RuntimeError(
                    f"[{shot_id}] "
                    "Reference mask is empty:\n"
                    f"{source}"
                )

            paths.append(
                source
            )

        return paths

    def _copy_reference_mask(
        self,
        mask_path: str,
        shot_id: str,
    ) -> str:

        source = (
            self._validate_mask_paths(
                [mask_path],
                shot_id,
            )[0]
        )

        destination = (
            self.comfy_input_dir
            / (
                f"{shot_id}_"
                f"{source.name}"
            )
        )

        shutil.copy2(
            source,
            destination,
        )

        return destination.name

    # ========================================================
    # MULTI-REFERENCE COMPOSITION
    # ========================================================

    @staticmethod
    def _grid_dimensions(
        count: int,
    ) -> tuple[int, int]:

        if count <= 0:

            raise ValueError(
                "Reference count must be greater than zero."
            )

        columns = max(
            1,
            math.ceil(
                math.sqrt(
                    count * 16 / 9
                )
            ),
        )

        rows = math.ceil(
            count / columns
        )

        return (
            columns,
            rows,
        )

    def _compose_reference_images(
        self,
        reference_paths: list[str],
        shot_id: str,
    ) -> Path:

        paths = (
            self._validate_reference_paths(
                reference_paths,
                shot_id,
            )
        )

        destination = (
            self.comfy_input_dir
            / (
                f"{shot_id}_"
                "character_references.png"
            )
        )

        if len(paths) == 1:

            shutil.copy2(
                paths[0],
                destination,
            )

            return destination

        try:

            images = [
                Image.open(
                    path
                ).convert(
                    "RGB"
                )
                for path
                in paths
            ]

            try:

                max_width = max(
                    image.width
                    for image
                    in images
                )

                max_height = max(
                    image.height
                    for image
                    in images
                )

                tile_width = max_width
                tile_height = max_height

                columns, rows = (
                    self._grid_dimensions(
                        len(images)
                    )
                )

                canvas_width = (
                    columns
                    * tile_width
                )

                canvas_height = (
                    rows
                    * tile_height
                )

                canvas = Image.new(
                    "RGB",
                    (
                        canvas_width,
                        canvas_height,
                    ),
                    "black",
                )

                for index, image in enumerate(
                    images
                ):

                    image.thumbnail(
                        (
                            tile_width,
                            tile_height,
                        ),
                        Image.Resampling.LANCZOS,
                    )

                    x = (
                        (
                            index
                            % columns
                        )
                        * tile_width
                        + (
                            tile_width
                            - image.width
                        )
                        // 2
                    )

                    y = (
                        (
                            index
                            // columns
                        )
                        * tile_height
                        + (
                            tile_height
                            - image.height
                        )
                        // 2
                    )

                    canvas.paste(
                        image,
                        (
                            x,
                            y,
                        ),
                    )

                canvas.save(
                    destination,
                    format="PNG",
                )

            finally:

                for image in images:

                    image.close()

        except Exception as error:

            raise RuntimeError(
                f"[{shot_id}] "
                "Could not create the "
                "multi-character reference "
                "image:\n"
                f"{error}"
            ) from error

        if (
            not destination.is_file()
            or
            destination.stat().st_size <= 0
        ):

            raise RuntimeError(
                f"[{shot_id}] "
                "Multi-character reference "
                "image was not created correctly."
            )

        return destination

    def _compose_reference_masks(
        self,
        mask_paths: list[str],
        reference_paths: list[str],
        shot_id: str,
    ) -> Path:

        masks = (
            self._validate_mask_paths(
                mask_paths,
                shot_id,
            )
        )

        references = (
            self._validate_reference_paths(
                reference_paths,
                shot_id,
            )
        )

        if len(masks) != len(
            references
        ):

            raise RuntimeError(
                f"[{shot_id}] "
                "Reference image/mask count "
                "does not match."
            )

        destination = (
            self.comfy_input_dir
            / (
                f"{shot_id}_"
                "character_reference_mask.png"
            )
        )

        if len(masks) == 1:

            shutil.copy2(
                masks[0],
                destination,
            )

            return destination

        try:

            images = [
                Image.open(
                    path
                ).convert(
                    "L"
                )
                for path
                in masks
            ]

            try:

                max_width = max(
                    image.width
                    for image
                    in images
                )

                max_height = max(
                    image.height
                    for image
                    in images
                )

                tile_width = max_width
                tile_height = max_height

                columns, rows = (
                    self._grid_dimensions(
                        len(images)
                    )
                )

                canvas_width = (
                    columns
                    * tile_width
                )

                canvas_height = (
                    rows
                    * tile_height
                )

                canvas = Image.new(
                    "L",
                    (
                        canvas_width,
                        canvas_height,
                    ),
                    0,
                )

                for index, image in enumerate(
                    images
                ):

                    image.thumbnail(
                        (
                            tile_width,
                            tile_height,
                        ),
                        Image.Resampling.LANCZOS,
                    )

                    x = (
                        (
                            index
                            % columns
                        )
                        * tile_width
                        + (
                            tile_width
                            - image.width
                        )
                        // 2
                    )

                    y = (
                        (
                            index
                            // columns
                        )
                        * tile_height
                        + (
                            tile_height
                            - image.height
                        )
                        // 2
                    )

                    canvas.paste(
                        image,
                        (
                            x,
                            y,
                        ),
                    )

                canvas.save(
                    destination,
                    format="PNG",
                )

            finally:

                for image in images:

                    image.close()

        except Exception as error:

            raise RuntimeError(
                f"[{shot_id}] "
                "Could not create the "
                "multi-character reference "
                "mask:\n"
                f"{error}"
            ) from error

        if (
            not destination.is_file()
            or
            destination.stat().st_size <= 0
        ):

            raise RuntimeError(
                f"[{shot_id}] "
                "Multi-character reference "
                "mask was not created correctly."
            )

        return destination

    def _prepare_shot_reference(
        self,
        shot,
    ) -> tuple[
        str | None,
        str | None,
    ]:

        references = [
            str(path)
            for path
            in (
                shot.reference_images
                or []
            )
            if str(path).strip()
        ]

        masks = [
            str(path)
            for path
            in (
                getattr(
                    shot,
                    "reference_masks",
                    [],
                )
                or []
            )
            if str(path).strip()
        ]

        if not references:

            return (
                None,
                None,
            )

        if len(references) != len(
            masks
        ):

            raise RuntimeError(
                f"[{shot.shot_id}] "
                "Every character reference "
                "must have a corresponding "
                "identity mask."
            )

        if len(references) == 1:

            reference_name = (
                self._copy_reference(
                    references[0],
                    shot.shot_id,
                )
            )

            mask_name = (
                self._copy_reference_mask(
                    masks[0],
                    shot.shot_id,
                )
            )

            return (
                reference_name,
                mask_name,
            )

        composite_image = (
            self._compose_reference_images(
                references,
                shot.shot_id,
            )
        )

        composite_mask = (
            self._compose_reference_masks(
                masks,
                references,
                shot.shot_id,
            )
        )

        return (
            composite_image.name,
            composite_mask.name,
        )

    # ========================================================
    # VIDEO INPUT FOR DETAILER
    # ========================================================

    def _copy_raw_to_comfy_input(
        self,
        raw_path: Path,
        shot_id: str,
    ) -> str:

        if not raw_path.exists():

            raise FileNotFoundError(
                f"Raw video does not exist: "
                f"{raw_path}"
            )

        destination = (
            self.comfy_input_dir
            / (
                f"{shot_id}_raw.mp4"
            )
        )

        if not destination.exists():

            try:

                os.link(
                    raw_path,
                    destination,
                )

            except OSError:

                shutil.copy2(
                    raw_path,
                    destination,
                )

        return destination.name

    # ========================================================
    # RAW GENERATION
    # ========================================================

    def execute_raw(
        self,
        shot,
        gpu_id: int,
        output_dir: Path,
    ) -> Path:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoints.mark_generating(
            shot.shot_id,
            gpu_id,
        )

        workflow = (
            self.workflow_adapter
            .apply_shot(
                self.api_workflow,
                shot,
            )
        )

        self.workflow_adapter.set_filename_prefix(
            workflow,
            f"ltx_raw/{shot.shot_id}",
        )

        if shot.seed is not None:

            self.workflow_adapter.set_seed(
                workflow,
                int(shot.seed),
            )

        (
            reference_name,
            reference_mask_name,
        ) = (
            self._prepare_shot_reference(
                shot
            )
        )

        if reference_name:

            self.workflow_adapter.set_identity_reference_image(
                workflow,
                reference_name,
            )

            self.workflow_adapter.set_identity_mask_image(
                workflow,
                reference_mask_name,
            )

            if len(
                shot.reference_images
                or []
            ) > 1:

                print(
                    f"[{shot.shot_id}] "
                    f"Using "
                    f"{len(shot.reference_images)} "
                    "character references "
                    "via dynamic composite."
                )

        print(
            f"[{shot.shot_id}] "
            f"GPU {gpu_id}: starting base generation"
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

        video_outputs = (
            self.client.find_video_outputs(
                history
            )
        )

        if not video_outputs:

            raise RuntimeError(
                f"[{shot.shot_id}] "
                "Base workflow completed but "
                "no video output was found."
            )

        video = (
            video_outputs[0]
        )

        destination = (
            output_dir
            / "shots"
            / shot.scene_id
            / "raw"
            / (
                f"{shot.shot_id}.mp4"
            )
        )

        self.client.download_file(
            filename=video[
                "filename"
            ],
            subfolder=video[
                "subfolder"
            ],
            file_type=video[
                "type"
            ],
            destination=destination,
        )

        if (
            not destination.exists()
            or not destination.is_file()
            or destination.stat().st_size <= 0
        ):

            raise RuntimeError(
                f"[{shot.shot_id}] "
                "Raw output download failed "
                "or produced an empty file."
            )

        self.checkpoints.mark_raw_complete(
            shot.shot_id,
            str(destination),
        )

        print(
            f"[{shot.shot_id}] "
            f"Raw generation complete:"
            f"\n    {destination}"
        )

        return destination

    # ========================================================
    # IC-LORA + DETAIL + UPSCALE
    # ========================================================

    def execute_detailer(
        self,
        shot,
        raw_path: Path,
        gpu_id: int,
        output_dir: Path,
    ) -> Path:

        if not raw_path.exists():

            raise FileNotFoundError(
                f"Raw clip required for detailer "
                f"does not exist: {raw_path}"
            )

        workflow = (
            self.detailer_workflow_adapter
            .apply_shot(
                self.api_detailer_workflow,
                shot,
            )
        )

        if shot.seed is not None:

            self.detailer_workflow_adapter.set_seed(
                workflow,
                int(shot.seed),
            )

        comfy_video_name = (
            self._copy_raw_to_comfy_input(
                raw_path,
                shot.shot_id,
            )
        )

        self.detailer_workflow_adapter.set_input_video(
            workflow,
            comfy_video_name,
        )

        self.detailer_workflow_adapter.set_filename_prefix(
            workflow,
            f"ltx_master/{shot.shot_id}",
        )

        print(
            f"[{shot.shot_id}] "
            f"GPU {gpu_id}: starting "
            "IC-LoRA detail + spatial upscale"
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

        video_outputs = (
            self.client.find_video_outputs(
                history
            )
        )

        if not video_outputs:

            raise RuntimeError(
                f"[{shot.shot_id}] "
                "Detailer workflow completed "
                "but no video output was found."
            )

        video = (
            video_outputs[0]
        )

        destination = (
            output_dir
            / "shots"
            / shot.scene_id
            / "upscaled"
            / (
                f"{shot.shot_id}.mp4"
            )
        )

        self.client.download_file(
            filename=video[
                "filename"
            ],
            subfolder=video[
                "subfolder"
            ],
            file_type=video[
                "type"
            ],
            destination=destination,
        )

        if (
            not destination.exists()
            or not destination.is_file()
            or destination.stat().st_size <= 0
        ):

            raise RuntimeError(
                f"[{shot.shot_id}] "
                "Detailer/upscale output "
                "download failed or produced "
                "an empty file."
            )

        self.checkpoints.mark_upscaled_complete(
            shot.shot_id,
            str(destination),
        )

        print(
            f"[{shot.shot_id}] "
            f"IC-LoRA + upscale complete:"
            f"\n    {destination}"
        )

        return destination

    # ========================================================
    # COMPLETE SHOT
    # ========================================================

    def execute_shot(
        self,
        shot,
        gpu_id: int,
        output_dir: Path,
    ):

        try:

            state = (
                self.checkpoints.get_shot(
                    shot.shot_id
                )
            )

            if (
                state
                and state.get(
                    "upscaled_path"
                )
                and Path(
                    state["upscaled_path"]
                ).exists()
            ):

                print(
                    f"[{shot.shot_id}] "
                    "already has completed "
                    "upscaled output; skipping."
                )

                if state.get(
                    "status"
                ) != "completed":

                    self.checkpoints.mark_complete(
                        shot.shot_id
                    )

                return Path(
                    state["upscaled_path"]
                )

            raw_path = None

            if (
                state
                and state.get(
                    "raw_path"
                )
                and Path(
                    state["raw_path"]
                ).exists()
            ):

                raw_path = Path(
                    state["raw_path"]
                )

                print(
                    f"[{shot.shot_id}] "
                    "Existing raw artifact found."
                )

                print(
                    f"[{shot.shot_id}] "
                    "Skipping base generation."
                )

            else:

                raw_path = (
                    self.execute_raw(
                        shot,
                        gpu_id,
                        output_dir,
                    )
                )

            state = (
                self.checkpoints.get_shot(
                    shot.shot_id
                )
            )

            if (
                state
                and state.get(
                    "upscaled_path"
                )
                and Path(
                    state["upscaled_path"]
                ).exists()
            ):

                final_path = Path(
                    state["upscaled_path"]
                )

                self.checkpoints.mark_complete(
                    shot.shot_id
                )

                return final_path

            final_path = (
                self.execute_detailer(
                    shot,
                    raw_path,
                    gpu_id,
                    output_dir,
                )
            )

            self.checkpoints.mark_complete(
                shot.shot_id
            )

            print(
                f"[{shot.shot_id}] "
                "✅ COMPLETE"
            )

            return final_path

        except Exception as error:

            self.checkpoints.mark_failed(
                shot.shot_id,
                str(error),
            )

            print(
                f"[{shot.shot_id}] "
                f"❌ FAILED: {error}"
            )

            raise
