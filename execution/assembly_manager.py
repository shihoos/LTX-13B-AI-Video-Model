import subprocess

from pathlib import Path


class AssemblyManager:

    def __init__(
        self,
        output_dir: Path,
    ):

        self.output_dir = (
            Path(output_dir)
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def check_ffmpeg(self):

        result = subprocess.run(
            [
                "ffmpeg",
                "-version",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "FFmpeg is not installed."
            )

    def create_concat_file(
        self,
        shot_paths: list,
        path: Path,
    ):

        lines = []

        for shot_path in shot_paths:

            absolute = (
                Path(
                    shot_path
                )
                .resolve()
            )

            escaped = str(
                absolute
            ).replace(
                "'",
                "'\\''",
            )

            lines.append(
                f"file '{escaped}'"
            )

        path.write_text(
            "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )

    def assemble(
        self,
        shot_paths: list,
        final_name: str = (
            "final_video.mp4"
        ),
    ) -> Path:

        self.check_ffmpeg()

        if not shot_paths:

            raise ValueError(
                "No completed shots "
                "were supplied."
            )

        output_path = (
            self.output_dir
            / final_name
        )

        concat_path = (
            self.output_dir
            / "concat.txt"
        )

        self.create_concat_file(
            shot_paths,
            concat_path,
        )

        temp_path = (
            self.output_dir
            / (
                ".final_video_tmp.mp4"
            )
        )

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),

            # Normalize to 24 FPS.
            "-vf",
            (
                "fps=24,"
                "scale=1280:720:"
                "force_original_aspect_ratio=increase,"
                "crop=1280:720"
            ),

            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "17",

            "-pix_fmt",
            "yuv420p",

            "-movflags",
            "+faststart",

            "-an",

            str(temp_path),
        ]

        subprocess.run(
            command,
            check=True,
        )

        temp_path.replace(
            output_path
        )

        try:

            concat_path.unlink()

        except FileNotFoundError:

            pass

        return output_path
