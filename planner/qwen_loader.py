import gc

from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from planner.config import (
    KAGGLE_INPUT_DIR,
    QWEN_LOCAL_PATH,
    QWEN_MAX_NEW_TOKENS,
    QWEN_MODEL_ID,
    QWEN_TEMPERATURE,
    QWEN_TOP_P,
)


class QwenStoryModel:
    """
    Loads and manages the Qwen story planning model.

    Loading priority:

    1. Explicit QWEN_LOCAL_PATH.
    2. Automatically detected model inside /kaggle/input.
    3. Hugging Face model ID.

    A local Kaggle model is always loaded with
    local_files_only=True to prevent repeated downloads.
    """

    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
    ):

        self.model_id = model_id

        self.model = None

        self.tokenizer = None

        self.model_source = None

        self.is_local_model = False

    def _is_model_directory(
        self,
        path: Path,
    ) -> bool:
        """
        Check whether a directory appears to contain
        a Hugging Face model.
        """

        if not path.is_dir():

            return False

        config_path = (
            path
            / "config.json"
        )

        if not config_path.is_file():

            return False

        has_tokenizer = any(
            [
                (
                    path
                    / "tokenizer.json"
                ).is_file(),

                (
                    path
                    / "tokenizer_config.json"
                ).is_file(),

                (
                    path
                    / "vocab.json"
                ).is_file(),
            ]
        )

        has_weights = any(
            path.glob(
                "*.safetensors"
            )
        ) or any(
            path.glob(
                "*.bin"
            )
        ) or (
            path
            / "model.safetensors.index.json"
        ).is_file() or (
            path
            / "pytorch_model.bin.index.json"
        ).is_file()

        return (
            has_tokenizer
            and has_weights
        )

    def _find_local_model(
        self,
    ) -> Path | None:
        """
        Find the Qwen model inside Kaggle input storage.
        """

        if QWEN_LOCAL_PATH is not None:

            if self._is_model_directory(
                QWEN_LOCAL_PATH
            ):

                return QWEN_LOCAL_PATH

            raise FileNotFoundError(
                "QWEN_LOCAL_PATH was provided, "
                "but it is not a valid Hugging Face "
                f"model directory:\n"
                f"{QWEN_LOCAL_PATH}"
            )

        if not KAGGLE_INPUT_DIR.exists():

            return None

        if self._is_model_directory(
            KAGGLE_INPUT_DIR
        ):

            return KAGGLE_INPUT_DIR

        for path in KAGGLE_INPUT_DIR.rglob(
            "config.json"
        ):

            candidate = (
                path.parent
            )

            if self._is_model_directory(
                candidate
            ):

                return candidate

        return None

    def _resolve_model_source(
        self,
    ):

        local_model_path = (
            self._find_local_model()
        )

        if local_model_path is not None:

            self.model_source = (
                str(
                    local_model_path
                )
            )

            self.is_local_model = True

            return self.model_source

        self.model_source = (
            self.model_id
        )

        self.is_local_model = False

        return self.model_source

    def load(
        self,
    ):
        """
        Load Qwen only when needed.
        """

        if self.model is not None:

            return

        source = (
            self._resolve_model_source()
        )

        print(
            "=" * 60
        )

        print(
            "Loading Qwen story planner"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Local model: "
            f"{self.is_local_model}"
        )

        print(
            "=" * 60
        )

        load_kwargs = {}

        if self.is_local_model:

            load_kwargs[
                "local_files_only"
            ] = True

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                source,
                **load_kwargs,
            )
        )

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }

        if self.is_local_model:

            model_kwargs[
                "local_files_only"
            ] = True

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                source,
                **model_kwargs,
            )
        )

        self.model.eval()

        print(
            "Qwen story planner "
            "loaded successfully."
        )

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            print(
                f"GPU memory allocated: "
                f"{allocated:.2f} GB"
            )

    def generate(
        self,
        messages: list,
        max_new_tokens: int = (
            QWEN_MAX_NEW_TOKENS
        ),
        temperature: float = (
            QWEN_TEMPERATURE
        ),
        top_p: float = QWEN_TOP_P,
    ) -> str:
        """
        Generate a response using Qwen chat format.
        """

        if self.model is None:

            self.load()

        text = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        model_inputs = (
            self.tokenizer(
                [text],
                return_tensors="pt",
            )
        )

        model_device = (
            next(
                self.model.parameters()
            ).device
        )

        model_inputs = {
            key: value.to(
                model_device
            )
            for key, value in (
                model_inputs.items()
            )
        }

        with torch.no_grad():

            output_ids = (
                self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            )

        generated_ids = (
            output_ids[
                :,
                model_inputs[
                    "input_ids"
                ].shape[1]:
            ]
        )

        response = (
            self.tokenizer
            .batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
        )

        return response.strip()

    def unload(
        self,
    ):
        """
        Release Qwen from memory.
        """

        if self.model is not None:

            print(
                "Unloading Qwen story planner..."
            )

            del self.model

            self.model = None

        if self.tokenizer is not None:

            del self.tokenizer

            self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            torch.cuda.synchronize()

            allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            reserved = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

            print(
                f"GPU allocated after unload: "
                f"{allocated:.2f} GB"
            )

            print(
                f"GPU reserved after unload: "
                f"{reserved:.2f} GB"
            )

        print(
            "Qwen memory released."
        )
