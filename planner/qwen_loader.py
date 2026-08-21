import gc

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from planner.config import (
    QWEN_KAGGLE_PATH,
    QWEN_LOCAL_PATH,
    QWEN_MODEL_ID,
    QWEN_MAX_NEW_TOKENS,
    QWEN_STORY_TEMPERATURE,
    QWEN_TOP_P,
)


class QwenStoryModel:

    """
    Loads and manages Qwen for:

    - Story creation
    - Character detection
    - Character planning
    - Scene planning
    - Shot planning

    The model is loaded only when required and can
    be unloaded before LTX generation begins.
    """

    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
    ):

        self.model_id = model_id

        self.model = None
        self.tokenizer = None

        self.model_path = (
            self._resolve_model_path()
        )

    def _resolve_model_path(self):

        required_files = (
            "config.json",
            "model.safetensors.index.json",
        )

        # ------------------------------------------------------------
        # Exact Kaggle dataset path
        # ------------------------------------------------------------

        if QWEN_KAGGLE_PATH.is_dir():

            if all(
                (
                    QWEN_KAGGLE_PATH
                    / filename
                ).is_file()
                for filename in required_files
            ):

                return QWEN_KAGGLE_PATH

            # --------------------------------------------------------
            # Search recursively inside the attached Kaggle dataset.
            # --------------------------------------------------------

            for config_path in (
                QWEN_KAGGLE_PATH.rglob(
                    "config.json"
                )
            ):

                model_path = (
                    config_path.parent
                )

                if (
                    model_path
                    / "model.safetensors.index.json"
                ).is_file():

                    return model_path

        # ------------------------------------------------------------
        # Local development path
        # ------------------------------------------------------------

        if QWEN_LOCAL_PATH.is_dir():

            if all(
                (
                    QWEN_LOCAL_PATH
                    / filename
                ).is_file()
                for filename in required_files
            ):

                return QWEN_LOCAL_PATH

        # ------------------------------------------------------------
        # Model not found
        # ------------------------------------------------------------

        raise FileNotFoundError(
            "Qwen model was not found.\n\n"
            "Expected Kaggle dataset path:\n"
            f"{QWEN_KAGGLE_PATH}\n\n"
            "Expected local development path:\n"
            f"{QWEN_LOCAL_PATH}\n\n"
            "The Qwen dataset must contain:\n"
            "  config.json\n"
            "  model.safetensors.index.json"
        )

    def load(self):

        if self.model is not None:
            return

        print("=" * 60)
        print("Loading Qwen story planner")
        print(f"Model: {self.model_id}")
        print(f"Path: {self.model_path}")
        print("=" * 60)

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
            )
        )

        self.model.eval()

        print(
            "Qwen story planner loaded successfully."
        )

    def generate(
        self,
        messages: list,
        max_new_tokens: int = (
            QWEN_MAX_NEW_TOKENS
        ),
        temperature: float = (
            QWEN_STORY_TEMPERATURE
        ),
        top_p: float = (
            QWEN_TOP_P
        ),
    ) -> str:

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

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        )

        model_inputs = {
            key: value.to(
                self.model.device
            )
            for key, value
            in model_inputs.items()
        }

        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": (
                max_new_tokens
            ),
        }

        if temperature <= 0:

            generation_kwargs[
                "do_sample"
            ] = False

        else:

            generation_kwargs[
                "do_sample"
            ] = True

            generation_kwargs[
                "temperature"
            ] = temperature

            generation_kwargs[
                "top_p"
            ] = top_p

        with torch.no_grad():

            output_ids = (
                self.model.generate(
                    **generation_kwargs
                )
            )

        generated_ids = output_ids[
            :,
            model_inputs[
                "input_ids"
            ].shape[1]:
        ]

        response = (
            self.tokenizer
            .batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
        )

        return response.strip()

    def unload(self):

        print(
            "Unloading Qwen story planner..."
        )

        if self.model is not None:

            del self.model
            self.model = None

        if self.tokenizer is not None:

            del self.tokenizer
            self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            try:

                torch.cuda.ipc_collect()

            except RuntimeError:

                pass

        print(
            "Qwen memory released."
        )
