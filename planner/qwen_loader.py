from __future__ import annotations

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
        required = (
            "config.json",
            "model.safetensors.index.json",
        )

        candidates = [
            QWEN_KAGGLE_PATH,
            QWEN_LOCAL_PATH,
        ]

        for root in candidates:

            if not root.is_dir():
                continue

            if all(
                (
                    root / file
                ).is_file()
                for file in required
            ):
                return root

            for config in root.rglob(
                "config.json"
            ):
                model_dir = (
                    config.parent
                )

                if (
                    model_dir
                    / "model.safetensors.index.json"
                ).is_file():
                    return model_dir

        raise FileNotFoundError(
            "Qwen model not found."
        )

    def load(self):

        if self.model is not None:
            return

        print(
            "Loading Qwen planner:"
        )
        print(
            self.model_path
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
            )
        )

        self.model.eval()

    def generate(
        self,
        messages: list,
        max_new_tokens=QWEN_MAX_NEW_TOKENS,
        temperature=QWEN_STORY_TEMPERATURE,
        top_p=QWEN_TOP_P,
    ) -> str:

        if self.model is None:
            self.load()

        prompt = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
        )

        device = (
            next(
                self.model.parameters()
            ).device
        )

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        kwargs = {
            **inputs,
            "max_new_tokens": (
                max_new_tokens
            ),
        }

        if temperature <= 0:
            kwargs["do_sample"] = False
        else:
            kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

        with torch.inference_mode():
            output = (
                self.model.generate(
                    **kwargs
                )
            )

        generated = output[
            :,
            inputs["input_ids"].shape[1]:,
        ]

        return (
            self.tokenizer
            .batch_decode(
                generated,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

    def unload(self):

        print(
            "Releasing Qwen before H3..."
        )

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            for device_id in range(
                torch.cuda.device_count()
            ):
                with torch.cuda.device(
                    device_id
                ):
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass

        print(
            "Qwen completely released."
        )
