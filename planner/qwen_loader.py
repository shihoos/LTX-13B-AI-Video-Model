import gc

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from planner.config import (
    QWEN_MODEL_ID,
)


class QwenStoryModel:
    """
    Loads and manages Qwen3-4B-Instruct-2507.

    The model is used for:
    - Story creation
    - Story preservation
    - Character planning
    - Scene planning
    - Shot planning
    - Continuity planning
    """

    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
    ):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self):
        """
        Load Qwen only when needed.

        This allows the planner model to be released before
        LTX video generation begins.
        """

        if self.model is not None:
            return

        print("=" * 60)
        print("Loading Qwen story planner")
        print(f"Model: {self.model_id}")
        print("=" * 60)

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_id,
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto",
            )
        )

        self.model.eval()

        print("Qwen story planner loaded successfully.")

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> str:
        """
        Generate a response using Qwen chat format.
        """

        if self.model is None:
            self.load()

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():

            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        generated_ids = output_ids[
            :,
            model_inputs.input_ids.shape[1]:
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return response.strip()

    def unload(self):
        """
        Release Qwen from GPU memory.

        This should normally happen before LTX generation.
        """

        if self.model is not None:

            print("Unloading Qwen story planner...")

            del self.model
            self.model = None

        if self.tokenizer is not None:

            del self.tokenizer
            self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Qwen memory released.")
