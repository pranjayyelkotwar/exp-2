from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from models.base import BaseLLM


class LlamaModel(BaseLLM):

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
    ) -> str:

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        generated = outputs[0][inputs.shape[-1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )