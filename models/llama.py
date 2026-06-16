from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LlamaModel:

    def __init__(self, model_name: str):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 1.2,
        max_new_tokens: int = 512,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Always follow formatting instructions exactly."
                )
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

        if getattr(self.tokenizer, "chat_template", None):

            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        else:

            full_prompt = f"""
SYSTEM:
You are a helpful assistant.

USER:
{prompt}

ASSISTANT:
"""

            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
            )["input_ids"]

        inputs = inputs.to(self.model.device)

        outputs = self.model.generate(
            inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][inputs.shape[1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )