from models.llama import LlamaModel
from core.storage import JsonlStorage
from core.generator import ParaphraseGenerator


MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

QUESTION = """
How can I improve the robustness of a retrieval-augmented generation system against noisy documents?
"""


def main():

    model = LlamaModel(
        MODEL_NAME
    )

    storage = JsonlStorage(
        "paraphrases.jsonl"
    )

    generator = ParaphraseGenerator(
        llm=model,
        storage=storage,
        target_size=250,
    )

    generator.run(
        QUESTION
    )


if __name__ == "__main__":
    main()