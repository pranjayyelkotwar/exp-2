from models.llama import LlamaModel
from core.storage import JsonlStorage
from core.generator import ParaphraseGenerator


MODEL_NAME = "meta-llama/Llama-3.2-1B"

QUESTION = """
How can I improve the robustness of a retrieval-augmented generation system against noisy documents?
"""


def main():

    model = LlamaModel(MODEL_NAME)

    storage = JsonlStorage(
        "paraphrases.jsonl"
    )

    generator = ParaphraseGenerator(
        llm=model,
        storage=storage,
        target_size=250,
        temperature=1.0,
    )

    generator.run(QUESTION)


if __name__ == "__main__":
    main()