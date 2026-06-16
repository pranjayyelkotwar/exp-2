from dataclasses import dataclass
from tqdm import tqdm

from core.parser import extract_paraphrases


@dataclass
class Node:

    id: int
    parent_id: int | None
    depth: int
    text: str


PARAPHRASE_PROMPT = """
Generate EXACTLY 4 paraphrases.

Rules:

1. Preserve meaning exactly.
2. Preserve answerability.
3. Do not add information.
4. Do not remove information.
5. Maximize wording diversity.

Return ONLY JSON.

{
  "paraphrases": [
    "...",
    "...",
    "...",
    "..."
  ]
}

Question:

{question}
"""


class ParaphraseGenerator:

    def __init__(
        self,
        llm,
        storage,
        target_size=250,
    ):

        self.llm = llm
        self.storage = storage
        self.target_size = target_size

    def run(
        self,
        question: str,
    ):

        seen = set()

        next_id = 0

        root = Node(
            id=0,
            parent_id=None,
            depth=0,
            text=question,
        )

        self.storage.save(
            root.__dict__
        )

        frontier = [root]

        seen.add(
            question.strip()
        )

        total = 1

        next_id = 1

        pbar = tqdm(
            total=self.target_size
        )

        pbar.update(1)

        while (
            frontier
            and total < self.target_size
        ):

            new_frontier = []

            for parent in frontier:

                prompt = PARAPHRASE_PROMPT.format(
                    question=parent.text
                )

                response = self.llm.generate(
                    prompt
                )

                paraphrases = extract_paraphrases(
                    response
                )

                for para in paraphrases:

                    para = para.strip()

                    if not para:
                        continue

                    if para in seen:
                        continue

                    node = Node(
                        id=next_id,
                        parent_id=parent.id,
                        depth=parent.depth + 1,
                        text=para,
                    )

                    self.storage.save(
                        node.__dict__
                    )

                    seen.add(para)

                    new_frontier.append(
                        node
                    )

                    next_id += 1
                    total += 1

                    pbar.update(1)

                    if total >= self.target_size:
                        break

                if total >= self.target_size:
                    break

            frontier = new_frontier

        pbar.close()

        print(
            f"\nGenerated {total} paraphrases"
        )