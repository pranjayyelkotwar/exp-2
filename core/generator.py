from dataclasses import dataclass
from tqdm import tqdm

from prompts.paraphrase import build_paraphrase_prompt
from core.parser import parse_response


@dataclass
class Node:
    id: int
    parent_id: int | None
    depth: int
    text: str


class ParaphraseGenerator:

    def __init__(
        self,
        llm,
        storage,
        target_size: int = 250,
        temperature: float = 1.0,
    ):
        self.llm = llm
        self.storage = storage
        self.target_size = target_size
        self.temperature = temperature

    def run(self, question: str):

        seen = set()

        node_id = 0

        root = Node(
            id=node_id,
            parent_id=None,
            depth=0,
            text=question,
        )

        self.storage.append(root.__dict__)

        seen.add(question)

        total_nodes = 1

        frontier = [root]

        pbar = tqdm(total=self.target_size)

        pbar.update(1)

        node_id += 1

        while frontier and total_nodes < self.target_size:

            next_frontier = []

            for parent in frontier:

                if total_nodes >= self.target_size:
                    break

                prompt = build_paraphrase_prompt(parent.text)

                response = self.llm.generate(
                    prompt,
                    temperature=self.temperature,
                )

                paraphrases = parse_response(response)

                for para in paraphrases:

                    para = para.strip()

                    if not para:
                        continue

                    if para in seen:
                        continue

                    node = Node(
                        id=node_id,
                        parent_id=parent.id,
                        depth=parent.depth + 1,
                        text=para,
                    )

                    self.storage.append(node.__dict__)

                    next_frontier.append(node)

                    seen.add(para)

                    node_id += 1
                    total_nodes += 1

                    pbar.update(1)

                    if total_nodes >= self.target_size:
                        break

            frontier = next_frontier

        pbar.close()