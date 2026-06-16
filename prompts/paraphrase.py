import json


def build_paraphrase_prompt(question: str) -> str:

    return f"""
You are generating paraphrases.

Requirements:

1. Preserve meaning exactly.
2. Preserve answerability.
3. Do not add information.
4. Do not remove information.
5. Do not change constraints.
6. Maximize linguistic diversity.

Generate exactly 4 paraphrases.

Paraphrase 1:
- Formal wording

Paraphrase 2:
- Conversational wording

Paraphrase 3:
- Different framing

Paraphrase 4:
- Different syntax

Return ONLY valid JSON.

Example:

{{
    "paraphrases": [
        "...",
        "...",
        "...",
        "..."
    ]
}}

Question:

{question}
"""