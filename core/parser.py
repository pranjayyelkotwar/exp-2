import json
import re


def extract_paraphrases(text: str):

    try:

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL,
        )

        if not match:
            return []

        obj = json.loads(match.group())

        paras = obj.get(
            "paraphrases",
            []
        )

        cleaned = []

        for p in paras:

            if isinstance(p, str):
                cleaned.append(
                    p.strip()
                )

        return cleaned

    except Exception:

        return []