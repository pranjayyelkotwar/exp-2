import json


def parse_response(response: str):

    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end == -1:
        return []

    try:
        obj = json.loads(response[start:end + 1])
        return obj["paraphrases"]
    except Exception:
        return []