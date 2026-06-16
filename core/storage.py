import json


class JsonlStorage:

    def __init__(self, path: str):
        self.path = path

    def append(self, record: dict):

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")