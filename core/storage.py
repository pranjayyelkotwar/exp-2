import json


class JsonlStorage:

    def __init__(self, path):

        self.path = path

        open(
            self.path,
            "w",
            encoding="utf-8",
        ).close()

    def save(self, row):

        with open(
            self.path,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            f.write("\n")