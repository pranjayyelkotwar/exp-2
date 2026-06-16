from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
    ) -> str:
        pass