from functools import partial

from openai import OpenAI


class LLM:
    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        self.chat = partial(self.client.chat.completions.create, model=self.model)
