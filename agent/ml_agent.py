import json
from typing import Optional, Iterator, Tuple

from agent.chain import MLChain
from llm import LLM


class MLAgent:
    def __init__(self,
                 llm: LLM,
                 name: Optional[str] = None,
                 description: Optional[str] = None):

        self.name = name
        self.description = description

        with open('models_cfg.json', 'r') as file:
            models_cfg = json.load(file)
        with open('datasets_cfg.json', 'r') as file:
            datasets_cfg = json.load(file)
        with open('components.json', 'r') as file:
            components = json.load(file)
        task_names = ["image_classification", "object_detection", "image_segmentation"]

        self.ml_chain = MLChain(llm=llm,
                                task_names=task_names,
                                datasets=datasets_cfg,
                                models=models_cfg,
                                components=components)

    def __call__(self, query: str) -> Iterator[Tuple]:
        return self.ml_chain.call(query)
