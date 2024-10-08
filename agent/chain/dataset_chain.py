from collections import deque
import json
from typing import List
import math

from llm import LLM
from llm.schema import ROLE, USER, CONTENT
from settings import MODEL
from utils.utils import extract_json

PROMPT_DATASET_SELECTION = """
你是一名AI专家,请帮助用户选择最适合的数据集!

上下文:
```
任务类别: {task_name}
用户请求: {query}
```

你可以选择以下数据集：

{dataset_descs}

请仅用json格式输出你选择的数据集，数据集必须是{dataset_names}之一，例如:
{{
    "dataset": "example dataset"
}}
"""


class DatasetChain:

    def __init__(self, llm: LLM, datasets: List):
        self.llm = llm
        self.datasets = datasets

    def call(self, query: str, task: str) -> str:
        recalled_datasets = [item for item in self.datasets if item.get("task") == task]
        return self.select_dataset(recalled_datasets, task, query)

    def select_dataset(self, recalled_datasets: List, task: str, query: str) -> str:
        recalled_datasets_dict = {item['name']: item['summary'] for item in recalled_datasets}
        cur_queue = deque([(item['name'], item['summary']) for item in recalled_datasets])
        selected_dataset = 'none'

        while len(cur_queue) > 0:
            cur_queue_len = len(cur_queue)
            if cur_queue_len == 1:
                selected_dataset = cur_queue.popleft()[0]
                break

            tmp_threshold = math.ceil(cur_queue_len / math.ceil(cur_queue_len / 5))
            tmp_datasets = []

            for i in range(cur_queue_len):
                tmp_datasets.append(cur_queue.popleft())
                if len(tmp_datasets) == tmp_threshold or i == cur_queue_len - 1:
                    dataset_descs = "\n".join(f"{idx + 1}. {name}: {summary}" for idx, (name, summary) in enumerate(tmp_datasets))
                    dataset_names = ",".join(name for name, _ in tmp_datasets)

                    if cur_queue_len > 5:
                        dataset_descs += f"\n{len(tmp_datasets) + 1}. none: 如果没有合适的数据集，请选择none."
                        dataset_names += ",none"

                    messages = [
                        {ROLE: USER, CONTENT: PROMPT_DATASET_SELECTION.format(
                            task_name=task,
                            query=query,
                            dataset_names=dataset_names,
                            dataset_descs=dataset_descs
                        )}
                    ]

                    try:
                        completion = self.llm.chat(
                            model=MODEL,
                            response_format={"type": "json_object"},
                            messages=messages
                        )
                        res = extract_json(completion.choices[0].message.content)
                        dataset = json.loads(res).get('dataset', 'none')
                        if dataset != 'none':
                            cur_queue.append((dataset, recalled_datasets_dict[dataset]))
                        tmp_datasets.clear()
                    except Exception as e:
                        raise ValueError(f"An error occurred: {e}")

        return selected_dataset
