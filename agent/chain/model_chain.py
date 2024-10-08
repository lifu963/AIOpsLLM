from collections import deque
import json
import math
from typing import List

from llm.schema import ROLE, USER, CONTENT
from settings import MODEL
from utils.utils import extract_json

PROMPT_MODEL_SELECTION = """
你是一名AI专家,请帮助用户选择最适合的模型!

上下文:
```
任务类别: {task_name}
用户请求: {query}
```

你可以选择以下模型：

{model_descs}

请仅用json格式输出你选择的模型，模型必须是{model_names}之一,例如:
{{
    "model": "example model"
}}
"""


class ModelChain:

    def __init__(self, llm, models: List):
        self.llm = llm
        self.models = models

    def call(self, query: str, task: str) -> str:
        recalled_models = []
        for item in self.models:
            if item.get("task") == task:
                recalled_models.append(item)

        model = self.select_model(recalled_models, task, query)

        return model

    def select_model(self, recalled_models: List, task: str, query: str) -> str:
        recalled_models_dict = {item['name']: item['summary'] for item in recalled_models}
        cur_queue = deque([(item['name'], item['summary']) for item in recalled_models])
        tmp_models = []
        ans_model = 'none'
        while len(cur_queue) > 0:
            cur_queue_len = len(cur_queue)
            if cur_queue_len == 1:
                ans_model = str(cur_queue.popleft()[0])
                break
            tmp_threshold = math.ceil(cur_queue_len / math.ceil(cur_queue_len / 5))
            for i in range(cur_queue_len):
                tmp_models.append(cur_queue.popleft())
                if len(tmp_models) == tmp_threshold or i == cur_queue_len - 1:
                    model_descs = ""
                    model_names = ""
                    for idx, item in enumerate(tmp_models, start=1):
                        model_descs += f"{idx}. {item[0]}: {item[1]}\n"
                        model_names += f"{item[0]},"

                    if cur_queue_len > 5:
                        model_descs += f"{len(tmp_models) + 1}. none: 如果没有合适的模型，请选择none.\n"
                        model_names += "none"

                    messages = [
                        {ROLE: USER, CONTENT: PROMPT_MODEL_SELECTION.format(task_name=task,
                                                                            query=query,
                                                                            model_names=model_names,
                                                                            model_descs=model_descs)}
                    ]
                    completion = self.llm.chat(
                        model=MODEL,
                        response_format={"type": "json_object"},
                        messages=messages
                    )

                    res = extract_json(completion.choices[0].message.content)
                    model = json.loads(res)['model']
                    if model != 'none':
                        cur_queue.append((model, recalled_models_dict[model]))
                    tmp_models.clear()
        return ans_model
