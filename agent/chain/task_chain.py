import json
from typing import List


from utils.utils import extract_json
from settings import MODEL
from llm.schema import ROLE, USER, CONTENT


PROMPT_TASK_RECOGNITION = """
你是一名AI专家,请识别用户请求属于哪一类AI任务!
用户请求: {query}
任务必须是{task_names}之一。
请仅用json格式输出你识别的任务,例如:
{{
    "task": "example task"
}}
"""


class TaskChain:

    def __init__(self, llm,  task_names: List):
        self.llm = llm
        self.task_names = task_names

    def call(self, query: str) -> str:
        messages = [
            {ROLE: USER, CONTENT: PROMPT_TASK_RECOGNITION.format(task_names=','.join(self.task_names), query=query)}
        ]
        completion = self.llm.chat(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=messages
        )

        res = extract_json(completion.choices[0].message.content)
        task = json.loads(res)['task']

        return task
