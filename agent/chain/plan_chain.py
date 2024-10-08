import json
from typing import List

from llm.schema import ROLE, USER, CONTENT
from settings import MODEL
from utils.utils import extract_json

PROMPT_PLANNER = """
你是一名AI专家,请帮助用户制定一条机器学习工作流!

上下文:
```
任务类别: {task_name}
已选模型: {model_name}
已选数据集: {dataset_name}
用户请求: {query}
```

你可以选择以下步骤构建你的工作流：

- download_model: 下载预训练模型参数到本地
- tune: 调整和优化模型的超参数；超参数是指在模型训练前设置的参数，这些参数会影响模型训练的过程和最终的表现
- train: 训练模型
- validate: 评估模型性能
- upload_model: 上传模型参数到云端

请仅用json格式输出你构建的工作流的步骤，每个步骤必须是download_model、tune、train、validate、upload_model中的一个，不要包含任何其他内容！例如:{{"steps": ["step_1", "step_2", "step_3"]}}
"""


class PlanChain:

    def __init__(self, llm):
        self.llm = llm

    def call(self, task: str, model: str, dataset: str, query: str) -> List:
        messages = [
            {ROLE: USER, CONTENT: PROMPT_PLANNER.format(task_name=task,
                                                        model_name=model,
                                                        dataset_name=dataset,
                                                        query=query)}]

        completion = self.llm.chat(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=messages
        )

        res = extract_json(completion.choices[0].message.content)
        steps = json.loads(res)['steps']
        return steps
