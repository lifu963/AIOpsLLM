from collections import deque
import json
from typing import List, Dict
import math

from llm.schema import ROLE, USER, CONTENT
from settings import MODEL
from utils.utils import extract_json

PROMPT_PIPELINE = """
你是一名AI专家,请帮助用户为机器学习工作流选择当前阶段最适合的组件！

当前阶段: {step}

上下文:
```
任务类别: {task_name}
已选模型: {model_name}
已选数据集: {dataset_name}
工作流: {workflow}
已选组件: {selected_components}
当前阶段: {step}
```

你可以选择以下组件：

{component_descs}

请仅用json格式输出你为当前阶段选择的组件，组件应该是{component_names}之一,例如:
{{
    "component": "example_component"
}}
"""


class PipelineChain:

    def __init__(self, llm, components: List):
        self.llm = llm
        self.components_dict = self._build_components_dict(components)

    def _build_components_dict(self, components_data: List) -> Dict:
        _components_dict = {}
        for item in components_data:
            name = item['name']
            _components_dict[name] = {'description': item['description']}
            if 'children' in item:
                _components_dict[name]['children'] = self._build_components_dict(item['children'])
            else:
                _components_dict[name]['path'] = item['path']
        return _components_dict

    def call(self, task: str, model: str, dataset: str, steps: List) -> List:
        selected_components = {}

        for step in steps:
            if step in ['download_model', 'upload_model']:
                component = step
            else:
                recalled_components = self.components_dict['Pytorch']['children'][step]['children'][task]['children']
                component = self.select_component(recalled_components, task, model, dataset, step, steps, selected_components)
            yield component
            selected_components[step] = component

    def select_component(self,
                         recalled_components: Dict,
                         task: str,
                         model: str,
                         dataset: str,
                         step: str,
                         steps: List,
                         selected_components: Dict):
        workflow = ' -- '.join(steps)
        selected_components_str = '' if not selected_components \
            else '--'.join(selected_components[step] for step in steps if step in selected_components)
        cur_queue = deque([(key, value['description']) for key, value in recalled_components.items()])
        tmp_components = []
        ans_component = 'none'
        while len(cur_queue) > 0:
            cur_queue_len = len(cur_queue)
            if cur_queue_len == 1:
                ans_component = str(cur_queue.popleft()[0])
                break
            tmp_threshold = math.ceil(cur_queue_len / math.ceil(cur_queue_len / 5))
            for i in range(cur_queue_len):
                tmp_components.append(cur_queue.popleft())
                if len(tmp_components) == tmp_threshold or i == cur_queue_len - 1:
                    component_descs = ""
                    component_names = ""
                    for idx, item in enumerate(tmp_components, start=1):
                        component_descs += f"{idx}. {item[0]}: {item[1]}\n"
                        component_names += f"{item[0]},"

                    if cur_queue_len > 5:
                        component_descs += f"{len(tmp_components) + 1}. none: 如果没有合适的组件，请选择none.\n"
                        component_names += "none"

                    messages = [{ROLE: USER, CONTENT: PROMPT_PIPELINE.format(task_name=task,
                                                                             model_name=model,
                                                                             dataset_name=dataset,
                                                                             workflow=workflow,
                                                                             step=step,
                                                                             component_names=component_names,
                                                                             component_descs=component_descs,
                                                                             selected_components=selected_components_str)}]
                    completion = self.llm.chat(
                        model=MODEL,
                        response_format={"type": "json_object"},
                        messages=messages
                    )
                    res = extract_json(completion.choices[0].message.content)
                    component = json.loads(res)['component']
                    if component != 'none':
                        cur_queue.append((component, recalled_components[component]['description']))
                    tmp_components.clear()
        return ans_component
