import json
from typing import Optional, Union

import pandas as pd

from log import logger
from llm import LLM
from llm.schema import ROLE, USER, CONTENT, SYSTEM
from llm_tools import BaseTool
from settings import MODEL, QWEN_URL, DASHSCOPE_API_KEY
from utils.utils import get_tool_definition

SYSTEM_PROMPT = """你是MLOps平台的智能助手，专注于帮助用户了解平台上的数据集、模型和低代码开发组件。
请调用`display_platform_assets`工具，以查询MLOps平台的知识库，了解平台所有数据集、模型和组件的概况。"""


class DisplayTool(BaseTool):
    name = 'display_platform_assets'
    description = '查询MLOps平台中所有数据集、模型、组件的概况。可以根据资产类型和任务类型进行筛选，并选择输出模式。'
    parameters = [
        {
            'name': 'type',
            'type': 'string',
            'description': "资产类型。可以是 'model'（模型）、'dataset'（数据集）、'component'（组件）。如果不指定，将返回所有类型的资产。",
            'required': False,
            'enum': ["model", "dataset"]
        },
        {
            'name': 'task',
            'type': 'string',
            'description': "任务类型。可以是 'image_classification'（图像分类）、'object_detection'（目标检测）、'image_segmentation'（图像分割）。如果不指定，将返回所有任务类型的资产。",
            'required': False,
            'enum': ['image_classification', 'object_detection', 'image_segmentation']
        },
        {
            'name': 'mode',
            'type': 'string',
            'description': "输出模式。可以是 'detailed'（详细信息）、'simple'（简要信息）、'count'（符合条件的数量）。",
            'required': False,
            "enum": ["detailed", "simple", "count"],
        }
    ]

    _cache = {}

    def __init__(self):
        super().__init__()

        if not self._cache:
            self._load_files_into_cache()

        self.assets_df = pd.DataFrame(self._cache["assets_list"])

    def _load_files_into_cache(self):
        try:
            with open('./components_cfg.json', 'r') as file:
                components_cfg = json.load(file)
            with open('./datasets_cfg.json', 'r') as file:
                datasets_cfg = json.load(file)
            with open('./models_cfg.json', 'r') as file:
                models_cfg = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading configuration files: {e}")

        for item in datasets_cfg:
            item['type'] = 'dataset'
        for item in models_cfg:
            item['type'] = 'model'
        for item in components_cfg:
            item['type'] = 'component'

        assets_list = []
        assets_list.extend(components_cfg)
        assets_list.extend(datasets_cfg)
        assets_list.extend(models_cfg)

        self._cache["assets_list"] = assets_list

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        return self.display_platform_assets(**params)

    def display_platform_assets(self,
                                type: Optional[str] = None,
                                task: Optional[str] = None,
                                mode: Optional[str] = 'detailed'
                                ):

        type_dict = {
            'model': '模型',
            'dataset': '数据集',
            'component': '组件'
        }

        task_dict = {
            'image_classification': 'image_classification(图像分类)',
            'object_detection': 'object_detection(目标检测)',
            'image_segmentation': 'image_segmentation(图像分割)'
        }

        display_df = self.assets_df.copy()
        display_columns = ['name', 'task', 'type', 'summary']

        if type is not None and type in ['model', 'dataset', 'component']:
            display_df = self.assets_df[self.assets_df['type'] == type]

        if task is not None and task in ["image_classification", "object_detection", "image_segmentation"]:
            display_df = self.assets_df[self.assets_df['task'] == task]

        type_cn = type_dict.get(type, '资产')
        task_cn = task_dict.get(task, 'unknown')

        if mode == 'count':
            count = len(display_df)
            return f"该MLOps平台中有{count}个用于{task_cn}的{type_cn}." if task_cn != "unknown" else f"该MLOps平台中有{count}个{type_cn}."

        if mode == 'simple':
            display_columns.remove('summary')

        display_df.reset_index(inplace=True)

        result_header = f"该MLOps平台中用于{task_cn}的{type_cn}概况:\n" if task_cn != "unknown" else "该MLOps平台中的{type_cn}概况:\n"
        result_body = display_df[display_columns].to_csv()

        return result_header + result_body


class QueryAPIRetriever:
    def __init__(self):
        self.display_api = DisplayTool()
        self.llm = LLM(model=MODEL, base_url=QWEN_URL, api_key=DASHSCOPE_API_KEY)
        self.tool_info = get_tool_definition(self.display_api)

    def process(self, query: str) -> str:
        messages = [
            {ROLE: SYSTEM, CONTENT: SYSTEM_PROMPT},
            {ROLE: USER, CONTENT: query}
        ]

        try:
            completion = self.llm.chat(messages=messages, tools=[self.tool_info])
            if not completion.choices[0].message.tool_calls:
                return ""
            args = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
            res = self.display_api.call(args)

            return res
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {e}")
            return ""
        except KeyError as e:
            logger.error(f"Missing key in LLM response: {e}")
            return f""
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return f""
