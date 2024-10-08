from typing import Union, Tuple

from agent.schema import ResponseStatus
from llm_tools import BaseTool
from utils.image_gen import ImageGenSDK


class ImageGenTool(BaseTool):
    name = 'image_gen'
    description = 'AI绘画（图像生成）服务，输入文本描述，返回根据文本信息绘制的图片URL。'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '详细描述了希望生成的图像具有什么内容，例如人物、环境、动作等细节描述',
        'required': True
    }]

    def __init__(self):
        super().__init__()
        self.client = ImageGenSDK()

    def call(self, params: Union[str, dict], **kwargs) -> Tuple:
        params = self._verify_json_format_args(params)
        prompt = params['prompt']

        try:
            urls = self.client.call(prompt=prompt)
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")

        return ResponseStatus.IMG, urls[0]
