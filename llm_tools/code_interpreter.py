from typing import Union, Tuple

import json5

from agent.schema import ResponseStatus
from llm_tools import BaseTool
from utils.code_interpreter import CodeInterpreter
from utils.utils import extract_code


class CodeInterpreterTool(BaseTool):
    name = 'code_interpreter'
    description = 'Python代码沙盒，可用于执行Python代码。'
    parameters = [{'name': 'code',
                   'type': 'string',
                   'description': '待执行的代码',
                   'required': True}]

    def __init__(self):
        super().__init__()
        self.code_interpreter = CodeInterpreter()

    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            fmt = '此工具的输入应为Markdown代码块。'
        return fmt

    def call(self, params: Union[str, dict], **kwargs) -> Tuple:
        try:
            params = json5.loads(params)
            code = params['code']
        except Exception:
            code = extract_code(params)

        if not code.strip():
            return ResponseStatus.STR, ''

        return ResponseStatus.STR, self.code_interpreter.execute_code(code)
