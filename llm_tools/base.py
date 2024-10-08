from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional

import json5


class BaseTool(ABC):
    name: str = ''
    description: str = ''
    parameters: List[Dict] = []

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        if not self.name:
            raise ValueError(f'You must set {self.__class__.__name__}.name')

    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        raise NotImplementedError

    def _verify_json_format_args(self, params: Union[str, dict]) -> Union[str, dict]:
        try:
            if isinstance(params, str):
                params_json = json5.loads(params)
            else:
                params_json = params
            for param in self.parameters:
                if 'required' in param and param['required']:
                    if param['name'] not in params_json:
                        raise ValueError('Parameters %s is required!' % param['name'])
            return params_json
        except Exception:
            raise ValueError('Parameters cannot be converted to Json Format!')

    @property
    def function(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'args_format': self.args_format
        }

    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            fmt = '此工具的输入应为JSON对象。'
        return fmt
