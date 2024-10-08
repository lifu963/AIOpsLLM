import re
import traceback
from typing import List, Dict
import sys

import json5

from log import logger


def extract_json(text: str) -> str:
    triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
    if triple_match:
        text = triple_match.group(1)
    return text


def extract_code(text: str) -> str:
    triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
    if triple_match:
        text = triple_match.group(1)
    else:
        try:
            text = json5.loads(text)['code']
        except Exception:
            print_traceback(is_error=False)
    return text


def print_traceback(is_error: bool = True):
    tb = ''.join(traceback.format_exception(*sys.exc_info(), limit=3))
    if is_error:
        logger.error(tb)
    else:
        logger.warning(tb)


def extract_components(components: List[Dict], current_task=None, current_step=None):

    result = []

    for component in components:
        name = component.get('name')
        description = component.get('description', '')
        path = component.get('path', None)
        children = component.get('children', [])

        task = current_task
        step = current_step

        if name in ['image_classification', 'object_detection', 'image_segmentation']:
            task = name
        elif name in ['download_model', 'upload_model', 'tune', 'train', 'validate']:
            step = name

        if path:
            result.append({
                'name': name,
                'summary': description,
                'path': path,
                'task': task if task else 'all',
                'step': step if step else 'none'
            })

        if children:
            result.extend(extract_components(children, task, step))

    return result


def get_tool_definition(cls):
    return {
        "type": "function",
        "function": {
            "name": cls.name,
            "description": cls.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param['name']: {
                        "type": param['type'],
                        "description": param['description'],
                        **({"enum": param['enum']} if 'enum' in param else {}),
                        **({"default": param['default']} if 'default' in param else {})
                    }
                    for param in cls.parameters
                },
                "required": [param['name'] for param in cls.parameters if param.get('required', False)]
            }
        }
    }
