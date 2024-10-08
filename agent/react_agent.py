from enum import Enum
import json
from typing import Tuple, List, Optional, Union, Iterator
import traceback
import re

from agent.schema import ResponseStatus
from llm import LLM
from llm.schema import ROLE, CONTENT, SYSTEM, USER
from llm_tools import BaseTool

SYSTEM_PROMPT = """you are a helpful assistant."""

FEEDBACK_PROMPT = """请根据以下用户提示进行重新推理，并确保避免出现以下错误推理。请注意，当前的用户提示优先级高于下文的Question！

错误推理： {error_output}

用户提示： {tip}
"""

TOOL_DESC = (
    '{name}: {description}\n'
    'Parameters: {parameters} {args_format}')

PROMPT_REACT = """Now, answer the following questions as best you can. You have access to the following llm_tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
Thought: """

PROMPT_FEEDBACK = "\n是否继续？"

MAX_SHOWN_OBSERVATION = 256
MAX_LLM_CALL_PER_RUN = 8


class ReactAgentStatus(Enum):
    START = 1
    THINK = 2
    ACT = 3


class ReActAgent:

    def __init__(self,
                 llm: LLM,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 tools: Optional[List[BaseTool]] = None,
                 is_human_in_loop: bool = False):

        self.name = name
        self.description = description
        self.llm = llm
        self.is_human_in_loop = is_human_in_loop
        self.tools = tools
        self.function_map = {}
        if self.tools:
            for tool in self.tools:
                self.function_map[tool.name] = tool

        self.status = ReactAgentStatus.START
        self.current_messages = []
        self.current_output = ''
        self.current_thought = ''
        self.current_action = ''
        self.current_action_input = ''
        self.current_observation = ''
        self.current_tip = ''
        self.current_error_output = ''

    def reset(self):
        self.status = ReactAgentStatus.START
        self.current_messages = []
        self.current_output = ''
        self.current_thought = ''
        self.current_action = ''
        self.current_action_input = ''
        self.current_observation = ''
        self.current_tip = ''
        self.current_error_output = ''

    def start(self, query: str):
        self.reset()
        self.current_messages = self._prepend_react_prompt(query)
        self.status = ReactAgentStatus.THINK

    def think(self) -> Tuple:
        if self.status != ReactAgentStatus.THINK:
            raise Exception("Cannot think: current Agent is not in THINK state.")

        if self.current_tip != '':
            if len(self.current_messages) > 2:
                self.current_messages[-2][CONTENT] = FEEDBACK_PROMPT.format(tip=self.current_tip,
                                                                            error_output=self.current_error_output)
            else:
                self.current_messages.insert(-1, {ROLE: SYSTEM, CONTENT: FEEDBACK_PROMPT.format(tip=self.current_tip,
                                                                                                error_output=self.current_error_output)})
        else:
            if len(self.current_messages) > 2:
                self.current_messages.pop(-2)

        completion = self.llm.chat(
            messages=self.current_messages,
            stop=['Observation:', 'Observation:\n'])

        self.current_output = completion.choices[0].message.content
        return ResponseStatus.STR, 'Thought: ' + self.current_output

    def act(self) -> Tuple:
        if self.status != ReactAgentStatus.ACT:
            raise Exception("Cannot act: current Agent is not in ACT state.")

        self.reset_feedback_msg()
        has_action, action, action_input, thought = self._detect_tool(self.current_output)
        if not has_action:
            match = re.search(r"Final Answer:\s*(.*)", thought)
            if match:
                ans = match.group(1)
                return ResponseStatus.FIN, ans
            else:
                return ()
        response = self._call_tool(action, action_input)
        observation = response[1]
        shown_observation = observation
        if ResponseStatus.STR == response[0] and len(observation) > MAX_SHOWN_OBSERVATION:
            shown_observation = observation[:MAX_SHOWN_OBSERVATION] + "…………\n"
        if ResponseStatus.IMG == response[0]:
            shown_observation = "图片已生成。"
        observation = f'\nObservation: {observation}\nThought: '
        shown_observation = f'\nObservation: {shown_observation}\n'
        self.current_thought = thought
        self.current_action = action
        self.current_action_input = action_input
        self.current_observation = observation
        return ResponseStatus.STR, shown_observation

    def observe(self):
        if self.status != ReactAgentStatus.ACT:
            raise Exception("Cannot observe: current Agent is not in ACT state.")
        self.reset_feedback_msg()
        if not (self.current_messages[-1][CONTENT].endswith('\nThought: ')) and (not self.current_thought.startswith('\n')):
            self.current_messages[-1][CONTENT] += '\n'
        if self.current_action_input.startswith('```'):
            self.current_action_input = '\n' + self.current_action_input
        self.current_messages[-1][CONTENT] += self.current_thought + f'\nAction: {self.current_action}\nAction Input:{self.current_action_input}' + self.current_observation

    def feedback(self, tip: str, error_output: str):
        self.current_tip = tip
        self.current_error_output = error_output

    def reset_feedback_msg(self):
        self.current_tip = ''
        self.current_error_output = ''

    def switch_think_to_act(self):
        if self.status != ReactAgentStatus.THINK:
            raise Exception("Cannot switch to ACT: current Agent is not in THINK state.")
        self.status = ReactAgentStatus.ACT

    def switch_act_to_think(self):
        if self.status != ReactAgentStatus.ACT:
            raise Exception("Cannot switch to THINK: current Agent is not in ACT state.")
        self.status = ReactAgentStatus.THINK

    def auto_run(self, query: str) -> Iterator[Tuple]:
        self.start(query)
        while True:
            response = self.think()
            self.switch_think_to_act()
            yield response

            response = self.act()
            if not response:
                break
            if response[0] == ResponseStatus.FIN:
                yield response
                break
            yield response
            self.observe()
            self.switch_act_to_think()

    def _prepend_react_prompt(self, query: str) -> List:
        tool_descs = []
        for tool in self.tools:
            function = tool.function
            tool_descs.append(
                TOOL_DESC.format(name=function['name'],
                                 description=function['description'],
                                 parameters=json.dumps(function['parameters'], ensure_ascii=False),
                                 args_format=function.get('args_format', '')).rstrip())
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool.name for tool in self.tools)
        messages = [{ROLE: SYSTEM, CONTENT: SYSTEM_PROMPT},
                    {ROLE: USER, CONTENT: PROMPT_REACT.format(tool_descs=tool_descs,
                                                              tool_names=tool_names,
                                                              query=query)}]
        return messages

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}') -> str:
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'
        tool = self.function_map[tool_name]
        try:
            tool_result = tool.call(tool_args)
        except Exception as e:
            exception_type = type(e).__name__
            exception_message = str(e)
            traceback_info = ''.join(traceback.format_tb(e.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            print(error_message)
            return error_message
        return tool_result

    @staticmethod
    def _detect_tool(text: str) -> Tuple[bool, str, str, str]:
        special_func_token = '\nAction:'
        special_args_token = '\nAction Input:'
        special_obs_token = '\nObservation:'
        func_name, func_args = None, None
        i = text.rfind(special_func_token)
        j = text.rfind(special_args_token)
        k = text.rfind(special_obs_token)
        if 0 <= i < j:
            if k < j:
                text = text.rstrip() + special_obs_token
            k = text.rfind(special_obs_token)
            func_name = text[i + len(special_func_token):j].strip()
            func_args = text[j + len(special_args_token):k].strip()
            text = text[:i]

        return (func_name is not None), func_name, func_args, text
