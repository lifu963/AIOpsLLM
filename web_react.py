from typing import List

import gradio as gr

from agent.react_agent import ReActAgent
from agent.schema import ResponseStatus
from llm import LLM
from llm_tools.code_interpreter import CodeInterpreterTool
from llm_tools.image_gen import ImageGenTool
from llm_tools.weather import WeatherTool
from llm_tools.retrieval import RetrievalTool
from settings import QWEN_URL, DASHSCOPE_API_KEY, MODEL

PROMPT_FEEDBACK = "\n是否继续？"


class Context:
    agent_last_output: str = ''

    def reset(self):
        self.agent_last_output = ''


llm = LLM(model=MODEL, base_url=QWEN_URL, api_key=DASHSCOPE_API_KEY)
tools = [CodeInterpreterTool(), WeatherTool(), ImageGenTool(), RetrievalTool()]
agent = ReActAgent(llm=llm, tools=tools)

context = Context()


with gr.Blocks() as demo:

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def reset():
        context.reset()
        return None

    # auto-run
    def user(query: str, history: List):
        return "", history + [[query, None]]

    def bot(history):
        history[-1][1] = ""
        for res in agent.auto_run(history[-1][0]):
            if res[0] != ResponseStatus.FIN:
                history[-1][1] += res[1]
                yield history

    # def user(query: str, history: List):
    #     query = query.strip()
    #     if agent.status == ReactAgentStatus.START:
    #         agent.start([{ROLE: USER, CONTENT: query}])
    #     elif agent.status == ReactAgentStatus.THINK:
    #         if query == '' or query[0].lower() == 'y':
    #             query = "Yes"
    #             agent.switch_think_to_act()
    #         else:
    #             if query[0].lower() == 'n':
    #                 query = "No"
    #             else:
    #                 agent.feedback(query, context.agent_last_output)
    #     else:
    #         if query == '' or query[0].lower() == 'y':
    #             query = "Yes"
    #             agent.observe()
    #         else:
    #             if query[0].lower() == 'n':
    #                 query = "No"
    #             else:
    #                 agent.feedback(query, context.agent_last_output)
    #         agent.switch_act_to_think()
    #     return "", history + [[query, None]]
    #
    #
    # def bot(history):
    #     if agent.status == ReactAgentStatus.THINK:
    #         response = agent.think()
    #         context.agent_last_output = response[0][CONTENT]
    #         history[-1][1] = response[0][CONTENT] + PROMPT_FEEDBACK
    #     else:
    #         response = agent.act()
    #         if not response:
    #             history[-1][1] = '\n完成。'
    #             agent.reset()
    #             context.reset()
    #         else:
    #             history[-1][1] = response[0][CONTENT] + PROMPT_FEEDBACK
    #     yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(reset, None, chatbot, queue=False)

demo.queue()
demo.launch()
