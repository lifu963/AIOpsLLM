from typing import List

import gradio as gr

from llm import LLM
from settings import MODEL, QWEN_URL, DASHSCOPE_API_KEY
from agent.agent_router import AgentRouter
from agent.schema import ResponseStatus

llm = LLM(model=MODEL, base_url=QWEN_URL, api_key=DASHSCOPE_API_KEY)
router = AgentRouter(llm=llm)


with gr.Blocks() as demo:

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def reset():
        router.reset()
        return None

    def user(query: str, history: List):
        return "", history + [[query, None]]

    def bot(history):
        history[-1][1] = ""
        for output in router(history[-1][0]):
            if output[0] == ResponseStatus.IMG:
                history += [[None, (output[1],)]]
            elif output[0] == ResponseStatus.STR:
                if type(history[-1][1]) is tuple:
                    history += [[None, output[1]]]
                else:
                    history[-1][1] += output[1]
            yield history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(reset, None, chatbot, queue=False)

demo.queue()
demo.launch()
