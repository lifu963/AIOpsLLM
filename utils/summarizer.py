import os
from typing import Optional

from tqdm import tqdm
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from llm import LLM
from llm.schema import ROLE, USER, CONTENT
from settings import MODEL, QWEN_URL, DASHSCOPE_API_KEY


PROMPT_SUMMARY = """你是一名AI专家！以下内容来自{name}的README.md：包括上文总结和当前片段。
请根据上文总结和当前片段，用一句话说明{name}的定义、特点、应用场景。
若上文总结中已经说明了{name}的定义、特点、应用场景，请充分利用上文总结！
若当前片段中包含代码段，请忽略代码段和它的解释！因为代码段和解释仅用于告诉用户如何加载并使用{name}。
上文总结：
{summing_up}
当前片段：
{content}
请用一句话说明{name}的定义、特点、应用场景！
"""


class DocumentSummarizer:

    def __init__(self):
        self.llm = LLM(model=MODEL, base_url=QWEN_URL, api_key=DASHSCOPE_API_KEY)
        self.chunk_size = 512
        self.node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_size//4)

    def __call__(self, path: str, name: Optional[str] = None):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        doc = Document(text=content)
        nodes = self.node_parser.get_nodes_from_documents([doc])
        if not name:
            name = os.path.basename(path).split('.')[0]

        res, last_res = '', ''
        for node in tqdm(nodes):
            content = node.get_content()
            messages = [{ROLE: USER, CONTENT: PROMPT_SUMMARY.format(name=name, summing_up=last_res, content=content)}]
            completion = self.llm.chat(messages=messages)
            res = completion.choices[0].message.content
            last_res = res
        return res
