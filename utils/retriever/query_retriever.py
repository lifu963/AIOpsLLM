import json

from llm import LLM
from llm.schema import ROLE, SYSTEM, USER, CONTENT
from settings import MODEL, QWEN_URL, DASHSCOPE_API_KEY
from utils.retriever.query_api_retriever import QueryAPIRetriever
from utils.retriever.query_vector_retriever import QueryVectorRetriever
from utils.vector_store_manager import VectorStoreManager


class QueryRetriever:

    TOOL_SELECTION_PROMPT = """请根据用户请求选择合适的检索工具，并严格按照指定的JSON格式返回结果：

如果请求涉及平台中数据集、模型、组件的概况，请选择 query_api_retriever。例如："平台中有几个组件？"
如果请求涉及具体数据集、模型或组件的详细信息，请选择 query_vector_retriever。例如："介绍一下用于数字识别的数据集。"
如果请求与上述情况均不涉及，则返回空字符。

请仅返回如下JSON格式，其中工具名称只能是 query_api_retriever、query_vector_retriever 或空字符之一：

```json
{
  "selected_tool": "工具名称"
}
"""

    def __init__(self):
        self.llm = LLM(model=MODEL, base_url=QWEN_URL, api_key=DASHSCOPE_API_KEY)
        self.api_retriever = QueryAPIRetriever()
        self.vector_index = VectorStoreManager().load_from_storage()
        self.vector_retriever = QueryVectorRetriever(self.vector_index)

    def select_tool(self, query):
        messages = [
            {ROLE: SYSTEM, CONTENT: self.TOOL_SELECTION_PROMPT},
            {ROLE: USER, CONTENT: query}
        ]
        try:
            completion = self.llm.chat(messages=messages, response_format={"type": "json_object"})
            args = json.loads(completion.choices[0].message.content)
            return args['selected_tool']
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error selecting tool: {e}")

    def process_request(self, query, selected_tool):
        if selected_tool == 'query_api_retriever':
            return self.api_retriever.process(query)
        elif selected_tool == 'query_vector_retriever':
            return self.vector_retriever.process(query)
        else:
            return ""

    def process(self, query):
        selected_tool = self.select_tool(query)
        res = self.process_request(query, selected_tool)
        if isinstance(res, list):
            res = '\n\n\n'.join(item.get_content() for item in res)
        return res
