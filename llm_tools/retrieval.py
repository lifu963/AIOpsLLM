from typing import Union, Tuple, Optional

from agent.schema import ResponseStatus
from llm_tools import BaseTool
from utils.retriever.query_retriever import QueryRetriever


class RetrievalTool(BaseTool):
    name = 'retrieval'
    description = '从知识库中检索与MLOps平台相关的问题内容'
    parameters = [{'name': 'query',
                   'type': 'string',
                   'description': '检索关键词，用逗号分隔，用于在知识库中匹配到相关的内容。由于知识库可能多语言，关键词最好中英文都有。',
                   'required': True}]

    def __init__(self,
                 retriever: Optional[QueryRetriever] = None):
        super().__init__()
        self.retriever = QueryRetriever() if not retriever else retriever

    def call(self, params: Union[str, dict], **kwargs) -> Tuple:
        params = self._verify_json_format_args(params)
        query = params.get('query', '')
        return ResponseStatus.STR, self.retriever.process(query)
