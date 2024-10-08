from typing import List, Optional

import dashscope
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import ReRankEndEvent, ReRankStartEvent
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode

from log import logger
from settings import DASHSCOPE_API_KEY

dispatcher = get_dispatcher(__name__)


class GTERerank(BaseNodePostprocessor):
    model: str = Field(description="GTE model name.")
    api_key: str = Field(description="DashScope api key.")
    top_n: int = Field(description="Top N nodes to return.")

    def __init__(self, top_n: int = 5):
        super().__init__(top_n=top_n, model='gte-rerank', api_key=DASHSCOPE_API_KEY)
        self.top_n = top_n
        self.model = 'gte-rerank'
        self.api_key = DASHSCOPE_API_KEY

    @classmethod
    def class_name(cls) -> str:
        return "GTERerank"

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:

        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle, nodes=nodes, top_n=self.top_n, model_name=self.model
            )
        )

        if query_bundle is None:
            raise ValueError("Query bundle is missing.")
        if not query_bundle.query_str:
            raise ValueError("Query string in query_bundle is empty.")
        if len(nodes) == 0:
            logger.warning("No nodes provided for reranking.")
            return []

        with self.callback_manager.event(
                CBEventType.RERANKING,
                payload={
                    EventPayload.NODES: nodes,
                    EventPayload.MODEL_NAME: self.model,
                    EventPayload.QUERY_STR: query_bundle.query_str,
                    EventPayload.TOP_K: self.top_n,
                }) as event:

            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in nodes
            ]

            try:
                response = dashscope.TextReRank.call(
                    model=self.model,
                    api_key=self.api_key,
                    query=query_bundle.query_str,
                    documents=texts,
                    top_n=self.top_n,
                    return_documents=True
                )
            except Exception as e:
                logger.error(f"Error during DashScope API call: {e}")
                raise

            new_nodes = []
            for result in response.output.results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result.index].node, score=result.relevance_score
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
