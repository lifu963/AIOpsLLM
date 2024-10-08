from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SimilarityPostprocessor

from log import logger
from utils.rerank import GTERerank


class QueryVectorRetriever:
    def __init__(self, vector_index: VectorStoreIndex):
        self.vector_retriever = vector_index.as_retriever(similarity_top_k=5)
        self.raw_filter_processor = SimilarityPostprocessor(similarity_cutoff=0.3)
        self.window_filter_processor = SimilarityPostprocessor(similarity_cutoff=0.4)
        self.window_processor = MetadataReplacementPostProcessor(target_metadata_key="window")
        self.rerank_postprocessor = GTERerank()

    def process(self, query):
        try:
            res = self.vector_retriever.retrieve(query)
            res = self.raw_filter_processor.postprocess_nodes(res)
            res = self.window_processor.postprocess_nodes(res)
            res = self.rerank_postprocessor.postprocess_nodes(query_str=query, nodes=res)
            res = self.window_filter_processor.postprocess_nodes(res)
            return res
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return []
