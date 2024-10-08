from typing import List, Optional

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.schema import Document
from llama_index.embeddings.dashscope import DashScopeEmbedding

from utils.spliter import split_sentences
from log import logger
from settings import DASHSCOPE_API_KEY


class VectorStoreManager:
    def __init__(self, persist_dir: str = './storage/docs'):
        self.embedding = DashScopeEmbedding(model_name='text-embedding-v2', api_key=DASHSCOPE_API_KEY)
        self.persist_dir = persist_dir

    def load_from_documents(self, documents: List[Document], persist_dir: Optional[str] = None) -> VectorStoreIndex:

        persist_dir = persist_dir or self.persist_dir

        try:
            node_parser = SentenceWindowNodeParser.from_defaults(
                sentence_splitter=split_sentences,
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
            nodes = node_parser.get_nodes_from_documents(documents)
            vector_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=StorageContext.from_defaults(),
                embed_model=self.embedding,
            )
            vector_index.storage_context.persist(persist_dir=persist_dir)
            logger.info(f"Vector store successfully persisted to {persist_dir}")
        except Exception as e:
            logger.error(f"Error while loading from documents: {str(e)}")
            raise

        return vector_index

    def load_from_storage(self, persist_dir: Optional[str] = None) -> VectorStoreIndex:

        persist_dir = persist_dir or self.persist_dir

        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            vector_index = load_index_from_storage(storage_context, embed_model=self.embedding)
            logger.info(f"Vector store successfully loaded from {persist_dir}")
        except Exception as e:
            logger.error(f"Error while loading from storage: {str(e)}")
            raise

        return vector_index
