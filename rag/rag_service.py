"""
RAG 服务 - 一站式检索压缩服务
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from utils_tool.config_handler import config
from langchain_huggingface import HuggingFaceEmbeddings
from rag.mysql_store import mysql_store
from rag.hybrid_retriever import HybridRetriever
from rag.query_expander import QueryExpander
from rag.context_compressor import ThreeStageContextProcessor
from rag.parent_child_store import parent_child_store

# ========== 初始化模型 ==========
chat_model = ChatOpenAI(
    model=config.LLM_MODEL,
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    temperature=config.TEMPERATURE
)

embed_model = HuggingFaceEmbeddings(
    model_name=config.MODEL_NAME,
    encode_kwargs={'normalize_embeddings': True}
)


class RAGService:
    """RAG 服务 - 一站式检索压缩"""

    def __init__(
            self,
            vectorstore=None,
            child_documents: Optional[List[Document]] = None,
    ):
        self.mysql_store = mysql_store
        # 混合检索器
        self.hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            child_documents=child_documents,
            mysql_store=self.mysql_store,
        )
        # 查询扩展器
        self.query_expander = QueryExpander(llm=chat_model, cfg=config)
        # 三阶段上下文处理器
        self.context_processor = ThreeStageContextProcessor(
            embeddings=embed_model,
            llm=chat_model
        )

    def retrieve_and_compress(self, query: str) -> str:
        """
        一站式检索压缩

        Args:
            query: 用户查询

        Returns:
            压缩后的文档内容字符串
        """
        try:
            self.load_documents(folder_path=r"D:\360安全浏览器下载\pythonProject\AI大模型Agent智能体\文件夹")
            # 1. 查询扩展
            weighted_queries = self.query_expander.expand(query)
        except Exception:
            weighted_queries = [(query, 1.0)]

        try:
            # 2. 混合检索
            docs = self.hybrid_retriever.retrieve_with_query_expansion(weighted_queries)
        except Exception:
            return ""

        if not docs:
            return ""

        try:
            # 3. 三阶段压缩
            compressed_text = self.context_processor.process(
                docs=docs,
                original_query=query,
                expanded_queries=weighted_queries
            )
            return compressed_text
        except Exception:
            return "\n\n---\n".join([d.page_content for d in docs])

    def load_documents(self, folder_path: str = None, web_urls: list = None):
        """加载文档到知识库，并重建 BM25 索引"""
        parent_child_store.load_and_add_documents(
            folder_path=folder_path,
            web_urls=web_urls
        )
        # 重建检索器的 BM25 索引
        new_child_docs = self.mysql_store.get_child_documents()
        self.hybrid_retriever.rebuild_bm25(new_child_docs)
        self.context_processor.build_global_bm25(new_child_docs)


# 模块级实例，供其他模块导入
rag_service = RAGService()