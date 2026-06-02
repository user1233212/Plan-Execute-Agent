"""
三阶段上下文处理器
1. 增强查询粗筛（BM25 + 向量）
2. LLM 二分类精判
3. LLM 提取压缩
"""

import re
from typing import List, Optional
import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


class ThreeStageContextProcessor:
    """三阶段上下文处理器"""

    def __init__(
        self,
        embeddings=None,
        llm: ChatOpenAI = None,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        fusion_top_k: int = 15,
        compress_max_chars: int = 3000
    ):
        print("[ThreeStageContextProcessor] 初始化...")
        self.embeddings = embeddings
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.fusion_top_k = fusion_top_k
        self.compress_max_chars = compress_max_chars
        self.global_bm25_retriever = None

        self.binary_prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template="问题：{query}\n判断以下文档是否相关，只输出数字（0=不相关，1=相关），用逗号分隔：\n{documents}"
        )

        self.compress_prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template="从以下文档提取与问题相关的关键信息，保持原文表述，用\"---\"分隔：\n问题：{query}\n文档：{documents}\n提取结果："
        )
        print("[ThreeStageContextProcessor] 初始化完成")

    def build_global_bm25(self, all_documents: List[Document]):
        """构建全局 BM25 索引"""
        print(f"[build_global_bm25] 开始构建，文档数: {len(all_documents)}")
        if not all_documents:
            print("[build_global_bm25] 文档列表为空，跳过")
            return
        self.global_bm25_retriever = BM25Retriever.from_documents(all_documents, k=self.bm25_top_k * 2)
        print("[build_global_bm25] BM25 索引构建完成")

    def process(
        self,
        docs: List[Document],
        original_query: str,
        expanded_queries: Optional[List[tuple]] = None
    ) -> str:
        """三阶段处理，返回压缩文本"""
        print(f"[process] 开始三阶段处理，输入文档数: {len(docs)}")
        print(f"[process] 原始查询: {original_query}")

        if not docs:
            print("[process] 文档为空，返回空字符串")
            return ""

        enhanced_query = original_query
        if expanded_queries:
            queries = [original_query] + [q for q, _ in expanded_queries if q != original_query]
            enhanced_query = " ".join(queries)
            print(f"[process] 增强查询: {enhanced_query[:100]}...")

        # 阶段1: BM25 + 向量融合
        print("[process] 阶段1: BM25 + 向量融合")
        bm25_docs = self._bm25_filter(docs, original_query) if self.global_bm25_retriever else docs[:self.bm25_top_k]
        print(f"[process] BM25 过滤后文档数: {len(bm25_docs)}")
        vector_docs = self._vector_filter(docs, enhanced_query)
        print(f"[process] 向量过滤后文档数: {len(vector_docs)}")
        docs = self._rrf_fusion(bm25_docs, vector_docs)
        print(f"[process] RRF 融合后文档数: {len(docs)}")

        # 阶段2: LLM 二分类
        print("[process] 阶段2: LLM 二分类")
        docs = self._llm_binary_filter(docs, original_query)
        print(f"[process] LLM 二分类后文档数: {len(docs)}")

        # 阶段3: LLM 提取压缩
        print("[process] 阶段3: LLM 提取压缩")
        result = self._extract_compress(docs, original_query)
        print(f"[process] 压缩完成，输出长度: {len(result)}")
        return result

    def _bm25_filter(self, docs: List[Document], query: str) -> List[Document]:
        print(f"[_bm25_filter] 输入文档数: {len(docs)}")
        retriever = BM25Retriever.from_documents(docs, k=min(self.bm25_top_k, len(docs)))
        result = retriever.invoke(query)
        print(f"[_bm25_filter] 输出文档数: {len(result)}")
        return result

    def _vector_filter(self, docs: List[Document], query: str) -> List[Document]:
        print(f"[_vector_filter] 输入文档数: {len(docs)}")
        if not self.embeddings or len(docs) <= self.vector_top_k:
            print(f"[_vector_filter] 跳过向量过滤 (embeddings={self.embeddings is not None}, docs_count={len(docs)} <= {self.vector_top_k})")
            return docs

        try:
            query_vec = self.embeddings.embed_query(query)
            doc_vecs = self.embeddings.embed_documents([d.page_content for d in docs])
            scores = [
                np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)
                for v in doc_vecs
            ]
            scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            result = [d for d, _ in scored[:self.vector_top_k]]
            print(f"[_vector_filter] 输出文档数: {len(result)}")
            return result
        except Exception as e:
            print(f"[_vector_filter] 向量过滤失败: {e}")
            return docs

    def _rrf_fusion(self, bm25_docs: List[Document], vector_docs: List[Document]) -> List[Document]:
        print(f"[_rrf_fusion] BM25文档数: {len(bm25_docs)}, 向量文档数: {len(vector_docs)}")
        scores = {}
        doc_map = {}
        c = 60

        for doc in bm25_docs + vector_docs:
            doc_id = doc.metadata.get("doc_id", id(doc))
            doc_map[doc_id] = doc

        for rank, doc in enumerate(bm25_docs, 1):
            doc_id = doc.metadata.get("doc_id", id(doc))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + c)
        for rank, doc in enumerate(vector_docs, 1):
            doc_id = doc.metadata.get("doc_id", id(doc))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + c)

        fused = []
        seen = set()
        for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if doc_id not in seen and doc_id in doc_map:
                seen.add(doc_id)
                fused.append(doc_map[doc_id])
                if len(fused) >= self.fusion_top_k:
                    break
        print(f"[_rrf_fusion] 融合后文档数: {len(fused)}")
        return fused

    def _llm_binary_filter(self, docs: List[Document], query: str) -> List[Document]:
        print(f"[_llm_binary_filter] 输入文档数: {len(docs)}")
        if len(docs) <= 1:
            print(f"[_llm_binary_filter] 文档数<=1，跳过二分类")
            return docs

        filtered = []
        for i, doc in enumerate(docs):
            try:
                response = (self.binary_prompt | self.llm | StrOutputParser()).invoke({
                    "query": query,
                    "documents": doc.page_content
                })
                results = re.findall(r'[01]', response)
                is_relevant = results and results[0] == '1'
                print(f"[_llm_binary_filter] 文档 {i+1}/{len(docs)}: {'相关' if is_relevant else '不相关'}")
                if is_relevant:
                    filtered.append(doc)
            except Exception as e:
                print(f"[_llm_binary_filter] 文档 {i+1} 判断失败，默认保留: {e}")
                filtered.append(doc)
        print(f"[_llm_binary_filter] 输出文档数: {len(filtered)}")
        return filtered or docs

    def _extract_compress(self, docs: List[Document], query: str) -> str:
        print(f"[_extract_compress] 输入文档数: {len(docs)}")
        if not docs:
            print("[_extract_compress] 文档为空，返回空字符串")
            return ""

        docs_text = "\n".join([f"[{i}]{d.page_content.replace(chr(10), ' ')}" for i, d in enumerate(docs, 1)])
        print(f"[_extract_compress] 文档文本长度: {len(docs_text)}")

        try:
            compressed = (self.compress_prompt | self.llm | StrOutputParser()).invoke(
                {"query": query, "documents": docs_text}
            )
            if len(compressed) > self.compress_max_chars:
                compressed = compressed[:self.compress_max_chars] + "..."
            print(f"[_extract_compress] 压缩完成，输出长度: {len(compressed)}")
            return compressed
        except Exception as e:
            print(f"[_extract_compress] LLM 压缩失败，降级返回原文: {e}")
            return "\n---\n".join([d.page_content for d in docs])