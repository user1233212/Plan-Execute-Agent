"""
混合检索器：BM25 + 向量检索 + RRF 融合

核心逻辑：
1. BM25 检索子文档 → 按 parent_id 聚合取平均值 → 父文档排名
2. 向量检索子文档 → 按 parent_id 聚合取最高值 → 父文档排名
3. RRF 算法融合两个排名 → 最终父文档列表
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from rag.mysql_store import mysql_store
from langchain_huggingface import HuggingFaceEmbeddings
from utils_tool.config_handler import config


@dataclass
class HybridConfig:
    """混合检索配置"""
    search_k: int = 4
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    rrf_c: int = 60


class HybridRetriever:
    """
    混合检索器

    差异化聚合策略：
    - BM25: 关键词匹配，多个子文档匹配说明覆盖全面 → 取平均值
    - 向量: 语义匹配，一个子文档高度相关就值得召回 → 取最高值
    """

    def __init__(
            self,
            vectorstore: Chroma = None,
            child_documents: List[Document] = None,
            mysql_store = mysql_store,
            cfg=None
    ):
        print("[HybridRetriever] 初始化开始...")
        self.config = cfg or config
        if vectorstore is None:
            print("[HybridRetriever] 创建新的向量存储...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.MODEL_NAME,
                encode_kwargs={'normalize_embeddings': True}
            )
            self.vectorstore = Chroma(
                collection_name='parent_child_docs',
                embedding_function=self.embeddings,
                persist_directory=self.config.CHROMA_PERSIST_DIR
            )
        else:
            print("[HybridRetriever] 使用已有的向量存储")
            self.vectorstore = vectorstore

        self.mysql_store = mysql_store

        # 延迟加载子文档：只在未传入时才查数据库
        if child_documents is None:
            try:
                print("[HybridRetriever] 从 MySQL 加载子文档...")
                self.child_documents = self.mysql_store.get_child_documents()
                print(f"[HybridRetriever] 加载了 {len(self.child_documents)} 个子文档")
            except Exception as e:
                print(f"[HybridRetriever] 加载子文档失败: {e}")
                self.child_documents = []
        else:
            print(f"[HybridRetriever] 使用传入的子文档，共 {len(child_documents)} 个")
            self.child_documents = child_documents

        # 构建 BM25 检索器（需要有文档）
        if self.child_documents:
            print("[HybridRetriever] 构建 BM25 检索器...")
            self.bm25_retriever = BM25Retriever.from_documents(
                self.child_documents,
                k=self.config.SEARCH_K * 3
            )
            print("[HybridRetriever] BM25 检索器构建完成")
        else:
            print("[HybridRetriever] 子文档为空，BM25 检索器未构建")
            self.bm25_retriever = None

        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.SEARCH_K * 3}
        )
        print("[HybridRetriever] 初始化完成")

    def rebuild_bm25(self, child_documents: List[Document]):
        """重建 BM25 索引（新增文档后调用）"""
        print(f"[rebuild_bm25] 重建 BM25，子文档数: {len(child_documents)}")
        self.child_documents = child_documents
        if child_documents:
            self.bm25_retriever = BM25Retriever.from_documents(
                child_documents,
                k=self.config.SEARCH_K * 3
            )
            print("[rebuild_bm25] BM25 重建完成")
        else:
            print("[rebuild_bm25] 子文档为空，跳过")

    # ==================== BM25 检索：取平均值 ====================
    def get_parent_ranking_by_bm25(self, query: str):
        """BM25检索 + RRF(1/(rank+c))，按 parent_id 分组取平均值，返回父文档分数"""
        if not self.bm25_retriever:
            print("[get_parent_ranking_by_bm25] BM25 检索器未初始化")
            return {}

        try:
            print(f"[get_parent_ranking_by_bm25] 执行 BM25 检索: {query}")
            bm25_docs = self.bm25_retriever.invoke(query)
            print(f"[get_parent_ranking_by_bm25] BM25 检索到 {len(bm25_docs)} 个文档")
            c = 60
            parent_scores_list: Dict[str, List[float]] = {}

            for rank, doc in enumerate(bm25_docs, 1):
                parent_id = doc.metadata.get("parent_id")
                if parent_id:
                    score = 1.0 / (rank + c)
                    parent_scores_list.setdefault(parent_id, []).append(score)

            parent_avg = {
                pid: sum(scores) / len(scores)
                for pid, scores in parent_scores_list.items()
            }
            print(f"[get_parent_ranking_by_bm25] BM25 排名: {len(parent_avg)} 个父文档")
            return parent_avg

        except Exception as e:
            print(f"[get_parent_ranking_by_bm25] BM25 检索失败: {e}")
            return {}

    def _bm25_get_ranking(self, query: str) -> List[str]:
        """返回父文档排名列表（按分数降序）"""
        scores = self.get_parent_ranking_by_bm25(query)
        ranking = [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        print(f"[_bm25_get_ranking] BM25 排名列表: {ranking[:5]}...")
        return ranking


    # ==================== 向量检索：取最高值 ====================

    def _vector_get_parent_scores(self, query: str) -> Dict[str, float]:
        """向量检索 → 按 parent_id 聚合取最高值"""
        try:
            print(f"[_vector_get_parent_scores] 执行向量检索: {query}")
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=self.config.SEARCH_K)
            print(f"[_vector_get_parent_scores] 向量检索到 {len(results)} 个结果")
        except Exception as e:
            print(f"[_vector_get_parent_scores] 向量检索失败: {e}")
            return {}

        parent_max_scores = {}
        for doc, score in results:
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                if parent_id not in parent_max_scores or score > parent_max_scores[parent_id]:
                    parent_max_scores[parent_id] = score

        print(f"[_vector_get_parent_scores] 向量排名: {len(parent_max_scores)} 个父文档")
        return parent_max_scores

    def _vector_get_ranking(self, query: str) -> List[str]:
        """向量父文档排名"""
        scores = self._vector_get_parent_scores(query)
        ranking = [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        print(f"[_vector_get_ranking] 向量排名列表: {ranking[:5]}...")
        return ranking

    # ==================== RRF 融合 ====================

    def _rrf_fusion(self, bm25_ranking: List[str], vector_ranking: List[str]) -> Dict[str, float]:
        """RRF 融合: score = Σ weight / (rank + c)"""
        print(f"[_rrf_fusion] 开始 RRF 融合，BM25排名数: {len(bm25_ranking)}, 向量排名数: {len(vector_ranking)}")
        rrf_scores: Dict[str, float] = {}
        c = self.config.RRF_C

        for rank, pid in enumerate(bm25_ranking, start=1):
            rrf_scores[pid] = rrf_scores.get(pid, 0) + self.config.BM25_WEIGHT / (rank + c)

        for rank, pid in enumerate(vector_ranking, start=1):
            rrf_scores[pid] = rrf_scores.get(pid, 0) + self.config.VECTOR_WEIGHT / (rank + c)

        print(f"[_rrf_fusion] RRF 融合结果: {len(rrf_scores)} 个父文档")
        return rrf_scores

    # ==================== 公开接口 ====================

    def retrieve(self, query: str) -> List[Document]:
        """检索（不带分数）"""
        print(f"[retrieve] 检索查询: {query}")
        docs, _ = self.retrieve_with_scores(query)
        print(f"[retrieve] 返回 {len(docs)} 个文档")
        return docs

    def retrieve_with_scores(self, query: str) -> Tuple[List[Document], List[float]]:
        """混合检索主流程"""
        print(f"[retrieve_with_scores] 开始混合检索: {query}")
        bm25_ranking = self._bm25_get_ranking(query)
        vector_ranking = self._vector_get_ranking(query)

        rrf_scores = self._rrf_fusion(bm25_ranking, vector_ranking)
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"[retrieve_with_scores] 排序后前5名: {sorted_items[:5]}")
        top_pids = [pid for pid, _ in sorted_items[:self.config.SEARCH_K]]
        print(f"[retrieve_with_scores] 选取 top {self.config.SEARCH_K} 个父文档ID: {top_pids}")

        parent_docs = self.mysql_store.get_parent_documents(top_pids)
        print(f"[retrieve_with_scores] 从 MySQL 获取到 {len(parent_docs)} 个父文档")
        doc_map = {doc.metadata.get("doc_id"): doc for doc in parent_docs}
        ordered_docs = [doc_map[pid] for pid in top_pids if pid in doc_map]
        ordered_scores = [rrf_scores[pid] for pid in top_pids if pid in doc_map]

        print(f"[retrieve_with_scores] 最终返回 {len(ordered_docs)} 个文档")
        return ordered_docs, ordered_scores

    def retrieve_with_query_expansion(
            self,
            weighted_queries: List[tuple]
    ) -> List[Document]:
        """带权重的多查询 RRF 融合"""
        print(f"[retrieve_with_query_expansion] 开始带权重的多查询检索，查询数: {len(weighted_queries)}")
        if not weighted_queries:
            print("[retrieve_with_query_expansion] 查询列表为空")
            return []

        final_scores: Dict[str, float] = {}
        c = self.config.RRF_C

        for query, q_weight in weighted_queries:
            print(f"[retrieve_with_query_expansion] 处理查询: {query[:50]}..., 权重: {q_weight}")
            try:
                docs = self.retrieve(query)
                print(f"[retrieve_with_query_expansion] 查询 '{query[:30]}...' 检索到 {len(docs)} 个文档")
            except Exception as e:
                print(f"[retrieve_with_query_expansion] 查询 '{query}' 检索失败: {e}")
                continue

            for rank, doc in enumerate(docs, start=1):
                pid = doc.metadata.get("doc_id")
                if pid:
                    final_scores[pid] = final_scores.get(pid, 0) + q_weight / (rank + c)

        print(f"[retrieve_with_query_expansion] 最终得分: {len(final_scores)} 个父文档")
        sorted_pids = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:self.config.SEARCH_K]
        print(f"[retrieve_with_query_expansion] top {self.config.SEARCH_K}: {sorted_pids}")
        top_pids = [pid for pid, _ in sorted_pids]

        parent_docs = self.mysql_store.get_parent_documents(top_pids)
        doc_map = {doc.metadata.get("doc_id"): doc for doc in parent_docs}
        result = [doc_map[pid] for pid in top_pids if pid in doc_map]
        print(f"[retrieve_with_query_expansion] 最终返回 {len(result)} 个文档")
        return result


hybrid_retriever = HybridRetriever()