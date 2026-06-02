"""
父子文档存储模块
职责：文档切分、存储到 MySQL 和 Chroma
"""

import uuid
from typing import List
from utils_tool.config_handler import config
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from rag.mysql_store import mysql_store
from utils_tool.file_handler import document_loader
from langchain_huggingface import HuggingFaceEmbeddings



class ParentChildStore:
    """
    父子文档存储

    流程：
    1. 父文档大块切分 → 存 MySQL
    2. 子文档小块切分 → 存 Chroma
    3. 子文档 metadata 记录 parent_id 关联
    """

    def __init__(
        self,
        vectorstore: Chroma = None,
        mysql_store = mysql_store,
        cfg=None
    ):
        self.config = cfg or config

        if vectorstore is None:
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
            self.vectorstore = vectorstore

        self.mysql_store = mysql_store

        # 存储所有子文档（用于后续构建 BM25）
        self.child_documents: List[Document] = []

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.PARENT_CHUNK_SIZE,
            chunk_overlap=self.config.PARENT_CHUNK_OVERLAP,
            separators=self.config.SEPARATORS
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHILD_CHUNK_SIZE,
            chunk_overlap=self.config.CHILD_CHUNK_OVERLAP,
            separators=self.config.SEPARATORS
        )

    def _generate_id(self) -> str:
        return f"parent_id_{uuid.uuid4()}"

    def add_documents(self, documents: List[Document]):
        print("进入文档切分成-->>父子文档阶段")
        """添加文档"""
        parent_docs_with_ids = []
        all_child_docs = []

        for doc in documents:
            source = doc.metadata.get("source", "unknown")

            # 1. 切分父文档（大块）
            print("开始切分父文档--子文档")
            parent_texts = self.parent_splitter.split_text(doc.page_content)

            for parent_idx, parent_text in enumerate(parent_texts):
                parent_id = self._generate_id()

                parent_doc = Document(
                    page_content=parent_text,
                    metadata={
                        **doc.metadata.copy(),
                        "doc_id": parent_id,
                        "doc_type": "parent",
                        "parent_index": parent_idx,
                        "source": source
                    }
                )
                parent_docs_with_ids.append((parent_id, parent_doc))

                # 2. 切分子文档（小块）
                child_texts = self.child_splitter.split_text(parent_text)

                for child_idx, child_text in enumerate(child_texts):
                    child_doc = Document(
                        page_content=child_text,
                        metadata={
                            **doc.metadata.copy(),
                            "parent_id": parent_id,
                            "child_index": child_idx,
                            "doc_type": "child",
                            "source": source
                        }
                    )
                    all_child_docs.append(child_doc)
            print("父子文档切分成功，开始存储父子文档")
        # 3. 赋值后再存储（关键修复：原来错误地使用 self.child_documents 空列表）
        self.child_documents = all_child_docs
        self.mysql_store.save_parent_documents(parent_docs_with_ids)
        self.mysql_store.save_child_documents(all_child_docs)
        try:
            # 获取底层的 chroma client 并删除整个集合
            # client = self.vectorstore._client
            # client.delete_collection("parent_child_docs")
            # print("删除chroma向量数据库中数据已完成")
            # 下次 add_documents 时会自动重新创建该 collection

            self.vectorstore.add_documents(all_child_docs)
            print("子文档添加到向量数据库保存成功")
        except Exception as e:
            raise

        print(f"[存储完成] 父文档: {len(parent_docs_with_ids)} 个, 子文档: {len(all_child_docs)} 个")

    def get_child_documents(self) -> List[Document]:
        """获取所有子文档（用于构建 BM25）"""
        return self.mysql_store.get_child_documents()

    def load_and_add_documents(
        self,
        folder_path: str = None,
        web_urls: list = None,
        allowed_types: tuple = ('.pdf', '.txt', '.docx', '.md')
    ):
        """加载文档并添加到存储"""
        docs = document_loader.load_documents(
            folder_path=folder_path,
            web_urls=web_urls,
            allowed_types=allowed_types
        )

        if not docs:
            # logger.warning("[ParentChildStore] 没有加载到任何文档")
            return

        self.add_documents(docs)
        print(f"[完成] 成功加载并存储 {len(docs)} 个文档")


parent_child_store = ParentChildStore()
