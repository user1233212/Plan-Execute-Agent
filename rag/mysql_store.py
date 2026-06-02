"""
MySQL 文档存储模块
职责：父文档与子文档的存储与读取
"""

from typing import List, Tuple
from sqlalchemy import create_engine, Column, String, Text, JSON, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from langchain_core.documents import Document
from utils_tool.config_handler import config

Base = declarative_base()


class ParentDocumentModel(Base):
    """父文档表结构"""
    __tablename__ = "parent_documents"

    doc_id = Column(String(64), primary_key=True)
    page_content = Column(Text, nullable=False)
    doc_metadata = Column(JSON, nullable=True)


class ChildDocumentModel(Base):
    """子文档表结构"""
    __tablename__ = "child_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parent_id = Column(String(64), nullable=False, index=True)
    page_content = Column(Text, nullable=False)
    child_index = Column(Integer, nullable=False)
    doc_metadata = Column(JSON, nullable=True)


class MySQLStore:
    """MySQL 文档存储"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        database: str = None
    ):
        host = host or config.MYSQL_HOST
        port = port or config.MYSQL_PORT
        user = user or config.MYSQL_USER
        password = password or config.MYSQL_PASSWORD
        database = database or config.MYSQL_DATABASE

        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        print(connection_string)
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def save_parent_documents(self, docs_with_ids: List[Tuple[str, Document]]):
        """批量保存父文档"""
        try:
            for doc_id, doc in docs_with_ids:
                model = ParentDocumentModel(
                    doc_id=doc_id,
                    page_content=doc.page_content,
                    doc_metadata=doc.metadata
                )
                self.session.merge(model)
            self.session.commit()
            print("父文档保存成功")
        except Exception as e:
            self.session.rollback()
            raise

    def get_parent_documents(self, doc_ids: List[str]) -> List[Document]:
        """批量获取父文档"""
        try:
            models = self.session.query(ParentDocumentModel).filter(
                ParentDocumentModel.doc_id.in_(doc_ids)
            ).all()
            return [
                Document(page_content=m.page_content, metadata=m.doc_metadata)
                for m in models
            ]
        except Exception as e:
            self.session.rollback()
            return []

    def save_child_documents(self, child_docs: List[Document]):
        """批量保存子文档"""
        try:
            for doc in child_docs:
                model = ChildDocumentModel(
                    parent_id=doc.metadata.get("parent_id"),
                    page_content=doc.page_content,
                    child_index=doc.metadata.get("child_index", 0),
                    doc_metadata=doc.metadata
                )
                self.session.merge(model)
            self.session.commit()
            print("子文档保存成功")
        except Exception as e:
            self.session.rollback()
            raise

    def get_child_documents(self) -> List[Document]:
        """获取所有子文档"""
        try:
            models = self.session.query(ChildDocumentModel).all()
            return [
                Document(page_content=m.page_content, metadata=m.doc_metadata or {})
                for m in models
            ]
        except Exception:
            return []

    def close(self):
        """关闭 session"""
        self.session.close()


mysql_store = MySQLStore()