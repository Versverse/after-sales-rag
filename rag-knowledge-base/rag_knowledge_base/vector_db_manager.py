"""Document ingestion and collection-scoped Milvus access."""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import load_dotenv

# Load env vars from explicit path
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# LangChain and Milvus imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, 
    CSVLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredExcelLoader
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """向量数据库管理器 (Milvus后端)"""
    
    def __init__(self, 
                 milvus_host: str = None,
                 milvus_port: int = None,
                 collection_name: str = None,
                 embedding_model: str = None,
                 dashscope_api_key: str = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        初始化向量数据库管理器
        
        Args:
            milvus_host: Milvus 服务主机
            milvus_port: Milvus 服务端口
            collection_name: Milvus 集合名称
            embedding_model: DashScope嵌入模型名称
            dashscope_api_key: DashScope API密钥
            chunk_size: 文档切分块大小
            chunk_overlap: 文档切分重叠大小
        """
        self.milvus_host = milvus_host or os.getenv("MILVUS_HOST", "127.0.0.1")
        # Ensure port is a string as pymilvus might expect it, or handle int gracefully
        self.milvus_port = str(milvus_port or os.getenv("MILVUS_PORT", "19530"))
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "agent_rag")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化嵌入模型
        self._init_embeddings()
        
        # 初始化文档切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 连接到Milvus
        self._connect_to_milvus()
        
    def _init_embeddings(self):
        # The same embedding model must be used for indexing and querying.
        provider = os.getenv("EMBEDDING_PROVIDER", "dashscope").lower()
        if provider == "local":
            self.embeddings = HuggingFaceEmbeddings(
                model_name=os.getenv(
                    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                model_kwargs={"device": "cpu"},
            )
        elif provider == "dashscope":
            if not self.dashscope_api_key:
                raise ValueError("DASHSCOPE_API_KEY is required for DashScope embeddings")
            self.embeddings = DashScopeEmbeddings(
                model=self.embedding_model, dashscope_api_key=self.dashscope_api_key
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _connect_to_milvus(self):
        from pymilvus import MilvusClient

        # pymilvus 3.x parses MILVUS_URI when imported and rejects local paths there,
        # so allow a dedicated local database path while keeping hosted MILVUS_URI support.
        self.milvus_uri = (
            os.getenv("MILVUS_DB_PATH")
            or os.getenv("MILVUS_URI")
            or f"http://{self.milvus_host}:{self.milvus_port}"
        )
        self.connection_args = {"uri": self.milvus_uri}
        token = os.getenv("MILVUS_TOKEN")
        if token:
            self.connection_args["token"] = token
        self.client = MilvusClient(**self.connection_args)

    def _open_store(self, collection_name: str, *, exists: bool):
        from langchain_milvus import Milvus

        if exists:
            schema = self.client.describe_collection(collection_name)
            fields = schema["fields"]
            field_names = {field["name"] for field in fields}
            settings = {
                "auto_id": schema["auto_id"],
                "enable_dynamic_field": schema.get("enable_dynamic_field", False),
                "primary_field": next(field["name"] for field in fields if field.get("is_primary")),
                "vector_field": "embedding" if "embedding" in field_names and "vector" not in field_names else "vector",
            }
        else:
            settings = {
                "auto_id": True,
                "enable_dynamic_field": True,
                "index_params": {
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 64},
                },
            }
        return Milvus(
            embedding_function=self.embeddings,
            collection_name=collection_name,
            connection_args=self.connection_args.copy(),
            consistency_level="Strong",
            drop_old=False,
            **settings,
        )

    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        # ... (代码与之前版本相同)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.csv':
                loader = CSVLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
                logger.warning(f"未识别的文件类型 {file_extension}, 使用文本加载器")
            
            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 个文档块")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            return []

    CLAUSE_PATTERN = re.compile(r"(?m)^[ \t]*\[(\d+)\][ \t]*")

    def split_numbered_clauses(
        self, documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """Split documents whose lines start with [n] into one chunk per clause.

        Returns the clause chunks and the documents that have no clause markers
        so callers can keep the existing splitter for ordinary content.
        """
        clause_documents: List[Document] = []
        other_documents: List[Document] = []
        for document in documents:
            matches = list(self.CLAUSE_PATTERN.finditer(document.page_content))
            if not matches:
                other_documents.append(document)
                continue

            for index, match in enumerate(matches):
                start = match.end()
                end = (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(document.page_content)
                )
                content = document.page_content[start:end].strip()
                if not content:
                    continue
                metadata = dict(document.metadata)
                metadata["clause_id"] = match.group(1)
                metadata["clause_label"] = f"[{match.group(1)}]"
                clause_documents.append(Document(page_content=content, metadata=metadata))

        return clause_documents, other_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        # ... (代码与之前版本相同)
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档切分完成: {len(documents)} -> {len(split_docs)} 个块")
            return split_docs
        except Exception as e:
            logger.error(f"文档切分失败: {e}")
            return documents

    def add_documents_to_db(self, documents: List[Document], collection_name: str = None):
        """Append documents. An incompatible write must never replace a collection."""
        if not documents:
            return
        target = collection_name or self.collection_name
        store = self._open_store(target, exists=self.client.has_collection(target))
        store.add_documents(documents)
        logger.info("Added %s chunks to collection %s", len(documents), target)

    def process_file(
        self, file_path: str, collection_name: str = None, *, source_name: str | None = None
    ) -> bool:
        """
        处理单个文件：加载、切分、存储
        
        Args:
            file_path: 文件路径
            collection_name: 集合名称
            
        Returns:
            处理是否成功
        """
        try:
            logger.info(f"开始处理文件: {file_path}")
            documents = self.load_document(file_path)
            if not documents:
                return False

            for document in documents:
                document.metadata["source"] = source_name or Path(file_path).name
            clause_docs, other_docs = self.split_numbered_clauses(documents)
            split_docs = list(clause_docs)
            if other_docs:
                split_docs.extend(self.split_documents(other_docs))
            if not split_docs:
                return False
            logger.info(
                "文档切分完成: %s 个条款块, %s 个普通块",
                len(clause_docs),
                len(split_docs) - len(clause_docs),
            )
            self.add_documents_to_db(split_docs, collection_name)
            
            logger.info(f"文件处理完成: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return False

    def process_csv_data(self, csv_path: str,
                         text_columns: List[str] = None,
                         metadata_columns: List[str] = None) -> bool:
        """
        处理CSV数据文件
        
        Args:
            csv_path: CSV文件路径
            text_columns: 需要向量化的文本列名列表
            
        Returns:
            处理是否成功
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"读取CSV文件: {csv_path}, 共 {len(df)} 行数据")
            if text_columns is None or not text_columns:
                object_cols = [c for c in df.columns if df[c].dtype == 'object']
                text_columns = [c for c in object_cols if not str(c).startswith('Unnamed')]

            if metadata_columns is None:
                metadata_columns = [c for c in df.columns if c not in text_columns]

            documents = []
            for idx, row in df.iterrows():
                content_parts = []
                metadata = {"source": csv_path, "row_index": idx}
                
                for col in text_columns:
                    if pd.notna(row[col]):
                        text = str(row[col]).strip()
                        if text:
                            content_parts.append(f"{col}: {text}")

                for col in metadata_columns:
                    val = row.get(col)
                    if pd.notna(val):
                        metadata[str(col)] = str(val)
                
                if content_parts:
                    content = "\n".join(content_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            
            logger.info(f"构建了 {len(documents)} 个文档")
            split_docs = self.split_documents(documents)
            self.add_documents_to_db(split_docs)
            
            return True
            
        except Exception as e:
            logger.error(f"处理CSV数据失败: {e}")
            return False

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """使用嵌入模型为一组文本生成嵌入向量"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return []

    def search(
        self, query: str, k: int = 5, filter_dict: Dict = None,
        collection_name: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Search one collection and return scores ordered from most to least similar."""
        target = collection_name or self.collection_name
        if not self.client.has_collection(target):
            return []

        store = self._open_store(target, exists=True)
        if filter_dict:
            logger.warning("Dictionary metadata filters are not supported")
        search_params = store.search_params or {}
        options = search_params.get("params", {})
        if "ef" in options and options["ef"] < k:
            search_params = {**search_params, "params": {**options, "ef": k}}
        results = store.similarity_search_with_score(query=query, k=k, param=search_params or None)
        if not results:
            return []

        metric = search_params.get("metric_type")
        if metric == "L2":
            # Milvus L2 is a distance. Convert it to a score where higher is closer.
            return [(doc, 1.0 / (1.0 + distance)) for doc, distance in results]
        if metric == "COSINE":
            return [(doc, float(score)) for doc, score in results]
        raise ValueError(f"Unsupported index metric for evidence threshold: {metric!r}")

    def get_database_info(self, collection_name: str = None) -> Dict[str, Any]:
        target = collection_name or self.collection_name
        exists = self.client.has_collection(target)
        count = int(self.client.get_collection_stats(target)["row_count"]) if exists else 0
        address = urlsplit(self.milvus_uri)
        return {
            "milvus_host": address.hostname,
            "milvus_port": str(address.port) if address.port is not None else None,
            "collection_name": target,
            "is_initialized": exists,
            "document_count": count,
        }

    def clear_database(self, collection_name: str = None):
        """Drop exactly the explicitly requested collection, when it exists."""
        target = collection_name or self.collection_name
        if self.client.has_collection(target):
            self.client.drop_collection(target)


def main():
    db_manager = VectorDatabaseManager()
    print(json.dumps(db_manager.get_database_info(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
