import unittest
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from langchain_core.documents import Document

from rag_knowledge_base.vector_db_manager import VectorDatabaseManager
from rag_knowledge_base.vector_retriever import VectorRetriever


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.manager = VectorDatabaseManager.__new__(VectorDatabaseManager)
        self.manager.collection_name = "default"
        self.manager.client = Mock()
        self.manager.client.has_collection.return_value = True

    def test_concurrent_searches_keep_their_requested_collections(self):
        both_stores_open = Barrier(2, timeout=5)

        def open_store(name, *, exists):
            both_stores_open.wait()
            return SimpleNamespace(
                search_params={"metric_type": "COSINE"},
                similarity_search_with_score=lambda **_: [(name, 0.9)],
            )

        self.manager._open_store = open_store
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(self.manager.search, "query", collection_name="A")
            second = workers.submit(self.manager.search, "query", collection_name="B")
            self.assertEqual(first.result(timeout=5), [("A", 0.9)])
            self.assertEqual(second.result(timeout=5), [("B", 0.9)])
        self.assertEqual(self.manager.collection_name, "default")

    def test_failed_insert_preserves_existing_collection(self):
        store = Mock()
        store.add_documents.side_effect = ValueError("inconsistent with defined schema")
        self.manager._open_store = Mock(return_value=store)
        with self.assertRaises(ValueError):
            self.manager.add_documents_to_db(["chunk"], "existing")
        self.manager._open_store.assert_called_once_with("existing", exists=True)
        self.manager.client.drop_collection.assert_not_called()

    def test_l2_distance_is_converted_in_the_right_direction(self):
        store = Mock()
        store.search_params = {"metric_type": "L2"}
        store.similarity_search_with_score.return_value = [("near", 0.25), ("far", 4.0)]
        self.manager._open_store = Mock(return_value=store)
        retriever = VectorRetriever(self.manager)
        self.assertEqual(retriever.search_similar_content("question", "existing"), [("near", 0.8)])

    def test_hnsw_search_budget_can_cover_requested_top_k(self):
        store = Mock()
        store.search_params = {"metric_type": "COSINE", "params": {"ef": 10}}
        store.similarity_search_with_score.return_value = [("chunk", 0.8)]
        self.manager._open_store = Mock(return_value=store)
        self.manager.search("question", k=20, collection_name="existing")
        store.similarity_search_with_score.assert_called_once_with(
            query="question", k=20, param={"metric_type": "COSINE", "params": {"ef": 20}},
        )
        self.assertEqual(store.search_params["params"]["ef"], 10)

    def test_collection_clear_uses_explicit_target(self):
        self.manager.clear_database("explicit")
        self.manager.client.drop_collection.assert_called_once_with("explicit")

    def test_milvus_uses_local_file_or_hosted_endpoint(self):
        milvus_client = Mock()
        sdk = SimpleNamespace(MilvusClient=milvus_client)
        with patch.dict(sys.modules, {"pymilvus": sdk}):
            with patch.dict(os.environ, {"MILVUS_URI": "./rag_demo.db"}, clear=True):
                self.manager._connect_to_milvus()
                milvus_client.assert_called_with(uri="./rag_demo.db")
            with patch.dict(os.environ, {
                "MILVUS_URI": "https://milvus.example:19530", "MILVUS_TOKEN": "test-db-token",
            }, clear=True):
                self.manager._connect_to_milvus()
                milvus_client.assert_called_with(uri="https://milvus.example:19530", token="test-db-token")


class ClauseSplitTests(unittest.TestCase):
    def setUp(self):
        self.manager = VectorDatabaseManager.__new__(VectorDatabaseManager)

    def test_numbered_policy_splits_per_clause_and_keeps_metadata(self):
        document = Document(
            page_content="[1] 无理由退货。\n\n[2] 退货运费。\n\n[4] 电子发票。\n\n[5] 包裹破损。",
            metadata={"source": "after-sales-policy.txt"},
        )
        clauses, other = self.manager.split_numbered_clauses([document])
        self.assertEqual([item.metadata["clause_id"] for item in clauses], ["1", "2", "4", "5"])
        self.assertEqual(other, [])
        self.assertEqual(clauses[2].page_content, "电子发票。")
        self.assertEqual(clauses[2].metadata["source"], "after-sales-policy.txt")
        self.assertEqual(clauses[2].metadata["clause_label"], "[4]")

    def test_plain_document_keeps_existing_splitter(self):
        document = Document(
            page_content="普通售后说明，没有条款编号。", metadata={"source": "faq.txt"}
        )
        clauses, other = self.manager.split_numbered_clauses([document])
        self.assertEqual(clauses, [])
        self.assertEqual(other, [document])


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.manager = Mock()
        self.retriever = VectorRetriever(self.manager)
        self.retriever._generate_answer_with_llm = Mock(return_value="answer [1]")

    def test_insufficient_evidence_skips_generation(self):
        self.manager.search.return_value = [(SimpleNamespace(page_content="unrelated"), 0.1)]
        result = self.retriever.answer_question("question", "policies")
        self.assertEqual(result.status, "insufficient_evidence")
        self.retriever._generate_answer_with_llm.assert_not_called()

    def test_retrieval_failure_is_distinct_from_insufficient_evidence(self):
        self.manager.search.side_effect = TimeoutError()
        result = self.retriever.answer_question("question", "policies")
        self.assertEqual(result.status, "retrieval_error")
        self.retriever._generate_answer_with_llm.assert_not_called()

    def test_answer_receives_numbered_references(self):
        document = SimpleNamespace(page_content="policy text")
        self.manager.search.return_value = [(document, 0.8)]
        result = self.retriever.answer_question("question", "policies")
        self.retriever._generate_answer_with_llm.assert_called_once_with("question", "[1] policy text")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.source_documents, [document])


class ModelConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.__enter__.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer [1]"))]
        )
        self.openai = Mock(return_value=self.client)
        self.retriever = VectorRetriever(Mock())

    def test_deepseek_uses_separate_model_credential(self):
        with patch.dict(os.environ, {
            "LLM_BASE_URL": "https://api.deepseek.com",
            "LLM_MODEL": "deepseek-flash",
            "LLM_API_KEY": "test-chat-key",
            "DASHSCOPE_API_KEY": "test-embedding-key",
        }, clear=True), patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=self.openai)}):
            self.assertEqual(self.retriever._generate_answer_with_llm("question", "evidence"), "answer [1]")
        self.openai.assert_called_once_with(api_key="test-chat-key", base_url="https://api.deepseek.com", timeout=30)
        request = self.client.__enter__.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "deepseek-flash")

    def test_dashscope_credentials_are_not_reused_for_third_party_endpoint(self):
        with patch.dict(os.environ, {
            "LLM_BASE_URL": "https://api.deepseek.com", "DASHSCOPE_API_KEY": "test-embedding-key",
        }, clear=True), patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=self.openai)}):
            with self.assertRaisesRegex(ValueError, "LLM_API_KEY"):
                self.retriever._generate_answer_with_llm("question", "evidence")
        self.openai.assert_not_called()

    def test_dashscope_workspace_can_reuse_its_own_key(self):
        base_url = "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        with patch.dict(os.environ, {
            "LLM_BASE_URL": base_url, "DASHSCOPE_API_KEY": "test-bailian-key",
        }, clear=True), patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=self.openai)}):
            self.retriever._generate_answer_with_llm("question", "evidence")
        self.openai.assert_called_once_with(api_key="test-bailian-key", base_url=base_url, timeout=30)


if __name__ == "__main__":
    unittest.main()
