"""
End-to-end integration tests for the vector database system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

from langchain_vector_db.manager import VectorDatabaseManager
from langchain_vector_db.models.config import VectorDBConfig, SecurityConfig, ObservabilityConfig
from langchain_vector_db.models.document import Document
from langchain_vector_db.models.auth import AuthToken
from langchain_vector_db.exceptions import VectorDBException


class TestEndToEndWorkflows:
    """End-to-end integration tests for complete workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path=self.temp_dir,
            chunk_size=500,
            chunk_overlap=50,
            security=SecurityConfig(
                auth_enabled=True,
                auth_type="api_key",
                rate_limiting_enabled=True,
                max_requests_per_minute=100
            ),
            observability=ObservabilityConfig(
                log_level="DEBUG",
                metrics_enabled=True,
                tracing_enabled=True
            )
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_complete_document_ingestion_workflow(self, mock_embeddings):
        """Test complete document ingestion workflow."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4] for _ in range(3)
        ]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        # Create manager
        manager = VectorDatabaseManager(self.config)
        
        # Create test documents
        documents = [
            Document(
                page_content="This is the first test document about machine learning.",
                metadata={"source": "test1.txt", "category": "ml"}
            ),
            Document(
                page_content="This is the second test document about artificial intelligence.",
                metadata={"source": "test2.txt", "category": "ai"}
            ),
            Document(
                page_content="This is the third test document about deep learning.",
                metadata={"source": "test3.txt", "category": "dl"}
            )
        ]
        
        # Test document ingestion
        doc_ids = manager.add_documents(documents)
        
        assert len(doc_ids) == 3
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)
        
        # Verify documents were stored
        for doc_id in doc_ids:
            retrieved_doc = manager.get_document(doc_id)
            assert retrieved_doc is not None
            assert retrieved_doc.page_content in [doc.page_content for doc in documents]
        
        # Test similarity search
        results = manager.similarity_search("machine learning", k=2)
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
        
        # Test similarity search with scores
        results_with_scores = manager.similarity_search_with_score("artificial intelligence", k=3)
        assert len(results_with_scores) <= 3
        assert all(isinstance(doc, Document) and isinstance(score, float) 
                  for doc, score in results_with_scores)
        
        # Test persistence
        assert manager.persist() is True
        
        # Test loading
        new_manager = VectorDatabaseManager(self.config)
        assert new_manager.load() is True
        
        # Verify data persisted correctly
        loaded_results = new_manager.similarity_search("deep learning", k=1)
        assert len(loaded_results) >= 1
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_document_crud_operations(self, mock_embeddings):
        """Test complete CRUD operations on documents."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        
        # Create
        document = Document(
            page_content="Original content for CRUD testing.",
            metadata={"source": "crud_test.txt", "version": 1}
        )
        doc_ids = manager.add_documents([document])
        doc_id = doc_ids[0]
        
        # Read
        retrieved_doc = manager.get_document(doc_id)
        assert retrieved_doc is not None
        assert retrieved_doc.page_content == document.page_content
        assert retrieved_doc.metadata["version"] == 1
        
        # Update
        updated_document = Document(
            page_content="Updated content for CRUD testing.",
            metadata={"source": "crud_test.txt", "version": 2}
        )
        update_result = manager.update_document(doc_id, updated_document)
        assert update_result is True
        
        # Verify update
        updated_retrieved = manager.get_document(doc_id)
        assert updated_retrieved.page_content == updated_document.page_content
        assert updated_retrieved.metadata["version"] == 2
        
        # Delete
        delete_result = manager.delete_documents([doc_id])
        assert delete_result is True
        
        # Verify deletion
        deleted_doc = manager.get_document(doc_id)
        assert deleted_doc is None
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_batch_operations_performance(self, mock_embeddings):
        """Test performance of batch operations."""
        # Mock embedding service for batch operations
        batch_size = 50
        mock_embeddings.return_value.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4] for _ in range(batch_size)
        ]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        
        # Create batch of documents
        documents = []
        for i in range(batch_size):
            documents.append(Document(
                page_content=f"This is test document number {i} for batch testing.",
                metadata={"source": f"batch_test_{i}.txt", "batch_id": "test_batch"}
            ))
        
        # Test batch ingestion
        import time
        start_time = time.time()
        doc_ids = manager.add_documents(documents)
        ingestion_time = time.time() - start_time
        
        assert len(doc_ids) == batch_size
        assert ingestion_time < 10.0  # Should complete within 10 seconds
        
        # Test batch search performance
        start_time = time.time()
        results = manager.similarity_search("test document", k=10)
        search_time = time.time() - start_time
        
        assert len(results) <= 10
        assert search_time < 2.0  # Should complete within 2 seconds
        
        # Test batch deletion
        start_time = time.time()
        delete_result = manager.delete_documents(doc_ids)
        deletion_time = time.time() - start_time
        
        assert delete_result is True
        assert deletion_time < 5.0  # Should complete within 5 seconds
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_error_handling_and_recovery(self, mock_embeddings):
        """Test error handling and recovery scenarios."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        
        # Test invalid document handling
        with pytest.raises(VectorDBException):
            manager.add_documents([])  # Empty list should be handled gracefully
        
        # Test invalid search parameters
        results = manager.similarity_search("", k=0)  # Empty query, k=0
        assert results == []
        
        # Test non-existent document retrieval
        non_existent_doc = manager.get_document("non_existent_id")
        assert non_existent_doc is None
        
        # Test deletion of non-existent documents
        delete_result = manager.delete_documents(["non_existent_id"])
        assert delete_result is True  # Should handle gracefully
        
        # Test recovery after error
        valid_document = Document(
            page_content="Valid document after error recovery.",
            metadata={"source": "recovery_test.txt"}
        )
        doc_ids = manager.add_documents([valid_document])
        assert len(doc_ids) == 1
        
        # Verify system is still functional
        results = manager.similarity_search("recovery", k=1)
        assert len(results) >= 1
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_security_integration(self, mock_embeddings):
        """Test security integration in end-to-end workflows."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        
        # Create auth token
        auth_token = AuthToken(
            user_id="test_user",
            roles=["writer"],
            permissions=["documents.create", "documents.read", "search.query"],
            expires_at=None,
            correlation_id="test_correlation"
        )
        
        # Test authenticated operations
        document = Document(
            page_content="Secure document content.",
            metadata={"source": "secure_test.txt", "classification": "internal"}
        )
        
        doc_ids = manager.add_documents(
            [document],
            auth_token=auth_token,
            user_id="test_user",
            ip_address="192.168.1.1"
        )
        assert len(doc_ids) == 1
        
        # Test authenticated search
        results = manager.similarity_search(
            "secure document",
            k=1,
            auth_token=auth_token,
            user_id="test_user",
            ip_address="192.168.1.1"
        )
        assert len(results) >= 1
        
        # Test unauthorized access (no token)
        with pytest.raises(VectorDBException):
            manager.add_documents([document])  # Should fail without auth token
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_observability_integration(self, mock_embeddings):
        """Test observability integration in workflows."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        
        # Perform operations to generate observability data
        document = Document(
            page_content="Observability test document.",
            metadata={"source": "observability_test.txt"}
        )
        
        doc_ids = manager.add_documents([document])
        results = manager.similarity_search("observability", k=1)
        
        # Check that observability manager is working
        if manager.observability_manager:
            # Check logs
            recent_logs = manager.observability_manager.get_recent_logs(10)
            assert len(recent_logs) > 0
            
            # Check metrics
            system_metrics = manager.observability_manager.get_system_metrics()
            assert system_metrics is not None
            assert system_metrics.documents_indexed > 0
            
            # Check traces
            traces = manager.observability_manager.get_traces(10)
            assert len(traces) > 0
            
            # Check health status
            health_status = manager.observability_manager.get_comprehensive_health_status()
            assert health_status["overall_health"] in ["healthy", "degraded"]
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_configuration_validation_and_switching(self, mock_embeddings):
        """Test configuration validation and backend switching."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        
        # Test configuration validation
        validation_result = manager.validate_configuration()
        assert isinstance(validation_result, dict)
        assert "valid" in validation_result
        
        # Test available backends
        backends = manager.get_available_backends()
        assert "local" in backends
        assert "s3" in backends
        
        # Add some test data
        document = Document(
            page_content="Configuration test document.",
            metadata={"source": "config_test.txt"}
        )
        doc_ids = manager.add_documents([document])
        
        # Test system info
        system_info = manager.get_system_info()
        assert isinstance(system_info, dict)
        assert "storage_type" in system_info
        assert "vector_count" in system_info
        assert system_info["storage_type"] == "local"
        assert system_info["vector_count"] >= 1
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_persistence_and_data_integrity(self, mock_embeddings):
        """Test persistence and data integrity across sessions."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4] for _ in range(5)
        ]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        # First session - create and persist data
        manager1 = VectorDatabaseManager(self.config)
        
        documents = [
            Document(
                page_content=f"Persistence test document {i}.",
                metadata={"source": f"persist_test_{i}.txt", "session": 1}
            )
            for i in range(5)
        ]
        
        doc_ids = manager1.add_documents(documents)
        assert len(doc_ids) == 5
        
        # Persist data
        persist_result = manager1.persist()
        assert persist_result is True
        
        # Get persistence info
        persistence_info = manager1.get_persistence_info()
        assert isinstance(persistence_info, dict)
        assert "last_persisted" in persistence_info
        
        # Validate data integrity
        integrity_result = manager1.validate_data_integrity()
        assert isinstance(integrity_result, dict)
        assert integrity_result.get("valid", False) is True
        
        # Second session - load and verify data
        manager2 = VectorDatabaseManager(self.config)
        load_result = manager2.load()
        assert load_result is True
        
        # Verify all documents are present
        for doc_id in doc_ids:
            retrieved_doc = manager2.get_document(doc_id)
            assert retrieved_doc is not None
            assert retrieved_doc.metadata["session"] == 1
        
        # Verify search functionality
        results = manager2.similarity_search("persistence test", k=3)
        assert len(results) >= 3
        
        # Test backup and restore
        backup_path = manager2.create_backup()
        assert backup_path is not None
        assert Path(backup_path).exists()
        
        # Add new data to second session
        new_document = Document(
            page_content="New document in second session.",
            metadata={"source": "session2_test.txt", "session": 2}
        )
        new_doc_ids = manager2.add_documents([new_document])
        
        # Restore from backup (should revert to session 1 state)
        restore_result = manager2.restore_from_backup(backup_path)
        assert restore_result is True
        
        # Verify restoration
        restored_doc = manager2.get_document(new_doc_ids[0])
        assert restored_doc is None  # Should be gone after restore
        
        # Original documents should still be there
        for doc_id in doc_ids:
            retrieved_doc = manager2.get_document(doc_id)
            assert retrieved_doc is not None


class TestMultiBackendIntegration:
    """Integration tests for multiple storage backends."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_local_backend_integration(self, mock_embeddings):
        """Test local backend integration."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path=self.temp_dir
        )
        
        manager = VectorDatabaseManager(config)
        
        # Test basic operations
        document = Document(
            page_content="Local backend test document.",
            metadata={"backend": "local"}
        )
        
        doc_ids = manager.add_documents([document])
        assert len(doc_ids) == 1
        
        results = manager.similarity_search("local backend", k=1)
        assert len(results) >= 1
        assert results[0].metadata["backend"] == "local"
    
    @patch('boto3.client')
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_s3_backend_integration(self, mock_embeddings, mock_boto3):
        """Test S3 backend integration."""
        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_boto3.return_value = mock_s3_client
        mock_s3_client.head_bucket.return_value = {}
        mock_s3_client.put_object.return_value = {}
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: b'{"test": "data"}')
        }
        mock_s3_client.list_objects_v2.return_value = {'Contents': []}
        
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        config = VectorDBConfig(
            storage_type="s3",
            embedding_model="openai",
            storage_path="test-bucket",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            aws_region="us-east-1"
        )
        
        manager = VectorDatabaseManager(config)
        
        # Test basic operations
        document = Document(
            page_content="S3 backend test document.",
            metadata={"backend": "s3"}
        )
        
        doc_ids = manager.add_documents([document])
        assert len(doc_ids) == 1
        
        # Verify S3 operations were called
        assert mock_s3_client.put_object.called


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path=self.temp_dir
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_large_scale_ingestion_performance(self, mock_embeddings):
        """Test performance with large-scale document ingestion."""
        # Mock embedding service for large batch
        batch_size = 1000
        mock_embeddings.return_value.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4] for _ in range(batch_size)
        ]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        
        # Create large batch of documents
        documents = []
        for i in range(batch_size):
            documents.append(Document(
                page_content=f"Large scale test document {i} with various content to test performance.",
                metadata={"doc_id": i, "batch": "large_scale_test"}
            ))
        
        # Measure ingestion performance
        import time
        start_time = time.time()
        doc_ids = manager.add_documents(documents)
        ingestion_time = time.time() - start_time
        
        assert len(doc_ids) == batch_size
        print(f"Ingested {batch_size} documents in {ingestion_time:.2f} seconds")
        print(f"Ingestion rate: {batch_size / ingestion_time:.2f} docs/second")
        
        # Measure search performance
        start_time = time.time()
        results = manager.similarity_search("test document performance", k=50)
        search_time = time.time() - start_time
        
        assert len(results) <= 50
        print(f"Search completed in {search_time:.4f} seconds")
        print(f"Search rate: {len(results) / search_time:.2f} results/second")
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_concurrent_operations_performance(self, mock_embeddings):
        """Test performance under concurrent operations."""
        import threading
        import time
        
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        
        # Add initial documents
        initial_docs = [
            Document(
                page_content=f"Initial document {i} for concurrent testing.",
                metadata={"type": "initial", "doc_id": i}
            )
            for i in range(100)
        ]
        manager.add_documents(initial_docs)
        
        # Define concurrent operations
        results = {"searches": [], "additions": [], "errors": []}
        
        def concurrent_search(thread_id):
            try:
                start_time = time.time()
                search_results = manager.similarity_search(f"concurrent test {thread_id}", k=5)
                duration = time.time() - start_time
                results["searches"].append(duration)
            except Exception as e:
                results["errors"].append(str(e))
        
        def concurrent_addition(thread_id):
            try:
                start_time = time.time()
                doc = Document(
                    page_content=f"Concurrent document from thread {thread_id}.",
                    metadata={"type": "concurrent", "thread_id": thread_id}
                )
                manager.add_documents([doc])
                duration = time.time() - start_time
                results["additions"].append(duration)
            except Exception as e:
                results["errors"].append(str(e))
        
        # Run concurrent operations
        threads = []
        num_threads = 10
        
        for i in range(num_threads):
            # Mix of search and addition operations
            if i % 2 == 0:
                thread = threading.Thread(target=concurrent_search, args=(i,))
            else:
                thread = threading.Thread(target=concurrent_addition, args=(i,))
            threads.append(thread)
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyze results
        assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"
        assert len(results["searches"]) > 0
        assert len(results["additions"]) > 0
        
        avg_search_time = sum(results["searches"]) / len(results["searches"])
        avg_addition_time = sum(results["additions"]) / len(results["additions"])
        
        print(f"Concurrent operations completed in {total_time:.2f} seconds")
        print(f"Average search time: {avg_search_time:.4f} seconds")
        print(f"Average addition time: {avg_addition_time:.4f} seconds")
        
        # Performance assertions
        assert avg_search_time < 1.0  # Searches should be fast
        assert avg_addition_time < 2.0  # Additions should be reasonable
        assert total_time < 10.0  # Overall should complete quickly