"""
Unit tests for VectorDatabaseManager.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from langchain_vector_db.manager import VectorDatabaseManager
from langchain_vector_db.models.config import VectorDBConfig
from langchain_vector_db.models.document import Document
from langchain_vector_db.exceptions import VectorDBException, ConfigurationException


class TestVectorDatabaseManager:
    """Test cases for VectorDatabaseManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db"
        )
        
        # Mock components
        self.mock_embedding_service = Mock()
        self.mock_embedding_service.embed_query.return_value = [1.0, 2.0, 3.0]
        self.mock_embedding_service.embed_documents.return_value = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.mock_embedding_service.get_embedding_dimension.return_value = 3
        self.mock_embedding_service.health_check.return_value = True
        self.mock_embedding_service.get_model_info.return_value = {"model_type": "huggingface"}
        
        self.mock_document_processor = Mock()
        self.mock_document_processor.process_texts.return_value = [
            Document(page_content="Test document 1", doc_id="doc1"),
            Document(page_content="Test document 2", doc_id="doc2")
        ]
        self.mock_document_processor.process_documents.return_value = [
            Document(page_content="File document", doc_id="doc3")
        ]
        self.mock_document_processor.health_check.return_value = True
        self.mock_document_processor.get_processor_info.return_value = {"chunk_size": 1000}
        self.mock_document_processor.validate_files.return_value = {
            "valid": ["test.txt"], "missing": [], "unsupported": []
        }
        self.mock_document_processor.get_supported_extensions.return_value = [".txt", ".pdf"]
        
        self.mock_vector_store = Mock()
        self.mock_vector_store.add_vectors.return_value = ["doc1", "doc2"]
        self.mock_vector_store.search_vectors.return_value = [
            (Document(page_content="Result 1", doc_id="doc1"), 0.9),
            (Document(page_content="Result 2", doc_id="doc2"), 0.8)
        ]
        self.mock_vector_store.get_document.return_value = Document(page_content="Test", doc_id="doc1")
        self.mock_vector_store.update_vector.return_value = True
        self.mock_vector_store.delete_vectors.return_value = True
        self.mock_vector_store.get_vector_count.return_value = 2
        self.mock_vector_store.persist.return_value = True
        self.mock_vector_store.load.return_value = True
        self.mock_vector_store.health_check.return_value = True
        self.mock_vector_store.get_storage_info.return_value = {"storage_type": "local"}
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_initialization_local_storage(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test manager initialization with local storage."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        assert manager.config == self.config
        assert manager.embedding_service == self.mock_embedding_service
        assert manager.document_processor == self.mock_document_processor
        assert manager.vector_store == self.mock_vector_store
        
        # Verify components were created with correct parameters
        mock_embedding.assert_called_once_with(
            embedding_model="huggingface",
            model_kwargs={}
        )
        mock_doc_processor.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=200
        )
        mock_local_store.assert_called_once_with(
            storage_path="./test_db"
        )
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.S3VectorStore')
    def test_initialization_s3_storage(self, mock_s3_store, mock_doc_processor, mock_embedding):
        """Test manager initialization with S3 storage."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_s3_store.return_value = self.mock_vector_store
        
        s3_config = VectorDBConfig(
            storage_type="s3",
            embedding_model="openai",
            storage_path="test-bucket",
            aws_region="us-west-2"
        )
        
        manager = VectorDatabaseManager(s3_config)
        
        # Verify S3 store was created with correct parameters
        mock_s3_store.assert_called_once_with(
            bucket_name="test-bucket",
            s3_prefix="vectors/",
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_region="us-west-2"
        )
    
    def test_initialization_invalid_storage_type(self):
        """Test initialization with invalid storage type."""
        invalid_config = VectorDBConfig(
            storage_type="invalid",
            embedding_model="openai",
            storage_path="./test_db"
        )
        
        with pytest.raises(ConfigurationException) as exc_info:
            VectorDatabaseManager(invalid_config)
        
        assert "Unsupported storage type" in str(exc_info.value)
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_add_documents_basic(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test adding documents to the database."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2")
        ]
        
        doc_ids = manager.add_documents(documents)
        
        assert doc_ids == ["doc1", "doc2"]
        
        # Verify embeddings were generated
        assert self.mock_embedding_service.embed_query.call_count == 2
        
        # Verify documents were added to vector store
        self.mock_vector_store.add_vectors.assert_called_once()
        call_args = self.mock_vector_store.add_vectors.call_args
        embeddings, docs = call_args[0]
        assert len(embeddings) == 2
        assert len(docs) == 2
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_add_documents_with_existing_embeddings(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test adding documents that already have embeddings."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        documents = [
            Document(page_content="Document 1", doc_id="doc1", embedding=[1.0, 2.0, 3.0])
        ]
        
        doc_ids = manager.add_documents(documents, generate_embeddings=False)
        
        assert doc_ids == ["doc1", "doc2"]
        
        # Verify embeddings were not generated
        self.mock_embedding_service.embed_query.assert_not_called()
        
        # Verify documents were added to vector store
        self.mock_vector_store.add_vectors.assert_called_once()
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_add_documents_empty_list(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test adding empty document list."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        doc_ids = manager.add_documents([])
        
        assert doc_ids == []
        self.mock_vector_store.add_vectors.assert_not_called()
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_add_texts(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test adding text strings to the database."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        texts = ["Text 1", "Text 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        
        doc_ids = manager.add_texts(texts, metadatas)
        
        assert doc_ids == ["doc1", "doc2"]
        
        # Verify document processor was called
        self.mock_document_processor.process_texts.assert_called_once_with(texts, metadatas)
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_add_documents_from_files(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test adding documents from files."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        file_paths = ["test1.txt", "test2.txt"]
        metadata_override = {"category": "test"}
        
        doc_ids = manager.add_documents_from_files(file_paths, metadata_override)
        
        assert doc_ids == ["doc1", "doc2"]
        
        # Verify document processor was called
        self.mock_document_processor.process_documents.assert_called_once_with(
            file_paths, metadata_override
        )
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_similarity_search(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test similarity search functionality."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        query = "test query"
        results = manager.similarity_search(query, k=2)
        
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        
        # Verify query embedding was generated
        self.mock_embedding_service.embed_query.assert_called_once_with(query)
        
        # Verify vector store search was called
        self.mock_vector_store.search_vectors.assert_called_once_with([1.0, 2.0, 3.0], 2)
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_similarity_search_with_score(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test similarity search with scores."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        query = "test query"
        results = manager.similarity_search_with_score(query, k=2)
        
        assert len(results) == 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        assert all(isinstance(result[0], Document) for result in results)
        assert all(isinstance(result[1], (int, float)) for result in results)
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_similarity_search_with_metadata_filter(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test similarity search with metadata filtering."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        # Mock search results with metadata
        mock_results = [
            (Document(page_content="Result 1", doc_id="doc1", metadata={"category": "A"}), 0.9),
            (Document(page_content="Result 2", doc_id="doc2", metadata={"category": "B"}), 0.8)
        ]
        self.mock_vector_store.search_vectors.return_value = mock_results
        
        manager = VectorDatabaseManager(self.config)
        
        query = "test query"
        filter_metadata = {"category": "A"}
        results = manager.similarity_search_with_score(query, k=2, filter_metadata=filter_metadata)
        
        # Should only return documents matching the filter
        assert len(results) == 1
        assert results[0][0].metadata["category"] == "A"
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_update_document(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test updating a document."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        doc_id = "doc1"
        updated_doc = Document(page_content="Updated content", doc_id=doc_id)
        
        result = manager.update_document(doc_id, updated_doc)
        
        assert result is True
        
        # Verify embedding was generated
        self.mock_embedding_service.embed_query.assert_called_once_with("Updated content")
        
        # Verify vector store update was called
        self.mock_vector_store.update_vector.assert_called_once()
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_delete_documents(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test deleting documents."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        doc_ids = ["doc1", "doc2"]
        result = manager.delete_documents(doc_ids)
        
        assert result is True
        self.mock_vector_store.delete_vectors.assert_called_once_with(doc_ids)
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_get_document(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test retrieving a document."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        doc_id = "doc1"
        document = manager.get_document(doc_id)
        
        assert document is not None
        assert document.doc_id == "doc1"
        self.mock_vector_store.get_document.assert_called_once_with(doc_id)
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_get_document_metadata(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test getting document metadata."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        test_doc = Document(
            page_content="Test", 
            doc_id="doc1", 
            metadata={"source": "test.txt"}
        )
        self.mock_vector_store.get_document.return_value = test_doc
        
        manager = VectorDatabaseManager(self.config)
        
        metadata = manager.get_document_metadata("doc1")
        
        assert metadata == {"source": "test.txt"}
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_persist_and_load(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test persist and load operations."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        # Test persist
        result = manager.persist()
        assert result is True
        self.mock_vector_store.persist.assert_called_once()
        
        # Test load
        result = manager.load()
        assert result is True
        self.mock_vector_store.load.assert_called_once()
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_get_vector_count(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test getting vector count."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        count = manager.get_vector_count()
        assert count == 2
        self.mock_vector_store.get_vector_count.assert_called_once()
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_get_embedding_dimension(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test getting embedding dimension."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        dimension = manager.get_embedding_dimension()
        assert dimension == 3
        self.mock_embedding_service.get_embedding_dimension.assert_called_once()
        
        # Test caching - second call should not call the service again
        dimension2 = manager.get_embedding_dimension()
        assert dimension2 == 3
        assert self.mock_embedding_service.get_embedding_dimension.call_count == 1
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_health_check(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test health check functionality."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        health = manager.health_check()
        
        assert "embedding_service" in health
        assert "document_processor" in health
        assert "vector_store" in health
        assert "overall" in health
        assert health["overall"] is True
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_health_check_with_failures(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test health check with component failures."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        # Make one component unhealthy
        self.mock_embedding_service.health_check.return_value = False
        
        manager = VectorDatabaseManager(self.config)
        
        health = manager.health_check()
        
        assert health["embedding_service"] is False
        assert health["document_processor"] is True
        assert health["vector_store"] is True
        assert health["overall"] is False
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_get_system_info(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test getting system information."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        info = manager.get_system_info()
        
        assert "config" in info
        assert "vector_count" in info
        assert "embedding_dimension" in info
        assert "embedding_service" in info
        assert "document_processor" in info
        assert "vector_store" in info
        assert "health" in info
        
        assert info["vector_count"] == 2
        assert info["embedding_dimension"] == 3
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_validate_files(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test file validation."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        file_paths = ["test1.txt", "test2.pdf"]
        result = manager.validate_files(file_paths)
        
        assert "valid" in result
        assert "missing" in result
        assert "unsupported" in result
        self.mock_document_processor.validate_files.assert_called_once_with(file_paths)
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_get_supported_file_extensions(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test getting supported file extensions."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        extensions = manager.get_supported_file_extensions()
        
        assert extensions == [".txt", ".pdf"]
        self.mock_document_processor.get_supported_extensions.assert_called_once()
    
    @patch('langchain_vector_db.manager.EmbeddingService')
    @patch('langchain_vector_db.manager.DocumentProcessor')
    @patch('langchain_vector_db.manager.LocalVectorStore')
    def test_string_representations(self, mock_local_store, mock_doc_processor, mock_embedding):
        """Test string representations of the manager."""
        mock_embedding.return_value = self.mock_embedding_service
        mock_doc_processor.return_value = self.mock_document_processor
        mock_local_store.return_value = self.mock_vector_store
        
        manager = VectorDatabaseManager(self.config)
        
        str_repr = str(manager)
        assert "VectorDatabaseManager" in str_repr
        assert "local" in str_repr
        assert "huggingface" in str_repr
        
        repr_str = repr(manager)
        assert "VectorDatabaseManager" in repr_str
        assert "local" in repr_str
        assert "huggingface" in repr_str


class TestMetadataFiltering:
    """Test metadata filtering functionality."""
    
    def test_document_matches_metadata_filter_simple(self):
        """Test simple metadata filtering."""
        manager = VectorDatabaseManager.__new__(VectorDatabaseManager)  # Skip __init__
        
        document = Document(
            page_content="Test",
            metadata={"category": "A", "score": 85}
        )
        
        # Test exact match
        assert manager._document_matches_metadata_filter(document, {"category": "A"}) is True
        assert manager._document_matches_metadata_filter(document, {"category": "B"}) is False
        
        # Test multiple criteria
        assert manager._document_matches_metadata_filter(
            document, {"category": "A", "score": 85}
        ) is True
        assert manager._document_matches_metadata_filter(
            document, {"category": "A", "score": 90}
        ) is False
    
    def test_document_matches_metadata_filter_operators(self):
        """Test metadata filtering with operators."""
        manager = VectorDatabaseManager.__new__(VectorDatabaseManager)  # Skip __init__
        
        document = Document(
            page_content="Test",
            metadata={"score": 85, "tags": ["A", "B"]}
        )
        
        # Test comparison operators
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$gt": 80}}
        ) is True
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$gt": 90}}
        ) is False
        
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$gte": 85}}
        ) is True
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$lt": 90}}
        ) is True
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$lte": 85}}
        ) is True
        
        # Test $in operator
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$in": [80, 85, 90]}}
        ) is True
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$in": [70, 75, 80]}}
        ) is False
        
        # Test $nin operator
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$nin": [70, 75, 80]}}
        ) is True
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$nin": [80, 85, 90]}}
        ) is False
        
        # Test $ne operator
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$ne": 90}}
        ) is True
        assert manager._document_matches_metadata_filter(
            document, {"score": {"$ne": 85}}
        ) is False


if __name__ == "__main__":
    pytest.main([__file__])