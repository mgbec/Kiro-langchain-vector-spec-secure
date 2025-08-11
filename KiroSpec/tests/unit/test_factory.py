"""
Unit tests for factory classes.
"""

import pytest
from unittest.mock import Mock, patch

from langchain_vector_db.factory import (
    VectorStoreFactory,
    EmbeddingServiceFactory,
    DocumentProcessorFactory,
    VectorDatabaseFactory,
    ConfigurationValidator
)
from langchain_vector_db.models.config import VectorDBConfig
from langchain_vector_db.storage.interface import VectorStoreInterface
from langchain_vector_db.exceptions import ConfigurationException


class MockVectorStore(VectorStoreInterface):
    """Mock vector store for testing."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def add_vectors(self, vectors, documents):
        return []
    
    def search_vectors(self, query_vector, k=4):
        return []
    
    def update_vector(self, doc_id, vector, document):
        return True
    
    def delete_vectors(self, doc_ids):
        return True
    
    def get_document(self, doc_id):
        return None
    
    def persist(self):
        return True
    
    def load(self):
        return True
    
    def get_vector_count(self):
        return 0
    
    def health_check(self):
        return True


class TestVectorStoreFactory:
    """Test cases for VectorStoreFactory."""
    
    def test_get_available_store_types(self):
        """Test getting available store types."""
        types = VectorStoreFactory.get_available_store_types()
        
        assert "local" in types
        assert "s3" in types
        assert isinstance(types, list)
    
    def test_is_store_type_supported(self):
        """Test checking if store type is supported."""
        assert VectorStoreFactory.is_store_type_supported("local") is True
        assert VectorStoreFactory.is_store_type_supported("s3") is True
        assert VectorStoreFactory.is_store_type_supported("LOCAL") is True  # Case insensitive
        assert VectorStoreFactory.is_store_type_supported("invalid") is False
    
    def test_register_store_type(self):
        """Test registering a new store type."""
        # Register a mock store type
        VectorStoreFactory.register_store_type("mock", MockVectorStore)
        
        assert VectorStoreFactory.is_store_type_supported("mock") is True
        assert "mock" in VectorStoreFactory.get_available_store_types()
        
        # Clean up
        del VectorStoreFactory._store_registry["mock"]
    
    def test_register_invalid_store_type(self):
        """Test registering an invalid store type."""
        class InvalidStore:
            pass
        
        with pytest.raises(ConfigurationException) as exc_info:
            VectorStoreFactory.register_store_type("invalid", InvalidStore)
        
        assert "must implement VectorStoreInterface" in str(exc_info.value)
    
    @patch('langchain_vector_db.factory.LocalVectorStore')
    def test_create_local_vector_store(self, mock_local_store):
        """Test creating a local vector store."""
        mock_local_store.return_value = Mock()
        
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db"
        )
        
        store = VectorStoreFactory.create_vector_store(config)
        
        assert store is not None
        mock_local_store.assert_called_once_with(storage_path="./test_db")
    
    @patch('langchain_vector_db.factory.S3VectorStore')
    def test_create_s3_vector_store(self, mock_s3_store):
        """Test creating an S3 vector store."""
        mock_s3_store.return_value = Mock()
        
        config = VectorDBConfig(
            storage_type="s3",
            embedding_model="openai",
            storage_path="test-bucket",
            aws_region="us-west-2"
        )
        
        store = VectorStoreFactory.create_vector_store(config)
        
        assert store is not None
        mock_s3_store.assert_called_once_with(
            bucket_name="test-bucket",
            s3_prefix="vectors/",
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_region="us-west-2"
        )
    
    def test_create_unsupported_vector_store(self):
        """Test creating an unsupported vector store."""
        config = VectorDBConfig(
            storage_type="unsupported",
            embedding_model="openai",
            storage_path="./test_db"
        )
        
        with pytest.raises(ConfigurationException) as exc_info:
            VectorStoreFactory.create_vector_store(config)
        
        assert "Unsupported storage type" in str(exc_info.value)


class TestEmbeddingServiceFactory:
    """Test cases for EmbeddingServiceFactory."""
    
    @patch('langchain_vector_db.factory.EmbeddingService')
    def test_create_embedding_service(self, mock_embedding_service):
        """Test creating an embedding service."""
        mock_embedding_service.return_value = Mock()
        
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db"
        )
        
        service = EmbeddingServiceFactory.create_embedding_service(config)
        
        assert service is not None
        mock_embedding_service.assert_called_once_with(
            embedding_model="huggingface",
            model_kwargs={}
        )
    
    @patch('langchain_vector_db.factory.EmbeddingService')
    def test_create_embedding_service_with_kwargs(self, mock_embedding_service):
        """Test creating an embedding service with model kwargs."""
        mock_embedding_service.return_value = Mock()
        
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path="./test_db",
            model_kwargs={"api_key": "test-key"}
        )
        
        service = EmbeddingServiceFactory.create_embedding_service(config)
        
        assert service is not None
        mock_embedding_service.assert_called_once_with(
            embedding_model="openai",
            model_kwargs={"api_key": "test-key"}
        )


class TestDocumentProcessorFactory:
    """Test cases for DocumentProcessorFactory."""
    
    @patch('langchain_vector_db.factory.DocumentProcessor')
    def test_create_document_processor(self, mock_doc_processor):
        """Test creating a document processor."""
        mock_doc_processor.return_value = Mock()
        
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db",
            chunk_size=500,
            chunk_overlap=50
        )
        
        processor = DocumentProcessorFactory.create_document_processor(config)
        
        assert processor is not None
        mock_doc_processor.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=50
        )


class TestVectorDatabaseFactory:
    """Test cases for VectorDatabaseFactory."""
    
    @patch('langchain_vector_db.factory.VectorStoreFactory.create_vector_store')
    @patch('langchain_vector_db.factory.DocumentProcessorFactory.create_document_processor')
    @patch('langchain_vector_db.factory.EmbeddingServiceFactory.create_embedding_service')
    def test_create_components(self, mock_embedding, mock_processor, mock_store):
        """Test creating all components."""
        mock_embedding.return_value = Mock()
        mock_processor.return_value = Mock()
        mock_store.return_value = Mock()
        
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db"
        )
        
        components = VectorDatabaseFactory.create_components(config)
        
        assert "embedding_service" in components
        assert "document_processor" in components
        assert "vector_store" in components
        
        mock_embedding.assert_called_once_with(config)
        mock_processor.assert_called_once_with(config)
        mock_store.assert_called_once_with(config)
    
    def test_validate_configuration_valid(self):
        """Test validating a valid configuration."""
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db"
        )
        
        result = VectorDatabaseFactory.validate_configuration(config)
        
        assert result["overall"] is True
        assert result["storage_type"] is True
        assert result["embedding_model"] is True
        assert result["storage_path"] is True
        assert result["chunk_parameters"] is True
    
    def test_validate_configuration_invalid_storage_type(self):
        """Test validating configuration with invalid storage type."""
        config = VectorDBConfig(
            storage_type="invalid",
            embedding_model="huggingface",
            storage_path="./test_db"
        )
        
        result = VectorDatabaseFactory.validate_configuration(config)
        
        assert result["overall"] is False
        assert result["storage_type"] is False
    
    def test_validate_configuration_invalid_embedding_model(self):
        """Test validating configuration with invalid embedding model."""
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="invalid",
            storage_path="./test_db"
        )
        
        result = VectorDatabaseFactory.validate_configuration(config)
        
        assert result["overall"] is False
        assert result["embedding_model"] is False
    
    def test_validate_configuration_empty_storage_path(self):
        """Test validating configuration with empty storage path."""
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path=""
        )
        
        result = VectorDatabaseFactory.validate_configuration(config)
        
        assert result["overall"] is False
        assert result["storage_path"] is False
    
    def test_validate_configuration_invalid_chunk_parameters(self):
        """Test validating configuration with invalid chunk parameters."""
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db",
            chunk_size=100,
            chunk_overlap=100  # Equal to chunk_size
        )
        
        result = VectorDatabaseFactory.validate_configuration(config)
        
        assert result["overall"] is False
        assert result["chunk_parameters"] is False
    
    def test_validate_s3_configuration_valid(self):
        """Test validating valid S3 configuration."""
        config = VectorDBConfig(
            storage_type="s3",
            embedding_model="openai",
            storage_path="test-bucket",
            aws_region="us-east-1",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret"
        )
        
        result = VectorDatabaseFactory.validate_configuration(config)
        
        assert result["overall"] is True
        assert result["s3_config"] is True
    
    def test_validate_s3_configuration_missing_region(self):
        """Test validating S3 configuration with missing region."""
        config = VectorDBConfig(
            storage_type="s3",
            embedding_model="openai",
            storage_path="test-bucket",
            aws_region=""
        )
        
        result = VectorDatabaseFactory.validate_configuration(config)
        
        assert result["overall"] is False
        assert result["s3_config"] is False
    
    def test_validate_s3_configuration_partial_credentials(self):
        """Test validating S3 configuration with partial credentials."""
        config = VectorDBConfig(
            storage_type="s3",
            embedding_model="openai",
            storage_path="test-bucket",
            aws_region="us-east-1",
            aws_access_key_id="test-key"
            # Missing aws_secret_access_key
        )
        
        result = VectorDatabaseFactory.validate_configuration(config)
        
        assert result["overall"] is False
        assert result["s3_config"] is False
    
    def test_get_configuration_recommendations(self):
        """Test getting configuration recommendations."""
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path="./test_db",
            chunk_size=3000  # Large chunk size
        )
        
        recommendations = VectorDatabaseFactory.get_configuration_recommendations(config)
        
        assert "storage" in recommendations
        assert "embedding" in recommendations
        assert "chunking" in recommendations
        assert "Local storage" in recommendations["storage"]
        assert "OpenAI embeddings" in recommendations["embedding"]
        assert "Large chunk sizes" in recommendations["chunking"]


class TestConfigurationValidator:
    """Test cases for ConfigurationValidator."""
    
    def test_validate_and_suggest_valid_config(self):
        """Test validating and suggesting for valid configuration."""
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db"
        )
        
        result = ConfigurationValidator.validate_and_suggest(config)
        
        assert result["is_valid"] is True
        assert "validation_results" in result
        assert "recommendations" in result
        assert "issues" in result
        assert "suggested_fixes" in result
        assert len(result["issues"]) == 0
    
    def test_validate_and_suggest_invalid_config(self):
        """Test validating and suggesting for invalid configuration."""
        config = VectorDBConfig(
            storage_type="invalid",
            embedding_model="invalid",
            storage_path=""
        )
        
        result = ConfigurationValidator.validate_and_suggest(config)
        
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert len(result["suggested_fixes"]) > 0
        assert "Unsupported storage type" in result["issues"][0]
        assert "Unsupported embedding model" in result["issues"][1]
        assert "Storage path is empty" in result["issues"][2]


if __name__ == "__main__":
    pytest.main([__file__])