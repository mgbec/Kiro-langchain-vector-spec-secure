"""
Unit tests for EmbeddingService.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.embeddings.base import Embeddings

from langchain_vector_db.services.embedding import EmbeddingService
from langchain_vector_db.exceptions import EmbeddingException, ConfigurationException


class MockEmbeddings(Embeddings):
    """Mock embeddings class for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_documents(self, texts):
        return [[1.0] * self.dimension for _ in texts]
    
    def embed_query(self, text):
        return [1.0] * self.dimension


class TestEmbeddingService:
    """Test cases for EmbeddingService."""
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_openai_embeddings_initialization(self, mock_openai):
        """Test OpenAI embeddings initialization."""
        mock_openai.return_value = MockEmbeddings()
        
        service = EmbeddingService("openai")
        
        assert service.embedding_model == "openai"
        mock_openai.assert_called_once()
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_openai_embeddings_with_api_key_in_kwargs(self, mock_openai):
        """Test OpenAI embeddings with API key in model_kwargs."""
        mock_openai.return_value = MockEmbeddings()
        
        service = EmbeddingService(
            "openai", 
            model_kwargs={"api_key": "test-key", "model": "text-embedding-ada-002"}
        )
        
        mock_openai.assert_called_once()
        call_args = mock_openai.call_args[1]
        assert call_args["openai_api_key"] == "test-key"
        assert call_args["model"] == "text-embedding-ada-002"
    
    def test_openai_embeddings_missing_api_key(self):
        """Test OpenAI embeddings initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                EmbeddingService("openai")
            
            assert "OpenAI API key must be provided" in str(exc_info.value)
    
    @patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings')
    def test_huggingface_embeddings_initialization(self, mock_hf):
        """Test HuggingFace embeddings initialization."""
        mock_hf.return_value = MockEmbeddings()
        
        service = EmbeddingService("huggingface")
        
        assert service.embedding_model == "huggingface"
        mock_hf.assert_called_once()
        call_args = mock_hf.call_args[1]
        assert "sentence-transformers/all-MiniLM-L6-v2" in call_args["model_name"]
    
    @patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings')
    def test_huggingface_embeddings_with_custom_model(self, mock_hf):
        """Test HuggingFace embeddings with custom model."""
        mock_hf.return_value = MockEmbeddings()
        
        service = EmbeddingService(
            "huggingface",
            model_kwargs={"model_name": "custom-model"}
        )
        
        call_args = mock_hf.call_args[1]
        assert call_args["model_name"] == "custom-model"
    
    def test_custom_embeddings_initialization(self):
        """Test custom embeddings initialization."""
        mock_class = Mock(return_value=MockEmbeddings())
        
        service = EmbeddingService(
            "custom",
            model_kwargs={"model_class": mock_class, "param1": "value1"}
        )
        
        assert service.embedding_model == "custom"
        mock_class.assert_called_once_with(param1="value1")
    
    def test_custom_embeddings_missing_model_class(self):
        """Test custom embeddings without model_class."""
        with pytest.raises(ConfigurationException) as exc_info:
            EmbeddingService("custom")
        
        assert "model_class is required for custom embedding models" in str(exc_info.value)
    
    def test_unsupported_embedding_model(self):
        """Test initialization with unsupported model."""
        with pytest.raises(ConfigurationException) as exc_info:
            EmbeddingService("unsupported")
        
        assert "Unsupported embedding model" in str(exc_info.value)
    
    def test_embed_documents(self):
        """Test document embedding generation."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = MockEmbeddings(dimension=384)
            
            service = EmbeddingService("huggingface")
            texts = ["document 1", "document 2", "document 3"]
            
            embeddings = service.embed_documents(texts)
            
            assert len(embeddings) == 3
            assert all(len(emb) == 384 for emb in embeddings)
            assert service.get_embedding_dimension() == 384
    
    def test_embed_documents_empty_list(self):
        """Test embedding empty document list."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = MockEmbeddings()
            
            service = EmbeddingService("huggingface")
            embeddings = service.embed_documents([])
            
            assert embeddings == []
    
    def test_embed_query(self):
        """Test query embedding generation."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = MockEmbeddings(dimension=512)
            
            service = EmbeddingService("huggingface")
            embedding = service.embed_query("test query")
            
            assert len(embedding) == 512
            assert service.get_embedding_dimension() == 512
    
    def test_embed_query_empty_text(self):
        """Test embedding empty query text."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = MockEmbeddings()
            
            service = EmbeddingService("huggingface")
            
            with pytest.raises(EmbeddingException) as exc_info:
                service.embed_query("")
            
            assert "Query text cannot be empty" in str(exc_info.value)
    
    def test_embed_documents_batch(self):
        """Test batch document embedding."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = MockEmbeddings(dimension=256)
            
            service = EmbeddingService("huggingface")
            texts = [f"document {i}" for i in range(10)]
            
            embeddings = service.embed_documents_batch(texts, batch_size=3)
            
            assert len(embeddings) == 10
            assert all(len(emb) == 256 for emb in embeddings)
    
    def test_embed_documents_batch_invalid_batch_size(self):
        """Test batch embedding with invalid batch size."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = MockEmbeddings()
            
            service = EmbeddingService("huggingface")
            
            with pytest.raises(EmbeddingException) as exc_info:
                service.embed_documents_batch(["text"], batch_size=0)
            
            assert "Batch size must be greater than 0" in str(exc_info.value)
    
    def test_get_embedding_dimension_cached(self):
        """Test getting embedding dimension when cached."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = MockEmbeddings(dimension=768)
            
            service = EmbeddingService("huggingface")
            
            # First call should cache the dimension
            service.embed_query("test")
            dimension = service.get_embedding_dimension()
            
            assert dimension == 768
    
    def test_get_embedding_dimension_not_cached(self):
        """Test getting embedding dimension when not cached."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_embeddings = MockEmbeddings(dimension=1024)
            mock_hf.return_value = mock_embeddings
            
            service = EmbeddingService("huggingface")
            dimension = service.get_embedding_dimension()
            
            assert dimension == 1024
    
    def test_get_model_info(self):
        """Test getting model information."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = MockEmbeddings()
            
            service = EmbeddingService(
                "huggingface",
                model_kwargs={"model_name": "test-model", "param1": "value1"}
            )
            
            info = service.get_model_info()
            
            assert info["model_type"] == "huggingface"
            assert info["model_name"] == "test-model"
            assert info["model_kwargs"]["param1"] == "value1"
            assert "api_key" not in str(info)  # Sensitive data should be excluded
    
    def test_health_check_healthy(self):
        """Test health check when service is healthy."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = MockEmbeddings()
            
            service = EmbeddingService("huggingface")
            assert service.health_check() is True
    
    def test_health_check_unhealthy(self):
        """Test health check when service is unhealthy."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_embeddings = Mock()
            mock_embeddings.embed_query.side_effect = Exception("Service unavailable")
            mock_hf.return_value = mock_embeddings
            
            service = EmbeddingService("huggingface")
            assert service.health_check() is False
    
    def test_embedding_service_error_handling(self):
        """Test error handling in embedding generation."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_embeddings = Mock()
            mock_embeddings.embed_documents.side_effect = Exception("API Error")
            mock_hf.return_value = mock_embeddings
            
            service = EmbeddingService("huggingface")
            
            with pytest.raises(EmbeddingException) as exc_info:
                service.embed_documents(["test"])
            
            assert "Failed to generate document embeddings" in str(exc_info.value)
    
    def test_string_representations(self):
        """Test string representations of embedding service."""
        with patch('langchain_vector_db.services.embedding.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = MockEmbeddings(dimension=384)
            
            service = EmbeddingService("huggingface", model_kwargs={"param1": "value1"})
            
            # Test after embedding to cache dimension
            service.embed_query("test")
            
            str_repr = str(service)
            assert "EmbeddingService" in str_repr
            assert "huggingface" in str_repr
            assert "384" in str_repr
            
            repr_str = repr(service)
            assert "EmbeddingService" in repr_str
            assert "huggingface" in repr_str
            assert "param1" in repr_str


class TestEmbeddingServiceIntegration:
    """Integration tests for EmbeddingService with real models (mocked)."""
    
    @patch('langchain_vector_db.services.embedding.CohereEmbeddings')
    @patch.dict('os.environ', {'COHERE_API_KEY': 'test-key'})
    def test_cohere_embeddings_initialization(self, mock_cohere):
        """Test Cohere embeddings initialization."""
        mock_cohere.return_value = MockEmbeddings()
        
        service = EmbeddingService("cohere")
        
        assert service.embedding_model == "cohere"
        mock_cohere.assert_called_once()
    
    @patch('langchain_vector_db.services.embedding.AzureOpenAIEmbeddings')
    def test_azure_openai_embeddings_initialization(self, mock_azure):
        """Test Azure OpenAI embeddings initialization."""
        mock_azure.return_value = MockEmbeddings()
        
        service = EmbeddingService(
            "azure-openai",
            model_kwargs={
                "azure_endpoint": "https://test.openai.azure.com/",
                "api_key": "test-key",
                "api_version": "2023-05-15"
            }
        )
        
        assert service.embedding_model == "azure-openai"
        mock_azure.assert_called_once()
    
    def test_azure_openai_missing_required_params(self):
        """Test Azure OpenAI embeddings with missing required parameters."""
        with pytest.raises(ConfigurationException) as exc_info:
            EmbeddingService("azure-openai", model_kwargs={"api_key": "test-key"})
        
        assert "azure_endpoint" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])