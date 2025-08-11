"""
Embedding service for generating vector embeddings using LangChain.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from ..exceptions import EmbeddingException, ConfigurationException


class EmbeddingService:
    """Service for generating vector embeddings using various LangChain embedding models."""
    
    def __init__(self, embedding_model: str, model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedding service.
        
        Args:
            embedding_model: Type of embedding model to use
            model_kwargs: Additional keyword arguments for the model
        """
        self.embedding_model = embedding_model
        self.model_kwargs = model_kwargs or {}
        self._embeddings: Optional[Embeddings] = None
        self._embedding_dimension: Optional[int] = None
        
        # Initialize the embedding model
        self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize the LangChain embeddings instance."""
        try:
            if self.embedding_model == "openai":
                self._embeddings = self._create_openai_embeddings()
            elif self.embedding_model == "huggingface":
                self._embeddings = self._create_huggingface_embeddings()
            elif self.embedding_model == "sentence-transformers":
                self._embeddings = self._create_sentence_transformers_embeddings()
            elif self.embedding_model == "cohere":
                self._embeddings = self._create_cohere_embeddings()
            elif self.embedding_model == "azure-openai":
                self._embeddings = self._create_azure_openai_embeddings()
            elif self.embedding_model == "custom":
                self._embeddings = self._create_custom_embeddings()
            else:
                raise ConfigurationException(
                    f"Unsupported embedding model: {self.embedding_model}"
                )
        except Exception as e:
            raise EmbeddingException(
                f"Failed to initialize {self.embedding_model} embeddings: {str(e)}"
            )
    
    def _create_openai_embeddings(self) -> OpenAIEmbeddings:
        """Create OpenAI embeddings instance."""
        # Get API key from model_kwargs or environment
        api_key = self.model_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationException(
                "OpenAI API key must be provided in model_kwargs or OPENAI_API_KEY environment variable"
            )
        
        # Set default model if not specified
        model = self.model_kwargs.get("model", "text-embedding-ada-002")
        
        kwargs = {
            "openai_api_key": api_key,
            "model": model,
            **{k: v for k, v in self.model_kwargs.items() if k not in ["api_key", "model"]}
        }
        
        return OpenAIEmbeddings(**kwargs)
    
    def _create_huggingface_embeddings(self) -> HuggingFaceEmbeddings:
        """Create HuggingFace embeddings instance."""
        # Set default model if not specified
        model_name = self.model_kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        
        kwargs = {
            "model_name": model_name,
            **{k: v for k, v in self.model_kwargs.items() if k != "model_name"}
        }
        
        return HuggingFaceEmbeddings(**kwargs)
    
    def _create_sentence_transformers_embeddings(self) -> HuggingFaceEmbeddings:
        """Create Sentence Transformers embeddings instance (using HuggingFace backend)."""
        # Set default model if not specified
        model_name = self.model_kwargs.get("model_name", "all-MiniLM-L6-v2")
        
        kwargs = {
            "model_name": model_name,
            **{k: v for k, v in self.model_kwargs.items() if k != "model_name"}
        }
        
        return HuggingFaceEmbeddings(**kwargs)
    
    def _create_cohere_embeddings(self) -> Embeddings:
        """Create Cohere embeddings instance."""
        try:
            from langchain.embeddings import CohereEmbeddings
        except ImportError:
            raise EmbeddingException(
                "Cohere embeddings require 'cohere' package. Install with: pip install cohere"
            )
        
        # Get API key from model_kwargs or environment
        api_key = self.model_kwargs.get("api_key") or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ConfigurationException(
                "Cohere API key must be provided in model_kwargs or COHERE_API_KEY environment variable"
            )
        
        kwargs = {
            "cohere_api_key": api_key,
            **{k: v for k, v in self.model_kwargs.items() if k != "api_key"}
        }
        
        return CohereEmbeddings(**kwargs)
    
    def _create_azure_openai_embeddings(self) -> Embeddings:
        """Create Azure OpenAI embeddings instance."""
        try:
            from langchain.embeddings import AzureOpenAIEmbeddings
        except ImportError:
            raise EmbeddingException(
                "Azure OpenAI embeddings require 'openai' package. Install with: pip install openai"
            )
        
        # Required parameters for Azure OpenAI
        required_params = ["azure_endpoint", "api_key", "api_version"]
        for param in required_params:
            if param not in self.model_kwargs:
                raise ConfigurationException(
                    f"Azure OpenAI requires '{param}' in model_kwargs"
                )
        
        return AzureOpenAIEmbeddings(**self.model_kwargs)
    
    def _create_custom_embeddings(self) -> Embeddings:
        """Create custom embeddings instance."""
        model_class = self.model_kwargs.get("model_class")
        if not model_class:
            raise ConfigurationException(
                "Custom embeddings require 'model_class' in model_kwargs"
            )
        
        # Remove model_class from kwargs before passing to constructor
        init_kwargs = {k: v for k, v in self.model_kwargs.items() if k != "model_class"}
        
        try:
            return model_class(**init_kwargs)
        except Exception as e:
            raise EmbeddingException(f"Failed to initialize custom embedding model: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingException: If embedding generation fails
        """
        if not texts:
            return []
        
        try:
            embeddings = self._embeddings.embed_documents(texts)
            
            # Validate embeddings
            if not embeddings or len(embeddings) != len(texts):
                raise EmbeddingException(
                    f"Expected {len(texts)} embeddings, got {len(embeddings) if embeddings else 0}"
                )
            
            # Cache embedding dimension from first result
            if embeddings and self._embedding_dimension is None:
                self._embedding_dimension = len(embeddings[0])
            
            return embeddings
            
        except Exception as e:
            if isinstance(e, EmbeddingException):
                raise
            raise EmbeddingException(f"Failed to generate document embeddings: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingException: If embedding generation fails
        """
        if not text:
            raise EmbeddingException("Query text cannot be empty")
        
        try:
            embedding = self._embeddings.embed_query(text)
            
            # Validate embedding
            if not embedding:
                raise EmbeddingException("Generated embedding is empty")
            
            # Cache embedding dimension from result
            if self._embedding_dimension is None:
                self._embedding_dimension = len(embedding)
            
            return embedding
            
        except Exception as e:
            if isinstance(e, EmbeddingException):
                raise
            raise EmbeddingException(f"Failed to generate query embedding: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
            
        Raises:
            EmbeddingException: If dimension cannot be determined
        """
        if self._embedding_dimension is not None:
            return self._embedding_dimension
        
        # Try to determine dimension by embedding a test string
        try:
            test_embedding = self.embed_query("test")
            return len(test_embedding)
        except Exception as e:
            raise EmbeddingException(f"Failed to determine embedding dimension: {str(e)}")
    
    def embed_documents_batch(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for documents in batches for efficiency.
        
        Args:
            texts: List of text documents to embed
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingException: If embedding generation fails
        """
        if not texts:
            return []
        
        if batch_size <= 0:
            raise EmbeddingException("Batch size must be greater than 0")
        
        all_embeddings = []
        
        try:
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            if isinstance(e, EmbeddingException):
                raise
            raise EmbeddingException(f"Failed to generate batch embeddings: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            "model_type": self.embedding_model,
            "model_kwargs": {k: v for k, v in self.model_kwargs.items() 
                           if k not in ["api_key", "cohere_api_key", "openai_api_key"]},  # Exclude sensitive data
            "embedding_dimension": self._embedding_dimension,
        }
        
        # Add model-specific information
        if self.embedding_model == "openai":
            info["model_name"] = self.model_kwargs.get("model", "text-embedding-ada-002")
        elif self.embedding_model in ["huggingface", "sentence-transformers"]:
            info["model_name"] = self.model_kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        
        return info
    
    def health_check(self) -> bool:
        """
        Check if the embedding service is healthy and can generate embeddings.
        
        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            # Try to generate a test embedding
            test_embedding = self.embed_query("health check test")
            return len(test_embedding) > 0
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of the embedding service."""
        return f"EmbeddingService(model={self.embedding_model}, dimension={self._embedding_dimension})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the embedding service."""
        return (
            f"EmbeddingService(model='{self.embedding_model}', "
            f"dimension={self._embedding_dimension}, "
            f"kwargs_keys={list(self.model_kwargs.keys())})"
        )