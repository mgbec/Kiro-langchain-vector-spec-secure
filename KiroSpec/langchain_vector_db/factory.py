"""
Factory classes for creating vector database components based on configuration.
"""

from typing import Dict, Type, Any
from .models.config import VectorDBConfig
from .storage.interface import VectorStoreInterface
from .storage.local import LocalVectorStore
from .storage.s3 import S3VectorStore
from .services.embedding import EmbeddingService
from .services.document_processor import DocumentProcessor
from .exceptions import ConfigurationException


class VectorStoreFactory:
    """Factory for creating vector store instances based on configuration."""
    
    # Registry of available vector store implementations
    _store_registry: Dict[str, Type[VectorStoreInterface]] = {
        "local": LocalVectorStore,
        "s3": S3VectorStore,
    }
    
    @classmethod
    def create_vector_store(cls, config: VectorDBConfig) -> VectorStoreInterface:
        """
        Create a vector store instance based on configuration.
        
        Args:
            config: Vector database configuration
            
        Returns:
            Vector store instance
            
        Raises:
            ConfigurationException: If storage type is not supported
        """
        storage_type = config.storage_type.lower()
        
        if storage_type not in cls._store_registry:
            available_types = list(cls._store_registry.keys())
            raise ConfigurationException(
                f"Unsupported storage type '{storage_type}'. "
                f"Available types: {available_types}"
            )
        
        store_class = cls._store_registry[storage_type]
        
        try:
            if storage_type == "local":
                return cls._create_local_store(config, store_class)
            elif storage_type == "s3":
                return cls._create_s3_store(config, store_class)
            else:
                # Generic creation for future store types
                return cls._create_generic_store(config, store_class)
        except Exception as e:
            raise ConfigurationException(
                f"Failed to create {storage_type} vector store: {str(e)}"
            )
    
    @classmethod
    def _create_local_store(
        cls, 
        config: VectorDBConfig, 
        store_class: Type[LocalVectorStore]
    ) -> LocalVectorStore:
        """Create a local vector store instance."""
        kwargs = {
            "storage_path": config.storage_path,
            **config.storage_kwargs
        }
        return store_class(**kwargs)
    
    @classmethod
    def _create_s3_store(
        cls, 
        config: VectorDBConfig, 
        store_class: Type[S3VectorStore]
    ) -> S3VectorStore:
        """Create an S3 vector store instance."""
        kwargs = {
            "bucket_name": config.storage_path,
            "s3_prefix": config.s3_prefix,
            "aws_access_key_id": config.aws_access_key_id,
            "aws_secret_access_key": config.aws_secret_access_key,
            "aws_region": config.aws_region,
            **config.storage_kwargs
        }
        return store_class(**kwargs)
    
    @classmethod
    def _create_generic_store(
        cls, 
        config: VectorDBConfig, 
        store_class: Type[VectorStoreInterface]
    ) -> VectorStoreInterface:
        """Create a generic vector store instance."""
        kwargs = {
            "storage_path": config.storage_path,
            **config.storage_kwargs
        }
        return store_class(**kwargs)
    
    @classmethod
    def register_store_type(
        cls, 
        storage_type: str, 
        store_class: Type[VectorStoreInterface]
    ) -> None:
        """
        Register a new vector store type.
        
        Args:
            storage_type: Name of the storage type
            store_class: Vector store class to register
        """
        if not issubclass(store_class, VectorStoreInterface):
            raise ConfigurationException(
                f"Store class must implement VectorStoreInterface"
            )
        
        cls._store_registry[storage_type.lower()] = store_class
    
    @classmethod
    def get_available_store_types(cls) -> list[str]:
        """
        Get list of available vector store types.
        
        Returns:
            List of available storage type names
        """
        return list(cls._store_registry.keys())
    
    @classmethod
    def is_store_type_supported(cls, storage_type: str) -> bool:
        """
        Check if a storage type is supported.
        
        Args:
            storage_type: Storage type to check
            
        Returns:
            True if supported, False otherwise
        """
        return storage_type.lower() in cls._store_registry


class EmbeddingServiceFactory:
    """Factory for creating embedding service instances."""
    
    @classmethod
    def create_embedding_service(cls, config: VectorDBConfig) -> EmbeddingService:
        """
        Create an embedding service instance based on configuration.
        
        Args:
            config: Vector database configuration
            
        Returns:
            Embedding service instance
            
        Raises:
            ConfigurationException: If embedding model is not supported
        """
        try:
            return EmbeddingService(
                embedding_model=config.embedding_model,
                model_kwargs=config.model_kwargs
            )
        except Exception as e:
            raise ConfigurationException(
                f"Failed to create embedding service: {str(e)}"
            )


class DocumentProcessorFactory:
    """Factory for creating document processor instances."""
    
    @classmethod
    def create_document_processor(cls, config: VectorDBConfig) -> DocumentProcessor:
        """
        Create a document processor instance based on configuration.
        
        Args:
            config: Vector database configuration
            
        Returns:
            Document processor instance
            
        Raises:
            ConfigurationException: If processor creation fails
        """
        try:
            return DocumentProcessor(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        except Exception as e:
            raise ConfigurationException(
                f"Failed to create document processor: {str(e)}"
            )


class VectorDatabaseFactory:
    """Main factory for creating complete vector database systems."""
    
    @classmethod
    def create_components(cls, config: VectorDBConfig) -> Dict[str, Any]:
        """
        Create all vector database components based on configuration.
        
        Args:
            config: Vector database configuration
            
        Returns:
            Dictionary containing all created components
            
        Raises:
            ConfigurationException: If component creation fails
        """
        components = {}
        
        try:
            # Create embedding service
            components["embedding_service"] = EmbeddingServiceFactory.create_embedding_service(config)
            
            # Create document processor
            components["document_processor"] = DocumentProcessorFactory.create_document_processor(config)
            
            # Create vector store
            components["vector_store"] = VectorStoreFactory.create_vector_store(config)
            
            return components
            
        except Exception as e:
            # Clean up any partially created components
            for component in components.values():
                if hasattr(component, 'close'):
                    try:
                        component.close()
                    except Exception:
                        pass
            
            raise ConfigurationException(
                f"Failed to create vector database components: {str(e)}"
            )
    
    @classmethod
    def validate_configuration(cls, config: VectorDBConfig) -> Dict[str, bool]:
        """
        Validate configuration for all components.
        
        Args:
            config: Vector database configuration
            
        Returns:
            Dictionary with validation results for each component
        """
        validation_results = {}
        
        # Validate storage type
        validation_results["storage_type"] = VectorStoreFactory.is_store_type_supported(
            config.storage_type
        )
        
        # Validate embedding model (basic check)
        valid_embedding_models = [
            "openai", "huggingface", "sentence-transformers", 
            "cohere", "azure-openai", "custom"
        ]
        validation_results["embedding_model"] = config.embedding_model in valid_embedding_models
        
        # Validate storage path
        validation_results["storage_path"] = bool(config.storage_path)
        
        # Validate chunk parameters
        validation_results["chunk_parameters"] = (
            config.chunk_size > 0 and 
            config.chunk_overlap >= 0 and 
            config.chunk_overlap < config.chunk_size
        )
        
        # Validate S3 specific configuration if needed
        if config.storage_type == "s3":
            validation_results["s3_config"] = cls._validate_s3_config(config)
        
        # Overall validation
        validation_results["overall"] = all(validation_results.values())
        
        return validation_results
    
    @classmethod
    def _validate_s3_config(cls, config: VectorDBConfig) -> bool:
        """Validate S3-specific configuration."""
        # Check required S3 parameters
        if not config.aws_region:
            return False
        
        # Check credential consistency
        has_access_key = bool(config.aws_access_key_id)
        has_secret_key = bool(config.aws_secret_access_key)
        
        if has_access_key != has_secret_key:
            return False  # Both or neither should be provided
        
        return True
    
    @classmethod
    def get_configuration_recommendations(cls, config: VectorDBConfig) -> Dict[str, str]:
        """
        Get configuration recommendations based on the current config.
        
        Args:
            config: Vector database configuration
            
        Returns:
            Dictionary with configuration recommendations
        """
        recommendations = {}
        
        # Storage type recommendations
        if config.storage_type == "local":
            recommendations["storage"] = (
                "Local storage is good for development and small datasets. "
                "Consider S3 for production and larger datasets."
            )
        elif config.storage_type == "s3":
            recommendations["storage"] = (
                "S3 storage provides scalability and durability. "
                "Ensure proper IAM permissions are configured."
            )
        
        # Embedding model recommendations
        if config.embedding_model == "openai":
            recommendations["embedding"] = (
                "OpenAI embeddings provide high quality but require API costs. "
                "Consider rate limiting for high-volume applications."
            )
        elif config.embedding_model == "huggingface":
            recommendations["embedding"] = (
                "HuggingFace embeddings are free but may be slower. "
                "Consider using GPU acceleration for better performance."
            )
        
        # Chunk size recommendations
        if config.chunk_size > 2000:
            recommendations["chunking"] = (
                "Large chunk sizes may reduce search precision. "
                "Consider smaller chunks (500-1500 characters) for better results."
            )
        elif config.chunk_size < 200:
            recommendations["chunking"] = (
                "Very small chunks may lose context. "
                "Consider larger chunks (500-1500 characters) for better context."
            )
        
        return recommendations


class ConfigurationValidator:
    """Utility class for validating vector database configurations."""
    
    @classmethod
    def validate_and_suggest(cls, config: VectorDBConfig) -> Dict[str, Any]:
        """
        Validate configuration and provide suggestions.
        
        Args:
            config: Vector database configuration
            
        Returns:
            Dictionary with validation results and suggestions
        """
        # Perform validation
        validation_results = VectorDatabaseFactory.validate_configuration(config)
        
        # Get recommendations
        recommendations = VectorDatabaseFactory.get_configuration_recommendations(config)
        
        # Identify issues and provide fixes
        issues = []
        fixes = []
        
        if not validation_results["storage_type"]:
            issues.append(f"Unsupported storage type: {config.storage_type}")
            available_types = VectorStoreFactory.get_available_store_types()
            fixes.append(f"Use one of: {available_types}")
        
        if not validation_results["embedding_model"]:
            issues.append(f"Unsupported embedding model: {config.embedding_model}")
            fixes.append("Use: openai, huggingface, sentence-transformers, cohere, azure-openai, or custom")
        
        if not validation_results["storage_path"]:
            issues.append("Storage path is empty")
            fixes.append("Provide a valid storage path or S3 bucket name")
        
        if not validation_results["chunk_parameters"]:
            issues.append("Invalid chunk parameters")
            fixes.append("Ensure chunk_size > 0 and chunk_overlap < chunk_size")
        
        if config.storage_type == "s3" and not validation_results.get("s3_config", True):
            issues.append("Invalid S3 configuration")
            fixes.append("Provide valid AWS region and consistent credentials")
        
        return {
            "is_valid": validation_results["overall"],
            "validation_results": validation_results,
            "recommendations": recommendations,
            "issues": issues,
            "suggested_fixes": fixes
        }