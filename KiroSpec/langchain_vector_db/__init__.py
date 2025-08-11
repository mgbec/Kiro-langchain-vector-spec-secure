"""
LangChain Vector Database - A comprehensive vector database system with security and observability.

This package provides a unified interface for document ingestion, vector storage, and semantic search
operations with support for multiple storage backends, comprehensive security controls, and full
observability capabilities.
"""

from .manager import VectorDatabaseManager
from .models import (
    VectorDBConfig, Document, SecurityConfig, ObservabilityConfig,
    AuthToken, AuditEvent, HealthStatus, SystemMetrics, PIIMatch
)
from .factory import (
    VectorStoreFactory,
    EmbeddingServiceFactory,
    DocumentProcessorFactory,
    VectorDatabaseFactory,
    ConfigurationValidator
)
from .exceptions import (
    VectorDBException,
    EmbeddingException,
    StorageException,
    ConfigurationException,
    SecurityException,
    ObservabilityException
)

__version__ = "0.1.0"
__author__ = "LangChain Vector DB Team"

__all__ = [
    "VectorDatabaseManager",
    "VectorDBConfig",
    "Document",
    "SecurityConfig",
    "ObservabilityConfig",
    "AuthToken",
    "AuditEvent", 
    "HealthStatus",
    "SystemMetrics",
    "PIIMatch",
    "VectorStoreFactory",
    "EmbeddingServiceFactory",
    "DocumentProcessorFactory",
    "VectorDatabaseFactory",
    "ConfigurationValidator",
    "VectorDBException",
    "EmbeddingException",
    "StorageException",
    "ConfigurationException",
    "SecurityException",
    "ObservabilityException",
]