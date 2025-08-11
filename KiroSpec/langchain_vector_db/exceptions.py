"""
Exception hierarchy for the vector database system.
"""

from typing import Optional


class VectorDBException(Exception):
    """Base exception for vector database operations."""
    
    def __init__(self, message: str, correlation_id: Optional[str] = None):
        super().__init__(message)
        self.correlation_id = correlation_id


class EmbeddingException(VectorDBException):
    """Raised when embedding generation fails."""
    pass


class StorageException(VectorDBException):
    """Raised when storage operations fail."""
    pass


class ConfigurationException(VectorDBException):
    """Raised when configuration is invalid."""
    pass


class S3Exception(StorageException):
    """Raised when S3 operations fail."""
    pass


class SecurityException(VectorDBException):
    """Raised when security operations fail."""
    pass


class AuthenticationException(SecurityException):
    """Raised when authentication fails."""
    pass


class AuthorizationException(SecurityException):
    """Raised when authorization fails."""
    pass


class EncryptionException(SecurityException):
    """Raised when encryption/decryption fails."""
    pass


class ObservabilityException(VectorDBException):
    """Raised when observability operations fail."""
    pass