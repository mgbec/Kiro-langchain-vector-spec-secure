"""
Configuration classes for the vector database system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from ..exceptions import ConfigurationException


@dataclass
class SecurityConfig:
    """Security configuration for the vector database."""
    
    # Authentication
    auth_enabled: bool = True
    auth_type: str = "api_key"  # "api_key", "jwt", "oauth"
    api_key_header: str = "X-API-Key"
    jwt_secret: Optional[str] = None
    
    # Authorization
    rbac_enabled: bool = True
    default_role: str = "reader"
    
    # Encryption
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None
    encryption_algorithm: str = "AES-256-GCM"
    
    # Data protection
    pii_detection_enabled: bool = True
    data_masking_enabled: bool = False
    
    # Security monitoring
    audit_logging_enabled: bool = True
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 100
    
    def validate(self) -> None:
        """Validate security configuration parameters."""
        valid_auth_types = ["api_key", "jwt", "oauth"]
        if self.auth_type not in valid_auth_types:
            raise ConfigurationException(
                f"Invalid auth_type '{self.auth_type}'. Must be one of: {valid_auth_types}"
            )
        
        if self.auth_enabled and self.auth_type == "jwt" and not self.jwt_secret:
            raise ConfigurationException(
                "JWT secret is required when auth_type is 'jwt'"
            )
        
        valid_algorithms = ["AES-256-GCM", "AES-256-CBC", "ChaCha20-Poly1305"]
        if self.encryption_algorithm not in valid_algorithms:
            raise ConfigurationException(
                f"Invalid encryption_algorithm '{self.encryption_algorithm}'. "
                f"Must be one of: {valid_algorithms}"
            )
        
        if self.max_requests_per_minute <= 0:
            raise ConfigurationException(
                "max_requests_per_minute must be greater than 0"
            )


@dataclass
class ObservabilityConfig:
    """Observability configuration for monitoring and logging."""
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    
    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 8080
    metrics_endpoint: str = "/metrics"
    
    # Tracing
    tracing_enabled: bool = True
    tracing_service_name: str = "vector-database"
    tracing_endpoint: Optional[str] = None
    
    # Health checks
    health_check_enabled: bool = True
    health_check_port: int = 8081
    health_check_endpoint: str = "/health"
    
    # Performance monitoring
    performance_monitoring_enabled: bool = True
    memory_threshold_mb: int = 1000
    cpu_threshold_percent: float = 80.0
    
    def validate(self) -> None:
        """Validate observability configuration parameters."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigurationException(
                f"Invalid log_level '{self.log_level}'. Must be one of: {valid_log_levels}"
            )
        
        valid_log_formats = ["json", "text", "structured"]
        if self.log_format not in valid_log_formats:
            raise ConfigurationException(
                f"Invalid log_format '{self.log_format}'. Must be one of: {valid_log_formats}"
            )
        
        if self.metrics_port < 1024 or self.metrics_port > 65535:
            raise ConfigurationException(
                "metrics_port must be between 1024 and 65535"
            )
        
        if self.health_check_port < 1024 or self.health_check_port > 65535:
            raise ConfigurationException(
                "health_check_port must be between 1024 and 65535"
            )
        
        if self.memory_threshold_mb <= 0:
            raise ConfigurationException(
                "memory_threshold_mb must be greater than 0"
            )
        
        if not 0 < self.cpu_threshold_percent <= 100:
            raise ConfigurationException(
                "cpu_threshold_percent must be between 0 and 100"
            )


@dataclass
class VectorDBConfig:
    """Main configuration class for the vector database system."""
    
    # Core configuration
    storage_type: str  # "local" or "s3"
    embedding_model: str  # "openai", "huggingface", etc.
    storage_path: str  # Local path or S3 bucket name
    
    # Embedding model specific config
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Storage specific config
    storage_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Document processing config
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # AWS S3 specific config (when storage_type="s3")
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_prefix: str = "vectors/"
    
    # Security config
    security: Optional[SecurityConfig] = None
    
    # Observability config
    observability: Optional[ObservabilityConfig] = None
    
    def __post_init__(self):
        """Initialize default configurations and validate."""
        if self.security is None:
            self.security = SecurityConfig()
        
        if self.observability is None:
            self.observability = ObservabilityConfig()
        
        self.validate()
    
    def validate(self) -> None:
        """Validate the complete configuration."""
        # Validate core configuration
        valid_storage_types = ["local", "s3"]
        if self.storage_type not in valid_storage_types:
            raise ConfigurationException(
                f"Invalid storage_type '{self.storage_type}'. Must be one of: {valid_storage_types}"
            )
        
        valid_embedding_models = [
            "openai", "huggingface", "sentence-transformers", 
            "cohere", "azure-openai", "custom"
        ]
        if self.embedding_model not in valid_embedding_models:
            raise ConfigurationException(
                f"Invalid embedding_model '{self.embedding_model}'. "
                f"Must be one of: {valid_embedding_models}"
            )
        
        if not self.storage_path:
            raise ConfigurationException("storage_path cannot be empty")
        
        # Validate document processing parameters
        if self.chunk_size <= 0:
            raise ConfigurationException("chunk_size must be greater than 0")
        
        if self.chunk_overlap < 0:
            raise ConfigurationException("chunk_overlap cannot be negative")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationException(
                "chunk_overlap must be less than chunk_size"
            )
        
        # Validate S3 specific configuration
        if self.storage_type == "s3":
            self._validate_s3_config()
        
        # Validate embedding model specific configuration
        self._validate_embedding_config()
        
        # Validate sub-configurations
        if self.security:
            self.security.validate()
        
        if self.observability:
            self.observability.validate()
    
    def _validate_s3_config(self) -> None:
        """Validate S3-specific configuration parameters."""
        if not self.aws_region:
            raise ConfigurationException("aws_region is required for S3 storage")
        
        # Check if credentials are provided (either directly or via environment/IAM)
        if not self.aws_access_key_id and not self.aws_secret_access_key:
            # This is okay - boto3 can use environment variables or IAM roles
            pass
        elif bool(self.aws_access_key_id) != bool(self.aws_secret_access_key):
            raise ConfigurationException(
                "Both aws_access_key_id and aws_secret_access_key must be provided together"
            )
        
        if not self.s3_prefix.endswith("/"):
            self.s3_prefix += "/"
    
    def _validate_embedding_config(self) -> None:
        """Validate embedding model specific configuration."""
        if self.embedding_model == "openai":
            # OpenAI specific validation
            if "api_key" not in self.model_kwargs and not self._has_openai_env_key():
                raise ConfigurationException(
                    "OpenAI API key must be provided in model_kwargs or OPENAI_API_KEY environment variable"
                )
        
        elif self.embedding_model == "huggingface":
            # HuggingFace specific validation
            if "model_name" not in self.model_kwargs:
                # Set default model if not specified
                self.model_kwargs["model_name"] = "sentence-transformers/all-MiniLM-L6-v2"
        
        elif self.embedding_model == "custom":
            # Custom embedding model validation
            if "model_class" not in self.model_kwargs:
                raise ConfigurationException(
                    "model_class is required for custom embedding models"
                )
    
    def _has_openai_env_key(self) -> bool:
        """Check if OpenAI API key is available in environment variables."""
        import os
        return bool(os.getenv("OPENAI_API_KEY"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            "storage_type": self.storage_type,
            "embedding_model": self.embedding_model,
            "storage_path": self.storage_path,
            "model_kwargs": self.model_kwargs,
            "storage_kwargs": self.storage_kwargs,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "aws_region": self.aws_region,
            "s3_prefix": self.s3_prefix,
        }
        
        # Only include AWS credentials if they exist (for security)
        if self.aws_access_key_id:
            config_dict["aws_access_key_id"] = "***REDACTED***"
        if self.aws_secret_access_key:
            config_dict["aws_secret_access_key"] = "***REDACTED***"
        
        if self.security:
            config_dict["security"] = self._security_to_dict()
        
        if self.observability:
            config_dict["observability"] = self._observability_to_dict()
        
        return config_dict
    
    def _security_to_dict(self) -> Dict[str, Any]:
        """Convert security config to dictionary with sensitive data redacted."""
        return {
            "auth_enabled": self.security.auth_enabled,
            "auth_type": self.security.auth_type,
            "rbac_enabled": self.security.rbac_enabled,
            "encryption_enabled": self.security.encryption_enabled,
            "pii_detection_enabled": self.security.pii_detection_enabled,
            "audit_logging_enabled": self.security.audit_logging_enabled,
            "rate_limiting_enabled": self.security.rate_limiting_enabled,
            "max_requests_per_minute": self.security.max_requests_per_minute,
            # Sensitive fields are redacted
            "jwt_secret": "***REDACTED***" if self.security.jwt_secret else None,
            "encryption_key": "***REDACTED***" if self.security.encryption_key else None,
        }
    
    def _observability_to_dict(self) -> Dict[str, Any]:
        """Convert observability config to dictionary."""
        return {
            "log_level": self.observability.log_level,
            "log_format": self.observability.log_format,
            "log_file": self.observability.log_file,
            "metrics_enabled": self.observability.metrics_enabled,
            "metrics_port": self.observability.metrics_port,
            "tracing_enabled": self.observability.tracing_enabled,
            "health_check_enabled": self.observability.health_check_enabled,
            "performance_monitoring_enabled": self.observability.performance_monitoring_enabled,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorDBConfig":
        """Create configuration from dictionary."""
        # Extract security config if present
        security_config = None
        if "security" in config_dict:
            security_dict = config_dict.pop("security")
            security_config = SecurityConfig(**security_dict)
        
        # Extract observability config if present
        observability_config = None
        if "observability" in config_dict:
            observability_dict = config_dict.pop("observability")
            observability_config = ObservabilityConfig(**observability_dict)
        
        # Create main config
        config = cls(**config_dict)
        config.security = security_config
        config.observability = observability_config
        
        return config