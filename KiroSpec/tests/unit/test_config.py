"""
Unit tests for configuration classes.
"""

import pytest
from langchain_vector_db.models.config import VectorDBConfig, SecurityConfig, ObservabilityConfig
from langchain_vector_db.exceptions import ConfigurationException


class TestSecurityConfig:
    """Test cases for SecurityConfig."""
    
    def test_default_security_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        assert config.auth_enabled is True
        assert config.auth_type == "api_key"
        assert config.encryption_enabled is True
        assert config.pii_detection_enabled is True
        
        # Should not raise any exceptions
        config.validate()
    
    def test_invalid_auth_type(self):
        """Test validation with invalid auth type."""
        config = SecurityConfig(auth_type="invalid")
        
        with pytest.raises(ConfigurationException) as exc_info:
            config.validate()
        
        assert "Invalid auth_type" in str(exc_info.value)
    
    def test_jwt_without_secret(self):
        """Test JWT auth type without secret."""
        config = SecurityConfig(auth_type="jwt", jwt_secret=None)
        
        with pytest.raises(ConfigurationException) as exc_info:
            config.validate()
        
        assert "JWT secret is required" in str(exc_info.value)
    
    def test_jwt_with_secret(self):
        """Test JWT auth type with secret."""
        config = SecurityConfig(auth_type="jwt", jwt_secret="test-secret")
        
        # Should not raise any exceptions
        config.validate()
    
    def test_invalid_encryption_algorithm(self):
        """Test validation with invalid encryption algorithm."""
        config = SecurityConfig(encryption_algorithm="INVALID")
        
        with pytest.raises(ConfigurationException) as exc_info:
            config.validate()
        
        assert "Invalid encryption_algorithm" in str(exc_info.value)
    
    def test_invalid_rate_limit(self):
        """Test validation with invalid rate limit."""
        config = SecurityConfig(max_requests_per_minute=0)
        
        with pytest.raises(ConfigurationException) as exc_info:
            config.validate()
        
        assert "max_requests_per_minute must be greater than 0" in str(exc_info.value)


class TestObservabilityConfig:
    """Test cases for ObservabilityConfig."""
    
    def test_default_observability_config(self):
        """Test default observability configuration."""
        config = ObservabilityConfig()
        assert config.log_level == "INFO"
        assert config.log_format == "json"
        assert config.metrics_enabled is True
        assert config.tracing_enabled is True
        
        # Should not raise any exceptions
        config.validate()
    
    def test_invalid_log_level(self):
        """Test validation with invalid log level."""
        config = ObservabilityConfig(log_level="INVALID")
        
        with pytest.raises(ConfigurationException) as exc_info:
            config.validate()
        
        assert "Invalid log_level" in str(exc_info.value)
    
    def test_case_insensitive_log_level(self):
        """Test that log level validation is case insensitive."""
        config = ObservabilityConfig(log_level="debug")
        
        # Should not raise any exceptions
        config.validate()
    
    def test_invalid_log_format(self):
        """Test validation with invalid log format."""
        config = ObservabilityConfig(log_format="invalid")
        
        with pytest.raises(ConfigurationException) as exc_info:
            config.validate()
        
        assert "Invalid log_format" in str(exc_info.value)
    
    def test_invalid_metrics_port(self):
        """Test validation with invalid metrics port."""
        config = ObservabilityConfig(metrics_port=80)  # Below 1024
        
        with pytest.raises(ConfigurationException) as exc_info:
            config.validate()
        
        assert "metrics_port must be between 1024 and 65535" in str(exc_info.value)
    
    def test_invalid_cpu_threshold(self):
        """Test validation with invalid CPU threshold."""
        config = ObservabilityConfig(cpu_threshold_percent=150)  # Above 100
        
        with pytest.raises(ConfigurationException) as exc_info:
            config.validate()
        
        assert "cpu_threshold_percent must be between 0 and 100" in str(exc_info.value)


class TestVectorDBConfig:
    """Test cases for VectorDBConfig."""
    
    def test_minimal_valid_config(self):
        """Test minimal valid configuration."""
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path="./test_db"
        )
        
        # Should create default security and observability configs
        assert config.security is not None
        assert config.observability is not None
        
        # Should not raise any exceptions (assuming OPENAI_API_KEY is set or mocked)
        # config.validate()
    
    def test_invalid_storage_type(self):
        """Test validation with invalid storage type."""
        with pytest.raises(ConfigurationException) as exc_info:
            VectorDBConfig(
                storage_type="invalid",
                embedding_model="openai",
                storage_path="./test_db"
            )
        
        assert "Invalid storage_type" in str(exc_info.value)
    
    def test_invalid_embedding_model(self):
        """Test validation with invalid embedding model."""
        with pytest.raises(ConfigurationException) as exc_info:
            VectorDBConfig(
                storage_type="local",
                embedding_model="invalid",
                storage_path="./test_db"
            )
        
        assert "Invalid embedding_model" in str(exc_info.value)
    
    def test_empty_storage_path(self):
        """Test validation with empty storage path."""
        with pytest.raises(ConfigurationException) as exc_info:
            VectorDBConfig(
                storage_type="local",
                embedding_model="openai",
                storage_path=""
            )
        
        assert "storage_path cannot be empty" in str(exc_info.value)
    
    def test_invalid_chunk_size(self):
        """Test validation with invalid chunk size."""
        with pytest.raises(ConfigurationException) as exc_info:
            VectorDBConfig(
                storage_type="local",
                embedding_model="openai",
                storage_path="./test_db",
                chunk_size=0
            )
        
        assert "chunk_size must be greater than 0" in str(exc_info.value)
    
    def test_invalid_chunk_overlap(self):
        """Test validation with invalid chunk overlap."""
        with pytest.raises(ConfigurationException) as exc_info:
            VectorDBConfig(
                storage_type="local",
                embedding_model="openai",
                storage_path="./test_db",
                chunk_size=1000,
                chunk_overlap=1000  # Equal to chunk_size
            )
        
        assert "chunk_overlap must be less than chunk_size" in str(exc_info.value)
    
    def test_s3_config_validation(self):
        """Test S3-specific configuration validation."""
        # Test with missing region
        with pytest.raises(ConfigurationException) as exc_info:
            VectorDBConfig(
                storage_type="s3",
                embedding_model="openai",
                storage_path="test-bucket",
                aws_region=""
            )
        
        assert "aws_region is required for S3 storage" in str(exc_info.value)
    
    def test_s3_partial_credentials(self):
        """Test S3 configuration with partial credentials."""
        with pytest.raises(ConfigurationException) as exc_info:
            VectorDBConfig(
                storage_type="s3",
                embedding_model="openai",
                storage_path="test-bucket",
                aws_access_key_id="test-key"
                # Missing aws_secret_access_key
            )
        
        assert "Both aws_access_key_id and aws_secret_access_key must be provided together" in str(exc_info.value)
    
    def test_s3_prefix_normalization(self):
        """Test S3 prefix normalization."""
        config = VectorDBConfig(
            storage_type="s3",
            embedding_model="huggingface",
            storage_path="test-bucket",
            s3_prefix="vectors"  # Without trailing slash
        )
        
        # Should add trailing slash
        assert config.s3_prefix == "vectors/"
    
    def test_huggingface_default_model(self):
        """Test HuggingFace embedding model with default model name."""
        config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db"
        )
        
        # Should set default model name
        assert "model_name" in config.model_kwargs
        assert "sentence-transformers" in config.model_kwargs["model_name"]
    
    def test_custom_embedding_model_validation(self):
        """Test custom embedding model validation."""
        with pytest.raises(ConfigurationException) as exc_info:
            VectorDBConfig(
                storage_type="local",
                embedding_model="custom",
                storage_path="./test_db"
                # Missing model_class in model_kwargs
            )
        
        assert "model_class is required for custom embedding models" in str(exc_info.value)
    
    def test_to_dict_redacts_sensitive_data(self):
        """Test that to_dict() redacts sensitive information."""
        config = VectorDBConfig(
            storage_type="s3",
            embedding_model="openai",
            storage_path="test-bucket",
            aws_access_key_id="secret-key",
            aws_secret_access_key="secret-value"
        )
        
        config_dict = config.to_dict()
        
        # Sensitive data should be redacted
        assert config_dict["aws_access_key_id"] == "***REDACTED***"
        assert config_dict["aws_secret_access_key"] == "***REDACTED***"
    
    def test_from_dict_reconstruction(self):
        """Test configuration reconstruction from dictionary."""
        original_config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db",
            chunk_size=500,
            security=SecurityConfig(auth_enabled=False),
            observability=ObservabilityConfig(log_level="DEBUG")
        )
        
        # Convert to dict and back
        config_dict = {
            "storage_type": "local",
            "embedding_model": "huggingface",
            "storage_path": "./test_db",
            "chunk_size": 500,
            "security": {"auth_enabled": False},
            "observability": {"log_level": "DEBUG"}
        }
        
        reconstructed_config = VectorDBConfig.from_dict(config_dict)
        
        assert reconstructed_config.storage_type == original_config.storage_type
        assert reconstructed_config.embedding_model == original_config.embedding_model
        assert reconstructed_config.chunk_size == original_config.chunk_size
        assert reconstructed_config.security.auth_enabled == False
        assert reconstructed_config.observability.log_level == "DEBUG"


if __name__ == "__main__":
    pytest.main([__file__])