"""
Unit tests for SecurityManager.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch

from langchain_vector_db.services.security import SecurityManager
from langchain_vector_db.models.config import SecurityConfig
from langchain_vector_db.models.auth import AuthToken
from langchain_vector_db.exceptions import (
    SecurityException,
    AuthenticationException,
    AuthorizationException,
    ConfigurationException
)


class TestSecurityManager:
    """Test cases for SecurityManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SecurityConfig(
            auth_enabled=True,
            auth_type="api_key",
            rbac_enabled=True,
            default_role="reader"
        )
        
        self.security_manager = SecurityManager(self.config)
    
    def test_initialization_with_defaults(self):
        """Test security manager initialization with default config."""
        config = SecurityConfig()
        manager = SecurityManager(config)
        
        assert manager.config == config
        assert len(manager._active_tokens) == 0
        assert len(manager._api_keys) == 0
    
    def test_initialization_with_jwt_generates_secret(self):
        """Test that JWT secret is generated if not provided."""
        config = SecurityConfig(auth_type="jwt")
        manager = SecurityManager(config)
        
        assert manager.config.jwt_secret is not None
        assert len(manager.config.jwt_secret) > 0
    
    def test_invalid_auth_type_raises_exception(self):
        """Test that invalid auth type raises exception."""
        config = SecurityConfig(auth_type="invalid")
        
        with pytest.raises(ConfigurationException) as exc_info:
            SecurityManager(config)
        
        assert "Unsupported auth type" in str(exc_info.value)
    
    def test_create_api_key(self):
        """Test creating an API key."""
        api_key = self.security_manager.create_api_key(
            user_id="test_user",
            roles=["writer"],
            description="Test key"
        )
        
        assert isinstance(api_key, str)
        assert len(api_key) > 0
        assert api_key in self.security_manager._api_keys
        
        key_info = self.security_manager._api_keys[api_key]
        assert key_info["user_id"] == "test_user"
        assert key_info["roles"] == ["writer"]
        assert key_info["description"] == "Test key"
        assert key_info["enabled"] is True
    
    def test_authenticate_api_key_success(self):
        """Test successful API key authentication."""
        api_key = self.security_manager.create_api_key(
            user_id="test_user",
            roles=["writer"]
        )
        
        token = self.security_manager.authenticate_api_key(api_key)
        
        assert isinstance(token, AuthToken)
        assert token.user_id == "test_user"
        assert "writer" in token.roles
        assert "documents.create" in token.permissions
        assert not token.is_expired()
        assert token.correlation_id in self.security_manager._active_tokens
    
    def test_authenticate_api_key_invalid(self):
        """Test authentication with invalid API key."""
        with pytest.raises(AuthenticationException) as exc_info:
            self.security_manager.authenticate_api_key("invalid_key")
        
        assert "Invalid API key" in str(exc_info.value)
    
    def test_authenticate_api_key_expired(self):
        """Test authentication with expired API key."""
        # Create expired key
        api_key = self.security_manager.create_api_key(
            user_id="test_user",
            roles=["reader"],
            expires_at=datetime.utcnow() - timedelta(hours=1)  # Expired
        )
        
        with pytest.raises(AuthenticationException) as exc_info:
            self.security_manager.authenticate_api_key(api_key)
        
        assert "API key expired" in str(exc_info.value)
    
    def test_authenticate_api_key_disabled(self):
        """Test authentication with disabled API key."""
        api_key = self.security_manager.create_api_key(
            user_id="test_user",
            roles=["reader"]
        )
        
        # Disable the key
        self.security_manager._api_keys[api_key]["enabled"] = False
        
        with pytest.raises(AuthenticationException) as exc_info:
            self.security_manager.authenticate_api_key(api_key)
        
        assert "API key disabled" in str(exc_info.value)
    
    def test_create_jwt_token(self):
        """Test creating a JWT token."""
        jwt_config = SecurityConfig(auth_type="jwt", jwt_secret="test_secret")
        manager = SecurityManager(jwt_config)
        
        jwt_token = manager.create_jwt_token(
            user_id="test_user",
            roles=["writer"],
            expires_in_hours=1
        )
        
        assert isinstance(jwt_token, str)
        assert len(jwt_token) > 0
    
    def test_authenticate_jwt_success(self):
        """Test successful JWT authentication."""
        jwt_config = SecurityConfig(auth_type="jwt", jwt_secret="test_secret")
        manager = SecurityManager(jwt_config)
        
        jwt_token = manager.create_jwt_token(
            user_id="test_user",
            roles=["writer"]
        )
        
        token = manager.authenticate_jwt(jwt_token)
        
        assert isinstance(token, AuthToken)
        assert token.user_id == "test_user"
        assert "writer" in token.roles
        assert "documents.create" in token.permissions
    
    def test_authenticate_jwt_invalid(self):
        """Test authentication with invalid JWT."""
        jwt_config = SecurityConfig(auth_type="jwt", jwt_secret="test_secret")
        manager = SecurityManager(jwt_config)
        
        with pytest.raises(AuthenticationException) as exc_info:
            manager.authenticate_jwt("invalid.jwt.token")
        
        assert "Invalid JWT token" in str(exc_info.value)
    
    def test_authenticate_disabled_auth(self):
        """Test authentication when auth is disabled."""
        config = SecurityConfig(auth_enabled=False)
        manager = SecurityManager(config)
        
        # Should return default token
        token = manager.authenticate_api_key("any_key")
        
        assert token.user_id == "anonymous"
        assert manager.config.default_role in token.roles
    
    def test_authorize_operation_success(self):
        """Test successful operation authorization."""
        api_key = self.security_manager.create_api_key(
            user_id="test_user",
            roles=["writer"]
        )
        token = self.security_manager.authenticate_api_key(api_key)
        
        # Writer should be able to create documents
        result = self.security_manager.authorize_operation(
            token, "documents.create", "test_doc"
        )
        
        assert result is True
    
    def test_authorize_operation_insufficient_permissions(self):
        """Test authorization with insufficient permissions."""
        api_key = self.security_manager.create_api_key(
            user_id="test_user",
            roles=["reader"]  # Reader can't manage system
        )
        token = self.security_manager.authenticate_api_key(api_key)
        
        with pytest.raises(AuthorizationException) as exc_info:
            self.security_manager.authorize_operation(token, "system.manage")
        
        assert "Insufficient permissions" in str(exc_info.value)
    
    def test_authorize_operation_expired_token(self):
        """Test authorization with expired token."""
        # Create token that's already expired
        token = AuthToken(
            user_id="test_user",
            roles=["writer"],
            permissions=["documents.create"],
            expires_at=datetime.utcnow() - timedelta(hours=1),  # Expired
            correlation_id="test_correlation"
        )
        
        with pytest.raises(AuthorizationException) as exc_info:
            self.security_manager.authorize_operation(token, "documents.create")
        
        assert "Token expired" in str(exc_info.value)
    
    def test_authorize_operation_inactive_token(self):
        """Test authorization with inactive token."""
        token = AuthToken(
            user_id="test_user",
            roles=["writer"],
            permissions=["documents.create"],
            expires_at=datetime.utcnow() + timedelta(hours=1),
            correlation_id="inactive_correlation"
        )
        
        # Token is not in active tokens
        with pytest.raises(AuthorizationException) as exc_info:
            self.security_manager.authorize_operation(token, "documents.create")
        
        assert "Token not active" in str(exc_info.value)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        config = SecurityConfig(
            rate_limiting_enabled=True,
            max_requests_per_minute=2
        )
        manager = SecurityManager(config)
        
        api_key = manager.create_api_key("test_user", ["reader"])
        token = manager.authenticate_api_key(api_key)
        
        # First two requests should succeed
        assert manager.authorize_operation(token, "documents.read") is True
        assert manager.authorize_operation(token, "documents.read") is True
        
        # Third request should fail due to rate limiting
        with pytest.raises(AuthorizationException) as exc_info:
            manager.authorize_operation(token, "documents.read")
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    def test_rate_limiting_disabled(self):
        """Test that rate limiting can be disabled."""
        config = SecurityConfig(rate_limiting_enabled=False)
        manager = SecurityManager(config)
        
        api_key = manager.create_api_key("test_user", ["reader"])
        token = manager.authenticate_api_key(api_key)
        
        # Should be able to make many requests
        for _ in range(10):
            assert manager.authorize_operation(token, "documents.read") is True
    
    def test_revoke_api_key(self):
        """Test revoking an API key."""
        api_key = self.security_manager.create_api_key(
            user_id="test_user",
            roles=["reader"]
        )
        
        # Key should exist
        assert api_key in self.security_manager._api_keys
        
        # Revoke key
        result = self.security_manager.revoke_api_key(api_key)
        assert result is True
        
        # Key should no longer exist
        assert api_key not in self.security_manager._api_keys
        
        # Revoking again should return False
        result = self.security_manager.revoke_api_key(api_key)
        assert result is False
    
    def test_invalidate_token(self):
        """Test invalidating an active token."""
        api_key = self.security_manager.create_api_key("test_user", ["reader"])
        token = self.security_manager.authenticate_api_key(api_key)
        
        # Token should be active
        assert token.correlation_id in self.security_manager._active_tokens
        
        # Invalidate token
        result = self.security_manager.invalidate_token(token.correlation_id)
        assert result is True
        
        # Token should no longer be active
        assert token.correlation_id not in self.security_manager._active_tokens
        
        # Invalidating again should return False
        result = self.security_manager.invalidate_token(token.correlation_id)
        assert result is False
    
    def test_get_active_tokens(self):
        """Test getting active tokens."""
        # Create multiple tokens
        api_key1 = self.security_manager.create_api_key("user1", ["reader"])
        api_key2 = self.security_manager.create_api_key("user2", ["writer"])
        
        token1 = self.security_manager.authenticate_api_key(api_key1)
        token2 = self.security_manager.authenticate_api_key(api_key2)
        
        active_tokens = self.security_manager.get_active_tokens()
        
        assert len(active_tokens) == 2
        
        # Check token information
        token_users = [t["user_id"] for t in active_tokens]
        assert "user1" in token_users
        assert "user2" in token_users
    
    def test_cleanup_expired_tokens(self):
        """Test cleaning up expired tokens."""
        # Create expired token
        expired_token = AuthToken(
            user_id="expired_user",
            roles=["reader"],
            permissions=["documents.read"],
            expires_at=datetime.utcnow() - timedelta(hours=1),
            correlation_id="expired_correlation"
        )
        
        # Add to active tokens
        self.security_manager._active_tokens["expired_correlation"] = expired_token
        
        # Create valid token
        api_key = self.security_manager.create_api_key("valid_user", ["reader"])
        valid_token = self.security_manager.authenticate_api_key(api_key)
        
        # Should have 2 tokens
        assert len(self.security_manager._active_tokens) == 2
        
        # Cleanup expired tokens
        cleaned_count = self.security_manager.cleanup_expired_tokens()
        
        assert cleaned_count == 1
        assert len(self.security_manager._active_tokens) == 1
        assert valid_token.correlation_id in self.security_manager._active_tokens
    
    def test_audit_logging(self):
        """Test audit logging functionality."""
        # Enable audit logging
        self.security_manager.config.audit_logging_enabled = True
        
        # Perform some operations
        api_key = self.security_manager.create_api_key("test_user", ["writer"])
        token = self.security_manager.authenticate_api_key(api_key)
        self.security_manager.authorize_operation(token, "documents.create")
        
        # Get audit events
        events = self.security_manager.get_audit_events()
        
        assert len(events) >= 3  # create_api_key, authenticate, authorize
        
        # Check event structure
        for event in events:
            assert "timestamp" in event
            assert "user_id" in event
            assert "operation" in event
            assert "status" in event
    
    def test_audit_logging_disabled(self):
        """Test that audit logging can be disabled."""
        # Disable audit logging
        self.security_manager.config.audit_logging_enabled = False
        
        # Perform operations
        api_key = self.security_manager.create_api_key("test_user", ["writer"])
        
        # Should have no audit events
        events = self.security_manager.get_audit_events()
        assert len(events) == 0
    
    def test_get_audit_events_with_filters(self):
        """Test getting audit events with filters."""
        self.security_manager.config.audit_logging_enabled = True
        
        # Create events for different users
        api_key1 = self.security_manager.create_api_key("user1", ["reader"])
        api_key2 = self.security_manager.create_api_key("user2", ["writer"])
        
        # Filter by user
        events = self.security_manager.get_audit_events(user_id="user1")
        user_ids = [e["user_id"] for e in events]
        assert all(uid == "user1" for uid in user_ids if uid != "system")
        
        # Filter by operation
        events = self.security_manager.get_audit_events(operation="api_key")
        operations = [e["operation"] for e in events]
        assert all(op == "api_key" for op in operations)
    
    def test_get_security_stats(self):
        """Test getting security statistics."""
        # Create some activity
        api_key = self.security_manager.create_api_key("test_user", ["writer"])
        token = self.security_manager.authenticate_api_key(api_key)
        
        stats = self.security_manager.get_security_stats()
        
        assert "auth_enabled" in stats
        assert "auth_type" in stats
        assert "active_tokens" in stats
        assert "total_api_keys" in stats
        assert stats["auth_enabled"] is True
        assert stats["auth_type"] == "api_key"
        assert stats["active_tokens"] >= 1
        assert stats["total_api_keys"] >= 1
    
    def test_health_check(self):
        """Test security health check."""
        result = self.security_manager.health_check()
        assert result is True
    
    def test_health_check_jwt(self):
        """Test security health check with JWT."""
        jwt_config = SecurityConfig(auth_type="jwt", jwt_secret="test_secret")
        manager = SecurityManager(jwt_config)
        
        result = manager.health_check()
        assert result is True
    
    def test_default_permissions(self):
        """Test default role permissions."""
        # Test admin permissions
        admin_perms = self.security_manager._get_permissions_for_roles(["admin"])
        assert "system.manage" in admin_perms
        assert "documents.create" in admin_perms
        assert "users.manage" in admin_perms
        
        # Test writer permissions
        writer_perms = self.security_manager._get_permissions_for_roles(["writer"])
        assert "documents.create" in writer_perms
        assert "documents.read" in writer_perms
        assert "system.manage" not in writer_perms
        
        # Test reader permissions
        reader_perms = self.security_manager._get_permissions_for_roles(["reader"])
        assert "documents.read" in reader_perms
        assert "search.query" in reader_perms
        assert "documents.create" not in reader_perms
        
        # Test viewer permissions
        viewer_perms = self.security_manager._get_permissions_for_roles(["viewer"])
        assert "system.status" in viewer_perms
        assert "documents.read" not in viewer_perms
    
    def test_string_representations(self):
        """Test string representations of security manager."""
        str_repr = str(self.security_manager)
        assert "SecurityManager" in str_repr
        assert "auth_enabled=True" in str_repr
        assert "api_key" in str_repr
        
        repr_str = repr(self.security_manager)
        assert "SecurityManager" in repr_str
        assert "rbac_enabled" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])