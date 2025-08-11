"""
Security manager for authentication, authorization, and data protection.
"""

import hashlib
import hmac
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import jwt
import bcrypt

from ..models.config import SecurityConfig
from ..models.auth import AuthToken, AuditEvent
from ..models.pii import PIIMatch
from .encryption import EncryptionService
from .pii_detection import PIIDetectionService
from .security_monitoring import SecurityMonitoringService
from ..exceptions import (
    SecurityException,
    AuthenticationException,
    AuthorizationException,
    ConfigurationException,
    EncryptionException
)


class SecurityManager:
    """Manages authentication, authorization, and security operations."""
    
    # Default role permissions
    DEFAULT_PERMISSIONS = {
        "admin": {
            "documents.create", "documents.read", "documents.update", "documents.delete",
            "search.query", "system.manage", "system.backup", "system.restore",
            "users.manage", "security.manage"
        },
        "writer": {
            "documents.create", "documents.read", "documents.update", "documents.delete",
            "search.query"
        },
        "reader": {
            "documents.read", "search.query"
        },
        "viewer": {
            "system.status", "system.health"
        }
    }
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize the security manager.
        
        Args:
            config: Security configuration
        """
        self.config = config
        
        # Validate configuration
        self._validate_config()
        
        # Active sessions and API keys
        self._active_tokens: Dict[str, AuthToken] = {}
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._rate_limits: Dict[str, List[float]] = {}
        
        # Audit log storage
        self._audit_events: List[AuditEvent] = []
        
        # Initialize encryption service if enabled
        self.encryption_service = None
        if self.config.encryption_enabled:
            self.encryption_service = EncryptionService(
                encryption_key=self.config.encryption_key,
                algorithm=self.config.encryption_algorithm
            )
        
        # Initialize PII detection service if enabled
        self.pii_detection_service = None
        if self.config.pii_detection_enabled:
            self.pii_detection_service = PIIDetectionService(
                enable_ml_detection=True  # Enable ML detection if available
            )
        
        # Initialize security monitoring service
        self.monitoring_service = SecurityMonitoringService(
            enable_monitoring=self.config.audit_logging_enabled
        )
        
        # Initialize rate limiting service if enabled
        self.rate_limiting_service = None
        if self.config.rate_limiting_enabled:
            from .rate_limiter import RateLimitingService, RateLimitRule, RateLimitType
            
            # Create default rate limiting rules
            default_rules = [
                RateLimitRule(
                    name="user_requests",
                    limit_type=RateLimitType.PER_USER,
                    max_requests=self.config.max_requests_per_minute,
                    window_seconds=60,
                    description="Per-user request limit per minute"
                ),
                RateLimitRule(
                    name="auth_attempts",
                    limit_type=RateLimitType.PER_USER,
                    max_requests=5,
                    window_seconds=300,  # 5 minutes
                    operation="authentication",
                    description="Authentication attempts per user"
                )
            ]
            
            self.rate_limiting_service = RateLimitingService(
                enable_rate_limiting=True,
                default_rules=default_rules
            )
        
        # Initialize JWT secret if needed
        if self.config.auth_type == "jwt" and not self.config.jwt_secret:
            self.config.jwt_secret = self._generate_secret()
    
    def _validate_config(self) -> None:
        """Validate security configuration."""
        if self.config.auth_enabled:
            if self.config.auth_type == "jwt" and not self.config.jwt_secret:
                # Will be generated automatically
                pass
            elif self.config.auth_type not in ["api_key", "jwt", "oauth"]:
                raise ConfigurationException(f"Unsupported auth type: {self.config.auth_type}")
        
        if self.config.max_requests_per_minute <= 0:
            raise ConfigurationException("max_requests_per_minute must be positive")
    
    def _generate_secret(self) -> str:
        """Generate a secure random secret."""
        return secrets.token_urlsafe(32)
    
    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        return str(uuid.uuid4())
    
    def authenticate_api_key(self, api_key: str, user_id: str = None, ip_address: str = None) -> AuthToken:
        """
        Authenticate using API key.
        
        Args:
            api_key: API key to authenticate
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
            
        Returns:
            Authentication token
            
        Raises:
            AuthenticationException: If authentication fails
        """
        # Check rate limits first
        if self.rate_limiting_service and user_id:
            allowed, violations = self.rate_limiting_service.check_rate_limit(
                user_id=user_id,
                ip_address=ip_address,
                operation="authentication"
            )
            
            if not allowed:
                self._audit_log("authentication", "api_key", "failed", 
                              {"reason": "rate_limit_exceeded", "violations": len(violations)})
                raise AuthenticationException("Rate limit exceeded for authentication attempts")
        
        if not self.config.auth_enabled:
            # Create a default token for unauthenticated access
            return self._create_default_token()
        
        if api_key not in self._api_keys:
            self._audit_log("authentication", "api_key", "failed", 
                          {"reason": "invalid_api_key"})
            raise AuthenticationException("Invalid API key")
        
        key_info = self._api_keys[api_key]
        
        # Check if key is expired
        if key_info.get("expires_at") and datetime.utcnow() > key_info["expires_at"]:
            self._audit_log("authentication", "api_key", "failed", 
                          {"reason": "expired_api_key"})
            raise AuthenticationException("API key expired")
        
        # Check if key is disabled
        if not key_info.get("enabled", True):
            self._audit_log("authentication", "api_key", "failed", 
                          {"reason": "disabled_api_key"})
            raise AuthenticationException("API key disabled")
        
        # Create auth token
        token = AuthToken(
            user_id=key_info["user_id"],
            roles=key_info.get("roles", [self.config.default_role]),
            permissions=self._get_permissions_for_roles(key_info.get("roles", [self.config.default_role])),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            correlation_id=self._generate_correlation_id()
        )
        
        # Store active token
        self._active_tokens[token.correlation_id] = token
        
        self._audit_log("authentication", "api_key", "success", 
                      {"user_id": token.user_id, "roles": token.roles})
        
        return token
    
    def authenticate_jwt(self, jwt_token: str, user_id: str = None, ip_address: str = None) -> AuthToken:
        """
        Authenticate using JWT token.
        
        Args:
            jwt_token: JWT token to authenticate
            user_id: User ID for rate limiting (optional, will be extracted from JWT)
            ip_address: IP address for rate limiting
            
        Returns:
            Authentication token
            
        Raises:
            AuthenticationException: If authentication fails
        """
        if not self.config.auth_enabled:
            return self._create_default_token()
        
        try:
            # Decode JWT token
            payload = jwt.decode(
                jwt_token, 
                self.config.jwt_secret, 
                algorithms=["HS256"]
            )
            
            # Extract user information
            extracted_user_id = payload.get("user_id")
            roles = payload.get("roles", [self.config.default_role])
            expires_at = datetime.fromtimestamp(payload.get("exp", time.time() + 3600))
            
            if not extracted_user_id:
                raise AuthenticationException("Invalid JWT token: missing user_id")
            
            # Use extracted user_id for rate limiting if not provided
            rate_limit_user_id = user_id or extracted_user_id
            
            # Check rate limits
            if self.rate_limiting_service:
                allowed, violations = self.rate_limiting_service.check_rate_limit(
                    user_id=rate_limit_user_id,
                    ip_address=ip_address,
                    operation="authentication"
                )
                
                if not allowed:
                    self._audit_log("authentication", "jwt", "failed", 
                                  {"reason": "rate_limit_exceeded", "violations": len(violations)})
                    raise AuthenticationException("Rate limit exceeded for authentication attempts")
            
            # Create auth token
            token = AuthToken(
                user_id=extracted_user_id,
                roles=roles,
                permissions=self._get_permissions_for_roles(roles),
                expires_at=expires_at,
                correlation_id=self._generate_correlation_id()
            )
            
            # Store active token
            self._active_tokens[token.correlation_id] = token
            
            self._audit_log("authentication", "jwt", "success", 
                          {"user_id": token.user_id, "roles": token.roles})
            
            return token
            
        except jwt.ExpiredSignatureError:
            self._audit_log("authentication", "jwt", "failed", 
                          {"reason": "expired_token"})
            raise AuthenticationException("JWT token expired")
        except jwt.InvalidTokenError as e:
            self._audit_log("authentication", "jwt", "failed", 
                          {"reason": "invalid_token", "error": str(e)})
            raise AuthenticationException(f"Invalid JWT token: {str(e)}")
    
    def _create_default_token(self) -> AuthToken:
        """Create a default token for unauthenticated access."""
        return AuthToken(
            user_id="anonymous",
            roles=[self.config.default_role],
            permissions=self._get_permissions_for_roles([self.config.default_role]),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            correlation_id=self._generate_correlation_id()
        )
    
    def _get_permissions_for_roles(self, roles: List[str]) -> List[str]:
        """Get permissions for a list of roles."""
        permissions = set()
        for role in roles:
            if role in self.DEFAULT_PERMISSIONS:
                permissions.update(self.DEFAULT_PERMISSIONS[role])
        return list(permissions)
    
    def check_rate_limit(
        self,
        user_id: str,
        operation: str = None,
        ip_address: str = None
    ) -> bool:
        """
        Check if operation is within rate limits.
        
        Args:
            user_id: User identifier
            operation: Operation being performed
            ip_address: IP address
            
        Returns:
            True if operation is allowed
            
        Raises:
            SecurityException: If rate limit is exceeded
        """
        if not self.rate_limiting_service:
            return True
        
        allowed, violations = self.rate_limiting_service.check_rate_limit(
            user_id=user_id,
            ip_address=ip_address,
            operation=operation
        )
        
        if not allowed:
            # Log rate limit violations
            for violation in violations:
                self._audit_log(
                    "rate_limit", 
                    violation.rule_name, 
                    "failed",
                    {
                        "reason": "rate_limit_exceeded",
                        "rule": violation.rule_name,
                        "current_count": violation.current_count,
                        "limit": violation.limit,
                        "window_seconds": violation.window_seconds
                    }
                )
            
            raise SecurityException(f"Rate limit exceeded for operation: {operation}")
        
        return True
    
    def authorize_operation(
        self, 
        token: AuthToken, 
        operation: str, 
        resource: str = ""
    ) -> bool:
        """
        Authorize an operation for a token.
        
        Args:
            token: Authentication token
            operation: Operation to authorize (e.g., "documents.create")
            resource: Optional resource identifier
            
        Returns:
            True if authorized
            
        Raises:
            AuthorizationException: If authorization fails
        """
        if not self.config.auth_enabled:
            return True
        
        # Check if token is expired
        if token.is_expired():
            self._audit_log("authorization", operation, "failed", 
                          {"reason": "expired_token", "user_id": token.user_id})
            raise AuthorizationException("Token expired")
        
        # Check if token is still active
        if token.correlation_id not in self._active_tokens:
            self._audit_log("authorization", operation, "failed", 
                          {"reason": "inactive_token", "user_id": token.user_id})
            raise AuthorizationException("Token not active")
        
        # Check permissions
        if operation not in token.permissions:
            self._audit_log("authorization", operation, "failed", 
                          {"reason": "insufficient_permissions", "user_id": token.user_id,
                           "required_permission": operation})
            raise AuthorizationException(f"Insufficient permissions for operation: {operation}")
        
        # Check rate limiting
        if not self._check_rate_limit(token.user_id):
            self._audit_log("authorization", operation, "failed", 
                          {"reason": "rate_limit_exceeded", "user_id": token.user_id})
            raise AuthorizationException("Rate limit exceeded")
        
        self._audit_log("authorization", operation, "success", 
                      {"user_id": token.user_id, "resource": resource})
        
        return True
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        if not self.config.rate_limiting_enabled:
            return True
        
        now = time.time()
        minute_ago = now - 60
        
        # Get user's recent requests
        if user_id not in self._rate_limits:
            self._rate_limits[user_id] = []
        
        user_requests = self._rate_limits[user_id]
        
        # Remove old requests
        user_requests[:] = [req_time for req_time in user_requests if req_time > minute_ago]
        
        # Check if under limit
        if len(user_requests) >= self.config.max_requests_per_minute:
            return False
        
        # Add current request
        user_requests.append(now)
        return True
    
    def create_api_key(
        self, 
        user_id: str, 
        roles: List[str], 
        expires_at: Optional[datetime] = None,
        description: str = ""
    ) -> str:
        """
        Create a new API key.
        
        Args:
            user_id: User ID for the key
            roles: List of roles for the key
            expires_at: Optional expiration time
            description: Optional description
            
        Returns:
            Generated API key
        """
        api_key = secrets.token_urlsafe(32)
        
        self._api_keys[api_key] = {
            "user_id": user_id,
            "roles": roles,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "enabled": True,
            "description": description
        }
        
        self._audit_log("api_key", "create", "success", 
                      {"user_id": user_id, "roles": roles, "description": description})
        
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if key was revoked
        """
        if api_key in self._api_keys:
            user_id = self._api_keys[api_key]["user_id"]
            del self._api_keys[api_key]
            
            self._audit_log("api_key", "revoke", "success", 
                          {"user_id": user_id})
            return True
        
        return False
    
    def create_jwt_token(
        self, 
        user_id: str, 
        roles: List[str], 
        expires_in_hours: int = 24
    ) -> str:
        """
        Create a JWT token.
        
        Args:
            user_id: User ID for the token
            roles: List of roles for the token
            expires_in_hours: Token expiration time in hours
            
        Returns:
            JWT token string
        """
        payload = {
            "user_id": user_id,
            "roles": roles,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=expires_in_hours)
        }
        
        token = jwt.encode(payload, self.config.jwt_secret, algorithm="HS256")
        
        self._audit_log("jwt", "create", "success", 
                      {"user_id": user_id, "roles": roles})
        
        return token
    
    def invalidate_token(self, correlation_id: str) -> bool:
        """
        Invalidate an active token.
        
        Args:
            correlation_id: Token correlation ID
            
        Returns:
            True if token was invalidated
        """
        if correlation_id in self._active_tokens:
            token = self._active_tokens[correlation_id]
            del self._active_tokens[correlation_id]
            
            self._audit_log("token", "invalidate", "success", 
                          {"user_id": token.user_id})
            return True
        
        return False
    
    def get_active_tokens(self) -> List[Dict[str, Any]]:
        """
        Get information about active tokens.
        
        Returns:
            List of active token information
        """
        active_tokens = []
        
        for correlation_id, token in self._active_tokens.items():
            if not token.is_expired():
                active_tokens.append({
                    "correlation_id": correlation_id,
                    "user_id": token.user_id,
                    "roles": token.roles,
                    "expires_at": token.expires_at.isoformat(),
                    "permissions_count": len(token.permissions)
                })
        
        return active_tokens
    
    def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired tokens.
        
        Returns:
            Number of tokens cleaned up
        """
        expired_tokens = []
        
        for correlation_id, token in self._active_tokens.items():
            if token.is_expired():
                expired_tokens.append(correlation_id)
        
        for correlation_id in expired_tokens:
            del self._active_tokens[correlation_id]
        
        if expired_tokens:
            self._audit_log("token", "cleanup", "success", 
                          {"expired_count": len(expired_tokens)})
        
        return len(expired_tokens)
    
    def _audit_log(
        self, 
        operation: str, 
        resource: str, 
        status: str, 
        details: Dict[str, Any]
    ) -> None:
        """Log a security audit event."""
        if not self.config.audit_logging_enabled:
            return
        
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            user_id=details.get("user_id", "system"),
            operation=operation,
            resource=resource,
            status=status,
            details=details,
            correlation_id=self._generate_correlation_id()
        )
        
        self._audit_events.append(event)
        
        # Process event through security monitoring
        alerts = self.monitoring_service.process_audit_event(event)
        
        # Handle any generated alerts
        for alert in alerts:
            self._handle_security_alert(alert)
        
        # Keep only recent events (last 1000)
        if len(self._audit_events) > 1000:
            self._audit_events = self._audit_events[-1000:]
    
    def get_audit_events(
        self, 
        limit: int = 100, 
        user_id: Optional[str] = None,
        operation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit events.
        
        Args:
            limit: Maximum number of events to return
            user_id: Filter by user ID
            operation: Filter by operation
            
        Returns:
            List of audit events
        """
        events = self._audit_events
        
        # Apply filters
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if operation:
            events = [e for e in events if e.operation == operation]
        
        # Sort by timestamp (most recent first) and limit
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]
        
        return [event.to_dict() for event in events]
    
    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get security statistics.
        
        Returns:
            Dictionary with security statistics
        """
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Count recent events
        recent_events = [e for e in self._audit_events if e.timestamp > hour_ago]
        
        # Count by status
        success_count = len([e for e in recent_events if e.status == "success"])
        failed_count = len([e for e in recent_events if e.status == "failed"])
        
        # Count active tokens
        active_tokens = [t for t in self._active_tokens.values() if not t.is_expired()]
        
        stats = {
            "auth_enabled": self.config.auth_enabled,
            "auth_type": self.config.auth_type,
            "active_tokens": len(active_tokens),
            "total_api_keys": len(self._api_keys),
            "recent_events_count": len(recent_events),
            "recent_success_count": success_count,
            "recent_failed_count": failed_count,
            "rate_limiting_enabled": self.config.rate_limiting_enabled,
            "audit_logging_enabled": self.config.audit_logging_enabled
        }
        
        # Add rate limiting metrics if available
        if self.rate_limiting_service:
            rate_limit_metrics = self.rate_limiting_service.get_metrics()
            stats["rate_limiting"] = rate_limit_metrics
        
        return stats
    
    def health_check(self) -> bool:
        """
        Perform security health check.
        
        Returns:
            True if security system is healthy
        """
        try:
            # Check if we can create a test token
            if self.config.auth_type == "jwt":
                test_token = self.create_jwt_token("health_check", ["viewer"], 1)
                # Try to decode it
                jwt.decode(test_token, self.config.jwt_secret, algorithms=["HS256"])
            
            # Clean up expired tokens
            self.cleanup_expired_tokens()
            
            # Check rate limiting service health
            if self.rate_limiting_service:
                if not self.rate_limiting_service.health_check():
                    return False
            
            # Check monitoring service health
            if self.monitoring_service:
                if not self.monitoring_service.health_check():
                    return False
            
            return True
            
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of security manager."""
        return (
            f"SecurityManager(auth_enabled={self.config.auth_enabled}, "
            f"auth_type={self.config.auth_type}, "
            f"active_tokens={len(self._active_tokens)})"
        )
    
    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
            
        Raises:
            SecurityException: If encryption is not enabled or fails
        """
        if not self.config.encryption_enabled or not self.encryption_service:
            raise SecurityException("Encryption is not enabled")
        
        try:
            return self.encryption_service.encrypt(data)
        except EncryptionException as e:
            self._audit_log("encryption", "encrypt", "failed", 
                          {"error": str(e)})
            raise SecurityException(f"Encryption failed: {str(e)}")
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            
        Returns:
            Decrypted data
            
        Raises:
            SecurityException: If encryption is not enabled or decryption fails
        """
        if not self.config.encryption_enabled or not self.encryption_service:
            raise SecurityException("Encryption is not enabled")
        
        try:
            return self.encryption_service.decrypt_to_string(encrypted_data)
        except EncryptionException as e:
            self._audit_log("encryption", "decrypt", "failed", 
                          {"error": str(e)})
            raise SecurityException(f"Decryption failed: {str(e)}")
    
    def detect_pii(self, text: str, confidence_threshold: float = 0.5) -> List[PIIMatch]:
        """
        Detect PII in text.
        
        Args:
            text: Text to analyze for PII
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of detected PII matches
            
        Raises:
            SecurityException: If PII detection is not enabled
        """
        if not self.config.pii_detection_enabled or not self.pii_detection_service:
            raise SecurityException("PII detection is not enabled")
        
        try:
            matches = self.pii_detection_service.detect_pii(text, confidence_threshold)
            
            if matches:
                self._audit_log("pii_detection", "detect", "success", 
                              {"matches_found": len(matches), 
                               "pii_types": [m.type for m in matches]})
            
            return matches
            
        except Exception as e:
            self._audit_log("pii_detection", "detect", "failed", 
                          {"error": str(e)})
            raise SecurityException(f"PII detection failed: {str(e)}")
    
    def mask_sensitive_data(
        self, 
        text: str, 
        mask_char: str = "*",
        preserve_format: bool = True
    ) -> tuple[str, List[PIIMatch]]:
        """
        Mask sensitive data in text.
        
        Args:
            text: Text to mask
            mask_char: Character to use for masking
            preserve_format: Whether to preserve format
            
        Returns:
            Tuple of (masked_text, detected_pii_matches)
            
        Raises:
            SecurityException: If PII detection is not enabled
        """
        if not self.config.pii_detection_enabled or not self.pii_detection_service:
            if self.config.data_masking_enabled:
                # If masking is enabled but PII detection is not, mask everything
                return mask_char * len(text), []
            else:
                return text, []
        
        try:
            masked_text, matches = self.pii_detection_service.mask_pii(
                text, mask_char, preserve_format
            )
            
            if matches:
                self._audit_log("data_masking", "mask", "success", 
                              {"original_length": len(text),
                               "masked_length": len(masked_text),
                               "pii_matches": len(matches)})
            
            return masked_text, matches
            
        except Exception as e:
            self._audit_log("data_masking", "mask", "failed", 
                          {"error": str(e)})
            raise SecurityException(f"Data masking failed: {str(e)}")
    
    def encrypt_and_store_sensitive_data(self, data: str) -> dict:
        """
        Detect PII, mask it, and encrypt the data.
        
        Args:
            data: Data to process
            
        Returns:
            Dictionary with processed data information
        """
        result = {
            "original_length": len(data),
            "pii_detected": False,
            "pii_matches": [],
            "encrypted": False,
            "processed_data": data
        }
        
        # Detect PII if enabled
        if self.config.pii_detection_enabled and self.pii_detection_service:
            try:
                pii_matches = self.detect_pii(data)
                if pii_matches:
                    result["pii_detected"] = True
                    result["pii_matches"] = [match.to_dict() for match in pii_matches]
                    
                    # Mask PII if data masking is enabled
                    if self.config.data_masking_enabled:
                        masked_data, _ = self.mask_sensitive_data(data)
                        result["processed_data"] = masked_data
            except Exception as e:
                # Don't fail the entire operation if PII detection fails
                self._audit_log("data_processing", "pii_detection", "failed", 
                              {"error": str(e)})
        
        # Encrypt data if encryption is enabled
        if self.config.encryption_enabled and self.encryption_service:
            try:
                encrypted_data = self.encrypt_data(result["processed_data"])
                result["processed_data"] = encrypted_data
                result["encrypted"] = True
            except Exception as e:
                # Don't fail the entire operation if encryption fails
                self._audit_log("data_processing", "encryption", "failed", 
                              {"error": str(e)})
        
        return result
    
    def rotate_encryption_key(self, new_key: Optional[str] = None) -> str:
        """
        Rotate the encryption key.
        
        Args:
            new_key: New encryption key (generated if not provided)
            
        Returns:
            New encryption key
            
        Raises:
            SecurityException: If encryption is not enabled
        """
        if not self.config.encryption_enabled or not self.encryption_service:
            raise SecurityException("Encryption is not enabled")
        
        if new_key is None:
            new_key = self.encryption_service.generate_new_key()
        
        old_key_info = self.encryption_service.get_key_info()
        self.encryption_service.rotate_key(new_key)
        new_key_info = self.encryption_service.get_key_info()
        
        self._audit_log("encryption", "key_rotation", "success", 
                      {"old_key_hash": old_key_info["key_hash"],
                       "new_key_hash": new_key_info["key_hash"]})
        
        return new_key
    
    def add_custom_pii_pattern(
        self, 
        name: str, 
        pattern: str, 
        confidence: float,
        description: str = ""
    ) -> None:
        """
        Add a custom PII detection pattern.
        
        Args:
            name: Name of the PII type
            pattern: Regex pattern
            confidence: Confidence score
            description: Pattern description
            
        Raises:
            SecurityException: If PII detection is not enabled
        """
        if not self.config.pii_detection_enabled or not self.pii_detection_service:
            raise SecurityException("PII detection is not enabled")
        
        try:
            self.pii_detection_service.add_custom_pattern(
                name, pattern, confidence, description
            )
            
            self._audit_log("pii_detection", "add_pattern", "success", 
                          {"pattern_name": name, "confidence": confidence})
            
        except Exception as e:
            self._audit_log("pii_detection", "add_pattern", "failed", 
                          {"pattern_name": name, "error": str(e)})
            raise SecurityException(f"Failed to add PII pattern: {str(e)}")
    
    def get_encryption_info(self) -> dict:
        """
        Get information about encryption configuration.
        
        Returns:
            Dictionary with encryption information
        """
        if not self.config.encryption_enabled or not self.encryption_service:
            return {"enabled": False}
        
        info = self.encryption_service.get_key_info()
        info["enabled"] = True
        return info
    
    def get_pii_detection_info(self) -> dict:
        """
        Get information about PII detection configuration.
        
        Returns:
            Dictionary with PII detection information
        """
        if not self.config.pii_detection_enabled or not self.pii_detection_service:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "ml_detection": self.pii_detection_service.enable_ml_detection,
            "available_patterns": self.pii_detection_service.get_available_patterns(),
            "pattern_count": len(self.pii_detection_service.patterns)
        }
    
    def _handle_security_alert(self, alert) -> None:
        """
        Handle a security alert by taking appropriate action.
        
        Args:
            alert: Security alert to handle
        """
        # Log the alert
        self._audit_log("security_alert", alert.alert_type, "generated", {
            "alert_id": alert.alert_id,
            "severity": alert.severity,
            "user_id": alert.user_id,
            "description": alert.description
        })
        
        # Take automatic actions based on alert type and severity
        if alert.severity == "critical":
            # For critical alerts, consider blocking the user/IP
            if alert.alert_type == "brute_force_attack":
                # Could implement automatic IP blocking here
                pass
        
        elif alert.severity == "high":
            # For high severity alerts, increase monitoring
            if alert.alert_type == "privilege_escalation_attempt":
                # Could implement additional monitoring for this user
                pass
    
    def get_security_alerts(
        self, 
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get security alerts from monitoring service.
        
        Args:
            severity: Filter by severity level
            alert_type: Filter by alert type
            limit: Maximum number of alerts to return
            
        Returns:
            List of security alerts
        """
        alerts = self.monitoring_service.get_active_alerts(severity, alert_type, limit)
        return [
            {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat(),
                "user_id": alert.user_id,
                "description": alert.description,
                "details": alert.details,
                "resolved": alert.resolved
            }
            for alert in alerts
        ]
    
    def resolve_security_alert(self, alert_id: str, resolved_by: str = "admin") -> bool:
        """
        Resolve a security alert.
        
        Args:
            alert_id: ID of the alert to resolve
            resolved_by: Who resolved the alert
            
        Returns:
            True if alert was resolved
        """
        result = self.monitoring_service.resolve_alert(alert_id, resolved_by)
        
        if result:
            self._audit_log("security_alert", "resolve", "success", {
                "alert_id": alert_id,
                "resolved_by": resolved_by
            })
        
        return result
    
    def get_threat_indicators(
        self, 
        indicator_type: Optional[str] = None,
        min_severity: str = "low"
    ) -> List[Dict[str, Any]]:
        """
        Get threat indicators from monitoring service.
        
        Args:
            indicator_type: Filter by indicator type
            min_severity: Minimum severity level
            
        Returns:
            List of threat indicators
        """
        indicators = self.monitoring_service.get_threat_indicators(indicator_type, min_severity)
        return [
            {
                "indicator_type": ind.indicator_type,
                "value": ind.value,
                "severity": ind.severity,
                "description": ind.description,
                "first_seen": ind.first_seen.isoformat(),
                "last_seen": ind.last_seen.isoformat(),
                "count": ind.count
            }
            for ind in indicators
        ]
    
    def block_ip_address(self, ip_address: str, reason: str = "Security violation") -> None:
        """
        Block an IP address.
        
        Args:
            ip_address: IP address to block
            reason: Reason for blocking
        """
        self.monitoring_service.block_ip(ip_address, reason)
        
        self._audit_log("ip_management", "block", "success", {
            "ip_address": ip_address,
            "reason": reason
        })
    
    def unblock_ip_address(self, ip_address: str) -> bool:
        """
        Unblock an IP address.
        
        Args:
            ip_address: IP address to unblock
            
        Returns:
            True if IP was unblocked
        """
        result = self.monitoring_service.unblock_ip(ip_address)
        
        if result:
            self._audit_log("ip_management", "unblock", "success", {
                "ip_address": ip_address
            })
        
        return result
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """
        Check if an IP address is blocked.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            True if IP is blocked
        """
        return self.monitoring_service.is_ip_blocked(ip_address)
    
    def get_rate_limit_status(
        self,
        user_id: str = None,
        ip_address: str = None,
        operation: str = None
    ) -> Dict[str, Any]:
        """
        Get rate limit status for a user/IP/operation.
        
        Args:
            user_id: User identifier
            ip_address: IP address
            operation: Operation to check
            
        Returns:
            Dictionary with rate limit status
        """
        if not self.rate_limiting_service:
            return {"rate_limiting_enabled": False}
        
        return self.rate_limiting_service.get_rate_limit_status(
            user_id=user_id,
            ip_address=ip_address,
            operation=operation
        )
    
    def get_rate_limit_violations(
        self,
        user_id: str = None,
        ip_address: str = None,
        rule_name: str = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get rate limit violations.
        
        Args:
            user_id: Filter by user ID
            ip_address: Filter by IP address
            rule_name: Filter by rule name
            hours: Hours to look back
            
        Returns:
            List of rate limit violations
        """
        if not self.rate_limiting_service:
            return []
        
        violations = self.rate_limiting_service.get_violations(
            user_id=user_id,
            ip_address=ip_address,
            rule_name=rule_name,
            hours=hours
        )
        
        # Convert to dictionaries
        return [
            {
                "rule_name": v.rule_name,
                "identifier": v.identifier,
                "timestamp": v.timestamp.isoformat(),
                "current_count": v.current_count,
                "limit": v.limit,
                "window_seconds": v.window_seconds,
                "operation": v.operation,
                "details": v.details
            }
            for v in violations
        ]
    
    def get_user_risk_assessment(self, user_id: str) -> Dict[str, Any]:
        """
        Get risk assessment for a user.
        
        Args:
            user_id: User ID to assess
            
        Returns:
            Dictionary with risk assessment
        """
        return self.monitoring_service.get_user_risk_score(user_id)
    
    def update_security_thresholds(self, thresholds: Dict[str, int]) -> None:
        """
        Update security monitoring thresholds.
        
        Args:
            thresholds: Dictionary of new threshold values
        """
        self.monitoring_service.update_thresholds(thresholds)
        
        self._audit_log("security_config", "update_thresholds", "success", {
            "updated_thresholds": thresholds
        })
    
    def get_security_monitoring_metrics(self) -> Dict[str, Any]:
        """
        Get security monitoring metrics.
        
        Returns:
            Dictionary with security monitoring metrics
        """
        return self.monitoring_service.get_security_metrics()
    
    def __repr__(self) -> str:
        """Detailed string representation of security manager."""
        return (
            f"SecurityManager(auth_enabled={self.config.auth_enabled}, "
            f"auth_type='{self.config.auth_type}', "
            f"rbac_enabled={self.config.rbac_enabled}, "
            f"encryption_enabled={self.config.encryption_enabled}, "
            f"pii_detection_enabled={self.config.pii_detection_enabled}, "
            f"active_tokens={len(self._active_tokens)}, "
            f"api_keys={len(self._api_keys)})"
        )