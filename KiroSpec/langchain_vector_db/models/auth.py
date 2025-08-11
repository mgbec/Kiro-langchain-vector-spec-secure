"""
Authentication and authorization models.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class AuthToken:
    """Represents an authentication token."""
    
    user_id: str
    roles: List[str]
    permissions: List[str]
    expires_at: datetime
    correlation_id: str
    token_type: str = "bearer"  # "bearer", "api_key", "jwt"
    issued_at: Optional[datetime] = None
    issuer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if token has specific permission."""
        return permission in self.permissions
    
    def has_role(self, role: str) -> bool:
        """Check if token has specific role."""
        return role in self.roles


@dataclass
class AuditEvent:
    """Represents an audit event for security logging."""
    
    timestamp: datetime
    user_id: str
    operation: str
    resource: str
    status: str  # "success", "failed", "denied"
    details: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "operation": self.operation,
            "resource": self.resource,
            "status": self.status,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id
        }


@dataclass
class User:
    """Represents a user in the system."""
    
    user_id: str
    username: str
    email: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles
    
    def add_role(self, role: str) -> None:
        """Add role to user."""
        if role not in self.roles:
            self.roles.append(role)
    
    def remove_role(self, role: str) -> None:
        """Remove role from user."""
        if role in self.roles:
            self.roles.remove(role)
    
    def add_permission(self, permission: str) -> None:
        """Add permission to user."""
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: str) -> None:
        """Remove permission from user."""
        if permission in self.permissions:
            self.permissions.remove(permission)


@dataclass
class Role:
    """Represents a role with associated permissions."""
    
    name: str
    description: str
    permissions: List[str] = field(default_factory=list)
    is_system_role: bool = False
    created_at: Optional[datetime] = None
    
    def has_permission(self, permission: str) -> bool:
        """Check if role has specific permission."""
        return permission in self.permissions
    
    def add_permission(self, permission: str) -> None:
        """Add permission to role."""
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: str) -> None:
        """Remove permission from role."""
        if permission in self.permissions:
            self.permissions.remove(permission)


@dataclass
class Permission:
    """Represents a system permission."""
    
    name: str
    description: str
    resource: str
    action: str
    is_system_permission: bool = False
    
    def matches(self, resource: str, action: str) -> bool:
        """Check if permission matches resource and action."""
        return (
            (self.resource == "*" or self.resource == resource) and
            (self.action == "*" or self.action == action)
        )


# Standard system roles
SYSTEM_ROLES = {
    "admin": Role(
        name="admin",
        description="Full system administrator",
        permissions=[
            "documents:*",
            "search:*",
            "config:*",
            "users:*",
            "security:*",
            "monitoring:*"
        ],
        is_system_role=True
    ),
    "writer": Role(
        name="writer",
        description="Can create, update, and delete documents",
        permissions=[
            "documents:create",
            "documents:update",
            "documents:delete",
            "documents:read",
            "search:query"
        ],
        is_system_role=True
    ),
    "reader": Role(
        name="reader",
        description="Can read documents and perform searches",
        permissions=[
            "documents:read",
            "search:query"
        ],
        is_system_role=True
    ),
    "viewer": Role(
        name="viewer",
        description="Can only view system status",
        permissions=[
            "monitoring:read"
        ],
        is_system_role=True
    )
}

# Standard system permissions
SYSTEM_PERMISSIONS = [
    Permission("documents:create", "Create new documents", "documents", "create", True),
    Permission("documents:read", "Read documents", "documents", "read", True),
    Permission("documents:update", "Update documents", "documents", "update", True),
    Permission("documents:delete", "Delete documents", "documents", "delete", True),
    Permission("documents:*", "All document operations", "documents", "*", True),
    
    Permission("search:query", "Perform search queries", "search", "query", True),
    Permission("search:*", "All search operations", "search", "*", True),
    
    Permission("config:read", "Read configuration", "config", "read", True),
    Permission("config:update", "Update configuration", "config", "update", True),
    Permission("config:*", "All configuration operations", "config", "*", True),
    
    Permission("users:create", "Create users", "users", "create", True),
    Permission("users:read", "Read user information", "users", "read", True),
    Permission("users:update", "Update users", "users", "update", True),
    Permission("users:delete", "Delete users", "users", "delete", True),
    Permission("users:*", "All user operations", "users", "*", True),
    
    Permission("security:read", "Read security information", "security", "read", True),
    Permission("security:update", "Update security settings", "security", "update", True),
    Permission("security:*", "All security operations", "security", "*", True),
    
    Permission("monitoring:read", "Read monitoring data", "monitoring", "read", True),
    Permission("monitoring:*", "All monitoring operations", "monitoring", "*", True)
]