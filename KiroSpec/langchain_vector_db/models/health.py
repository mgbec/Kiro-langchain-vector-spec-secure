"""
Health monitoring and system metrics data models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any


@dataclass
class HealthStatus:
    """System health status information."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    components: Dict[str, str]  # Component name -> status
    timestamp: datetime
    
    def is_healthy(self) -> bool:
        """Check if the system is healthy."""
        return self.status == "healthy"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary."""
        return {
            "status": self.status,
            "components": self.components,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SystemMetrics:
    """System performance metrics."""
    
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_mb: float
    active_connections: int
    request_count: int
    error_count: int
    avg_response_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system metrics to dictionary."""
        return {
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "disk_usage_mb": self.disk_usage_mb,
            "active_connections": self.active_connections,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_response_time_ms": self.avg_response_time_ms,
        }