"""
Observability models for logging, metrics, and tracing.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


class TraceStatus(Enum):
    """Status values for distributed tracing."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class LogContext:
    """Context information for structured logging."""
    
    correlation_id: str
    user_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log context to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "operation": self.operation,
            "component": self.component,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "metadata": self.metadata
        }


@dataclass
class LogEntry:
    """Structured log entry."""
    
    timestamp: datetime
    level: LogLevel
    message: str
    context: LogContext
    logger_name: str = "vector_db"
    thread_id: Optional[str] = None
    process_id: Optional[str] = None
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "logger": self.logger_name,
            "thread_id": self.thread_id,
            "process_id": self.process_id,
            "exception": self.exception,
            "stack_trace": self.stack_trace,
            **self.context.to_dict()
        }


@dataclass
class Metric:
    """Represents a metric measurement."""
    
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # "gauge", "counter", "histogram", "timer"
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "type": self.metric_type,
            "unit": self.unit
        }


@dataclass
class TraceContext:
    """Context for distributed tracing."""
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: TraceStatus = TraceStatus.SUCCESS
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize start time if not provided."""
        if self.start_time is None:
            self.start_time = datetime.utcnow()
    
    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the trace context."""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs) -> None:
        """Add a log entry to the trace context."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def finish(self, status: TraceStatus = TraceStatus.SUCCESS) -> None:
        """Finish the trace span."""
        self.end_time = datetime.utcnow()
        self.status = status
    
    def get_duration_ms(self) -> Optional[float]:
        """Get the duration of the trace in milliseconds."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace context to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.get_duration_ms(),
            "status": self.status.value,
            "tags": self.tags,
            "logs": self.logs
        }


@dataclass
class HealthStatus:
    """Health status information."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    overall_health: bool = True
    
    def add_component(
        self, 
        name: str, 
        healthy: bool, 
        message: str = "", 
        details: Dict[str, Any] = None
    ) -> None:
        """Add a component health status."""
        self.components[name] = {
            "healthy": healthy,
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update overall health
        if not healthy:
            self.overall_health = False
            if self.status == "healthy":
                self.status = "degraded"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary."""
        return {
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "overall_health": self.overall_health,
            "components": self.components
        }


@dataclass
class SystemMetrics:
    """System performance metrics."""
    
    timestamp: datetime
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_mb: float
    active_connections: int = 0
    request_count: int = 0
    error_count: int = 0
    avg_response_time_ms: float = 0.0
    
    # Application-specific metrics
    documents_indexed: int = 0
    searches_performed: int = 0
    embeddings_generated: int = 0
    cache_hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "disk_usage_mb": self.disk_usage_mb,
            "active_connections": self.active_connections,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_response_time_ms": self.avg_response_time_ms,
            "documents_indexed": self.documents_indexed,
            "searches_performed": self.searches_performed,
            "embeddings_generated": self.embeddings_generated,
            "cache_hit_rate": self.cache_hit_rate
        }


def generate_correlation_id() -> str:
    """Generate a unique correlation ID."""
    return str(uuid.uuid4())


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())


def generate_span_id() -> str:
    """Generate a unique span ID."""
    return str(uuid.uuid4())[:16]  # Shorter span IDs


def create_log_context(
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    operation: Optional[str] = None,
    component: Optional[str] = None,
    **kwargs
) -> LogContext:
    """Create a log context with optional parameters."""
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    
    return LogContext(
        correlation_id=correlation_id,
        user_id=user_id,
        operation=operation,
        component=component,
        **kwargs
    )


def create_trace_context(
    operation_name: str,
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None
) -> TraceContext:
    """Create a trace context for an operation."""
    if trace_id is None:
        trace_id = generate_trace_id()
    
    span_id = generate_span_id()
    
    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        operation_name=operation_name
    )