"""
ObservabilityManager for comprehensive monitoring, logging, and tracing.
"""

import json
import logging
import os
import psutil
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TextIO, Union
from contextlib import contextmanager
from collections import defaultdict, deque

from ..models.config import ObservabilityConfig
from ..models.observability import (
    LogLevel, LogContext, LogEntry, Metric, TraceContext, TraceStatus,
    HealthStatus, SystemMetrics, generate_correlation_id, create_log_context,
    create_trace_context
)
from ..exceptions import VectorDBException
from .metrics import EnhancedMetricsCollector
from .tracing import DistributedTracer, create_tracer, TracingInstrumentation


class StructuredLogger:
    """Structured logger with JSON output and context propagation."""
    
    def __init__(
        self,
        name: str = "vector_db",
        level: LogLevel = LogLevel.INFO,
        log_format: str = "json",
        log_file: Optional[str] = None
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Log level
            log_format: Log format ("json" or "text")
            log_file: Optional log file path
        """
        self.name = name
        self.level = level
        self.log_format = log_format
        self.log_file = log_file
        
        # Thread-local storage for context
        self._local = threading.local()
        
        # Setup Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.value))
        
        # Clear existing handlers
        self._logger.handlers.clear()
        
        # Add handlers
        self._setup_handlers()
        
        # Log entries storage for retrieval
        self._log_entries: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def _setup_handlers(self) -> None:
        """Setup log handlers."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.level.value))
        
        # File handler if specified
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(getattr(logging, self.level.value))
            self._logger.addHandler(file_handler)
        
        self._logger.addHandler(console_handler)
        
        # Custom formatter
        if self.log_format == "json":
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        for handler in self._logger.handlers:
            handler.setFormatter(formatter)
    
    def set_context(self, context: LogContext) -> None:
        """Set logging context for current thread."""
        self._local.context = context
    
    def get_context(self) -> Optional[LogContext]:
        """Get logging context for current thread."""
        return getattr(self._local, 'context', None)
    
    def clear_context(self) -> None:
        """Clear logging context for current thread."""
        if hasattr(self._local, 'context'):
            delattr(self._local, 'context')
    
    def log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[LogContext] = None,
        exception: Optional[Exception] = None,
        **kwargs
    ) -> None:
        """
        Log a structured message.
        
        Args:
            level: Log level
            message: Log message
            context: Optional log context (uses thread-local if not provided)
            exception: Optional exception to log
            **kwargs: Additional context data
        """
        # Skip if level is below threshold
        if self._should_skip_level(level):
            return
        
        # Use provided context or thread-local context
        if context is None:
            context = self.get_context()
        
        # Create default context if none exists
        if context is None:
            context = create_log_context()
        
        # Add kwargs to context metadata
        if kwargs:
            context.metadata.update(kwargs)
        
        # Create log entry
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            context=context,
            logger_name=self.name,
            thread_id=str(threading.get_ident()),
            process_id=str(os.getpid()),
            exception=str(exception) if exception else None,
            stack_trace=self._get_stack_trace(exception) if exception else None
        )
        
        # Store log entry
        with self._lock:
            self._log_entries.append(log_entry)
        
        # Log using Python logger
        python_level = getattr(logging, level.value)
        
        if self.log_format == "json":
            # For JSON format, pass the structured data
            self._logger.log(python_level, json.dumps(log_entry.to_dict()))
        else:
            # For text format, create readable message
            formatted_message = self._format_text_message(log_entry)
            self._logger.log(python_level, formatted_message)
    
    def _should_skip_level(self, level: LogLevel) -> bool:
        """Check if log level should be skipped."""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARN: 2,
            LogLevel.ERROR: 3,
            LogLevel.FATAL: 4
        }
        return level_order[level] < level_order[self.level]
    
    def _get_stack_trace(self, exception: Exception) -> Optional[str]:
        """Get stack trace from exception."""
        import traceback
        if exception:
            return ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        return None
    
    def _format_text_message(self, log_entry: LogEntry) -> str:
        """Format log entry for text output."""
        parts = [log_entry.message]
        
        if log_entry.context.correlation_id:
            parts.append(f"correlation_id={log_entry.context.correlation_id}")
        
        if log_entry.context.user_id:
            parts.append(f"user_id={log_entry.context.user_id}")
        
        if log_entry.context.operation:
            parts.append(f"operation={log_entry.context.operation}")
        
        if log_entry.context.metadata:
            for key, value in log_entry.context.metadata.items():
                parts.append(f"{key}={value}")
        
        return " | ".join(parts)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warn(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(LogLevel.WARN, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, exception=exception, **kwargs)
    
    def fatal(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log fatal message."""
        self.log(LogLevel.FATAL, message, exception=exception, **kwargs)
    
    def get_recent_logs(self, limit: int = 100) -> List[LogEntry]:
        """Get recent log entries."""
        with self._lock:
            return list(self._log_entries)[-limit:]


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        try:
            # If message is already JSON, return as-is
            if record.getMessage().startswith('{'):
                return record.getMessage()
            
            # Otherwise, create basic JSON structure
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "thread_id": str(threading.get_ident()),
                "process_id": str(os.getpid())
            }
            
            return json.dumps(log_data)
        except Exception:
            # Fallback to standard formatting
            return super().format(record)


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metrics
        """
        self.retention_hours = retention_hours
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = threading.Lock()
        
        # System metrics tracking
        self._request_count = 0
        self._error_count = 0
        self._response_times: deque = deque(maxlen=1000)
        
        # Application metrics
        self._documents_indexed = 0
        self._searches_performed = 0
        self._embeddings_generated = 0
        self._cache_hits = 0
        self._cache_misses = 0
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        metric_type: str = "gauge",
        unit: Optional[str] = None
    ) -> None:
        """
        Record a metric measurement.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
            metric_type: Type of metric (gauge, counter, histogram, timer)
            unit: Optional unit of measurement
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            metric_type=metric_type,
            unit=unit
        )
        
        with self._lock:
            self._metrics[name].append(metric)
            self._cleanup_old_metrics()
    
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, 1, tags, "counter")
    
    def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric."""
        self.record_metric(name, duration_ms, tags, "timer", "ms")
    
    def record_gauge(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self.record_metric(name, value, tags, "gauge")
    
    def record_request(self, response_time_ms: float, error: bool = False) -> None:
        """Record a request with response time and error status."""
        with self._lock:
            self._request_count += 1
            if error:
                self._error_count += 1
            self._response_times.append(response_time_ms)
    
    def record_document_indexed(self) -> None:
        """Record a document indexing operation."""
        with self._lock:
            self._documents_indexed += 1
        self.increment_counter("documents_indexed")
    
    def record_search_performed(self) -> None:
        """Record a search operation."""
        with self._lock:
            self._searches_performed += 1
        self.increment_counter("searches_performed")
    
    def record_embedding_generated(self) -> None:
        """Record an embedding generation."""
        with self._lock:
            self._embeddings_generated += 1
        self.increment_counter("embeddings_generated")
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._cache_hits += 1
        self.increment_counter("cache_hits")
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._cache_misses += 1
        self.increment_counter("cache_misses")
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        hours: int = 1
    ) -> List[Metric]:
        """
        Get metrics for a specific name or all metrics.
        
        Args:
            name: Optional metric name filter
            hours: Hours to look back
            
        Returns:
            List of metrics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            if name:
                metrics = [
                    m for m in self._metrics.get(name, [])
                    if m.timestamp > cutoff
                ]
            else:
                metrics = []
                for metric_list in self._metrics.values():
                    metrics.extend([
                        m for m in metric_list
                        if m.timestamp > cutoff
                    ])
        
        return sorted(metrics, key=lambda m: m.timestamp)
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        # Get system information
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        disk_info = psutil.disk_usage('/')
        
        # Calculate average response time
        with self._lock:
            avg_response_time = (
                sum(self._response_times) / len(self._response_times)
                if self._response_times else 0.0
            )
            
            cache_hit_rate = (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0.0
            )
        
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            memory_usage_mb=memory_info.used / (1024 * 1024),
            cpu_usage_percent=cpu_percent,
            disk_usage_mb=disk_info.used / (1024 * 1024),
            request_count=self._request_count,
            error_count=self._error_count,
            avg_response_time_ms=avg_response_time,
            documents_indexed=self._documents_indexed,
            searches_performed=self._searches_performed,
            embeddings_generated=self._embeddings_generated,
            cache_hit_rate=cache_hit_rate
        )
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics beyond retention period."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        for name, metric_list in self._metrics.items():
            while metric_list and metric_list[0].timestamp < cutoff:
                metric_list.popleft()


class TracingManager:
    """Manages distributed tracing operations."""
    
    def __init__(self, service_name: str = "vector-database"):
        """
        Initialize tracing manager.
        
        Args:
            service_name: Name of the service for tracing
        """
        self.service_name = service_name
        self._active_traces: Dict[str, TraceContext] = {}
        self._completed_traces: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        # Thread-local storage for current trace context
        self._local = threading.local()
    
    def start_trace(
        self,
        operation_name: str,
        parent_context: Optional[TraceContext] = None
    ) -> TraceContext:
        """
        Start a new trace span.
        
        Args:
            operation_name: Name of the operation being traced
            parent_context: Optional parent trace context
            
        Returns:
            New trace context
        """
        trace_context = create_trace_context(
            operation_name=operation_name,
            trace_id=parent_context.trace_id if parent_context else None,
            parent_span_id=parent_context.span_id if parent_context else None
        )
        
        # Add service tag
        trace_context.add_tag("service.name", self.service_name)
        
        with self._lock:
            self._active_traces[trace_context.span_id] = trace_context
        
        # Set as current trace context
        self._local.current_trace = trace_context
        
        return trace_context
    
    def get_current_trace(self) -> Optional[TraceContext]:
        """Get current trace context for this thread."""
        return getattr(self._local, 'current_trace', None)
    
    def end_trace(
        self,
        context: TraceContext,
        status: TraceStatus = TraceStatus.SUCCESS
    ) -> None:
        """
        End a trace span.
        
        Args:
            context: Trace context to end
            status: Final status of the trace
        """
        context.finish(status)
        
        with self._lock:
            # Move from active to completed
            if context.span_id in self._active_traces:
                del self._active_traces[context.span_id]
            
            self._completed_traces.append(context)
        
        # Clear current trace if it matches
        current_trace = getattr(self._local, 'current_trace', None)
        if current_trace and current_trace.span_id == context.span_id:
            self._local.current_trace = None
    
    @contextmanager
    def trace_operation(self, operation_name: str):
        """
        Context manager for tracing an operation.
        
        Args:
            operation_name: Name of the operation
        """
        trace_context = self.start_trace(operation_name)
        try:
            yield trace_context
            self.end_trace(trace_context, TraceStatus.SUCCESS)
        except Exception as e:
            trace_context.add_log(f"Error: {str(e)}", "error")
            self.end_trace(trace_context, TraceStatus.ERROR)
            raise
    
    def add_trace_log(self, message: str, level: str = "info", **kwargs) -> None:
        """Add a log entry to the current trace."""
        current_trace = self.get_current_trace()
        if current_trace:
            current_trace.add_log(message, level, **kwargs)
    
    def add_trace_tag(self, key: str, value: str) -> None:
        """Add a tag to the current trace."""
        current_trace = self.get_current_trace()
        if current_trace:
            current_trace.add_tag(key, value)
    
    def get_traces(self, limit: int = 100) -> List[TraceContext]:
        """Get recent completed traces."""
        with self._lock:
            return list(self._completed_traces)[-limit:]


class ObservabilityManager:
    """
    Main observability manager that coordinates logging, metrics, and tracing.
    """
    
    def __init__(self, config: ObservabilityConfig):
        """
        Initialize observability manager.
        
        Args:
            config: Observability configuration
        """
        self.config = config
        
        # Initialize components
        self.logger = StructuredLogger(
            name="vector_db",
            level=LogLevel(config.log_level),
            log_format=config.log_format,
            log_file=config.log_file
        )
        
        self.metrics = EnhancedMetricsCollector()
        
        # Initialize distributed tracing
        self.distributed_tracer = None
        self.tracing_instrumentation = None
        if config.tracing_enabled:
            # Create distributed tracer
            exporter_config = {}
            if config.tracing_endpoint:
                exporter_type = "http"
                exporter_config = {"endpoint": config.tracing_endpoint}
            else:
                exporter_type = "memory"  # Default to in-memory for testing
            
            self.distributed_tracer = create_tracer(
                service_name=config.tracing_service_name,
                service_version="1.0.0",
                exporter_type=exporter_type,
                exporter_config=exporter_config
            )
            
            # Create instrumentation helper
            self.tracing_instrumentation = TracingInstrumentation(self.distributed_tracer)
        
        # Keep legacy tracing for backward compatibility
        self.tracing = TracingManager(
            service_name=config.tracing_service_name
        ) if config.tracing_enabled else None
        
        # Health monitoring
        self._health_checks: Dict[str, callable] = {}
        
        # Performance monitoring
        self._performance_alerts: List[Dict[str, Any]] = []
        
        # Start background monitoring if enabled
        if config.performance_monitoring_enabled:
            self._start_performance_monitoring()
    
    def log_event(
        self,
        level: str,
        message: str,
        context: Optional[LogContext] = None,
        **kwargs
    ) -> None:
        """
        Log a structured event.
        
        Args:
            level: Log level (DEBUG, INFO, WARN, ERROR, FATAL)
            message: Log message
            context: Optional log context
            **kwargs: Additional context data
        """
        log_level = LogLevel(level.upper())
        self.logger.log(log_level, message, context, **kwargs)
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric measurement.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        if self.config.metrics_enabled:
            self.metrics.record_metric(name, value, tags)
    
    def start_trace(self, operation: str) -> Optional[TraceContext]:
        """
        Start a distributed trace.
        
        Args:
            operation: Operation name
            
        Returns:
            Trace context if tracing is enabled
        """
        if self.tracing:
            return self.tracing.start_trace(operation)
        return None
    
    def end_trace(
        self,
        context: TraceContext,
        status: str = "success"
    ) -> None:
        """
        End a distributed trace.
        
        Args:
            context: Trace context
            status: Final status
        """
        if self.tracing and context:
            trace_status = TraceStatus(status.lower())
            self.tracing.end_trace(context, trace_status)
    
    @contextmanager
    def trace_operation(self, operation_name: str):
        """Context manager for tracing operations."""
        if self.tracing:
            with self.tracing.trace_operation(operation_name) as trace_context:
                yield trace_context
        else:
            yield None
    
    def set_log_context(self, context: LogContext) -> None:
        """Set logging context for current thread."""
        self.logger.set_context(context)
    
    def clear_log_context(self) -> None:
        """Clear logging context for current thread."""
        self.logger.clear_context()
    
    def register_health_check(self, name: str, check_func: callable) -> None:
        """
        Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns (healthy: bool, message: str, details: dict)
        """
        self._health_checks[name] = check_func
    
    def health_check(self) -> HealthStatus:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status information
        """
        health_status = HealthStatus(
            status="healthy",
            timestamp=datetime.utcnow()
        )
        
        # Run registered health checks
        for name, check_func in self._health_checks.items():
            try:
                healthy, message, details = check_func()
                health_status.add_component(name, healthy, message, details)
            except Exception as e:
                health_status.add_component(
                    name, False, f"Health check failed: {str(e)}"
                )
        
        # Check system resources
        try:
            system_metrics = self.metrics.get_system_metrics()
            
            # Memory check
            memory_healthy = system_metrics.memory_usage_mb < self.config.memory_threshold_mb
            health_status.add_component(
                "memory",
                memory_healthy,
                f"Memory usage: {system_metrics.memory_usage_mb:.1f}MB",
                {"usage_mb": system_metrics.memory_usage_mb, "threshold_mb": self.config.memory_threshold_mb}
            )
            
            # CPU check
            cpu_healthy = system_metrics.cpu_usage_percent < self.config.cpu_threshold_percent
            health_status.add_component(
                "cpu",
                cpu_healthy,
                f"CPU usage: {system_metrics.cpu_usage_percent:.1f}%",
                {"usage_percent": system_metrics.cpu_usage_percent, "threshold_percent": self.config.cpu_threshold_percent}
            )
            
        except Exception as e:
            health_status.add_component(
                "system_metrics", False, f"Failed to get system metrics: {str(e)}"
            )
        
        return health_status
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self.metrics.get_system_metrics()
    
    def get_recent_logs(self, limit: int = 100) -> List[LogEntry]:
        """Get recent log entries."""
        return self.logger.get_recent_logs(limit)
    
    def get_metrics(self, name: Optional[str] = None, hours: int = 1) -> List[Metric]:
        """Get metrics data."""
        return self.metrics.get_metrics(name, hours)
    
    def get_traces(self, limit: int = 100) -> List[TraceContext]:
        """Get recent traces."""
        if self.tracing:
            return self.tracing.get_traces(limit)
        return []
    
    def record_performance_metric(
        self,
        operation: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a performance metric.
        
        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            tags: Optional tags for the metric
        """
        if self.config.metrics_enabled:
            self.metrics.record_performance_metric(operation, duration_ms, tags)
    
    def record_business_metric(
        self,
        name: str,
        value: Union[int, float],
        description: str = "",
        unit: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a business metric.
        
        Args:
            name: Metric name
            value: Metric value
            description: Metric description
            unit: Unit of measurement
            tags: Optional tags
        """
        if self.config.metrics_enabled:
            self.metrics.record_business_metric(name, value, description, unit, tags)
    
    @contextmanager
    def time_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations with both tracing and performance metrics."""
        # Start tracing if enabled
        trace_context = None
        if self.tracing:
            trace_context = self.tracing.start_trace(operation_name)
            if trace_context and tags:
                for key, value in tags.items():
                    trace_context.add_tag(key, value)
        
        # Time the operation for performance metrics
        with self.metrics.time_operation(operation_name, tags):
            try:
                yield trace_context
                if trace_context:
                    self.tracing.end_trace(trace_context, TraceStatus.SUCCESS)
            except Exception as e:
                if trace_context:
                    trace_context.add_log(f"Error: {str(e)}", "error")
                    self.tracing.end_trace(trace_context, TraceStatus.ERROR)
                raise
    
    def get_performance_statistics(
        self,
        operation: Optional[str] = None,
        window_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get performance statistics for operations.
        
        Args:
            operation: Optional specific operation name
            window_minutes: Time window for statistics
            
        Returns:
            Performance statistics
        """
        return self.metrics.get_performance_statistics(operation, window_minutes)
    
    def get_slow_operations(
        self,
        threshold_ms: float = 1000,
        window_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Get operations that are performing slowly.
        
        Args:
            threshold_ms: Threshold for slow operations
            window_minutes: Time window to analyze
            
        Returns:
            List of slow operations
        """
        return self.metrics.get_slow_operations(threshold_ms, window_minutes)
    
    def get_business_metrics(self) -> Dict[str, Any]:
        """Get current business metrics."""
        business_metrics = self.metrics.get_business_metrics()
        return {name: metric.to_dict() for name, metric in business_metrics.items()}
    
    def get_business_metric_trend(
        self,
        name: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get trend analysis for a business metric.
        
        Args:
            name: Metric name
            hours: Hours of history to analyze
            
        Returns:
            Trend analysis
        """
        return self.metrics.get_business_metric_trend(name, hours)
    
    def get_comprehensive_health_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive health status including all registered checks.
        
        Args:
            force_refresh: Force refresh of all health checks
            
        Returns:
            Comprehensive health status
        """
        return self.metrics.get_health_status(force_refresh)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of health check status."""
        return self.metrics.get_health_summary()
    
    def start_distributed_span(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent_context: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Start a distributed tracing span.
        
        Args:
            operation_name: Name of the operation
            attributes: Initial span attributes
            parent_context: Parent span context
            
        Returns:
            Span instance if tracing is enabled
        """
        if self.distributed_tracer:
            return self.distributed_tracer.start_span(
                operation_name,
                parent_context=parent_context,
                attributes=attributes
            )
        return None
    
    def end_distributed_span(self, span: Any) -> None:
        """
        End a distributed tracing span.
        
        Args:
            span: Span to end
        """
        if self.distributed_tracer and span:
            self.distributed_tracer.end_span(span)
    
    def get_current_span(self) -> Optional[Any]:
        """Get the current active span."""
        if self.distributed_tracer:
            return self.distributed_tracer.get_current_span()
        return None
    
    def set_current_span(self, span: Optional[Any]) -> None:
        """Set the current active span."""
        if self.distributed_tracer:
            self.distributed_tracer.set_current_span(span)
    
    @contextmanager
    def trace_distributed_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for distributed tracing operations.
        
        Args:
            operation_name: Name of the operation
            attributes: Initial span attributes
        """
        if self.distributed_tracer:
            with self.distributed_tracer.start_as_current_span(operation_name, attributes) as span:
                # Correlate with logging
                if span:
                    log_context = create_log_context(
                        correlation_id=span.trace_id,
                        operation=operation_name,
                        trace_id=span.trace_id,
                        span_id=span.span_id
                    )
                    self.set_log_context(log_context)
                
                try:
                    yield span
                finally:
                    self.clear_log_context()
        else:
            # Fallback to legacy tracing
            with self.time_operation(operation_name, attributes):
                yield None
    
    def add_span_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        span: Optional[Any] = None
    ) -> None:
        """
        Add an event to a span.
        
        Args:
            name: Event name
            attributes: Event attributes
            span: Target span (uses current span if None)
        """
        target_span = span or self.get_current_span()
        if target_span and hasattr(target_span, 'add_event'):
            target_span.add_event(name, attributes)
    
    def set_span_attribute(
        self,
        key: str,
        value: Any,
        span: Optional[Any] = None
    ) -> None:
        """
        Set a span attribute.
        
        Args:
            key: Attribute key
            value: Attribute value
            span: Target span (uses current span if None)
        """
        target_span = span or self.get_current_span()
        if target_span and hasattr(target_span, 'set_attribute'):
            target_span.set_attribute(key, value)
    
    def record_span_exception(
        self,
        exception: Exception,
        span: Optional[Any] = None
    ) -> None:
        """
        Record an exception in a span.
        
        Args:
            exception: Exception to record
            span: Target span (uses current span if None)
        """
        target_span = span or self.get_current_span()
        if target_span and hasattr(target_span, 'record_exception'):
            target_span.record_exception(exception)
    
    def get_active_spans(self) -> List[Dict[str, Any]]:
        """Get all currently active spans."""
        if self.distributed_tracer:
            spans = self.distributed_tracer.get_active_spans()
            return [span.to_dict() for span in spans]
        return []
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        stats = {
            "distributed_tracing_enabled": self.distributed_tracer is not None,
            "legacy_tracing_enabled": self.tracing is not None,
            "active_spans": len(self.get_active_spans()),
            "service_name": self.config.tracing_service_name if self.config.tracing_enabled else None
        }
        
        if self.distributed_tracer:
            stats.update({
                "sampling_rate": self.distributed_tracer.sampling_rate,
                "service_version": self.distributed_tracer.service_version,
                "processors_count": len(self.distributed_tracer.processors)
            })
        
        return stats
    
    def instrument_function(
        self,
        func: Callable,
        operation_name: Optional[str] = None
    ) -> Callable:
        """
        Instrument a function with distributed tracing.
        
        Args:
            func: Function to instrument
            operation_name: Optional operation name
            
        Returns:
            Instrumented function
        """
        if self.tracing_instrumentation:
            return self.tracing_instrumentation.instrument_function(func, operation_name)
        return func
    
    def instrument_class(
        self,
        cls: type,
        methods: Optional[List[str]] = None
    ) -> None:
        """
        Instrument a class with distributed tracing.
        
        Args:
            cls: Class to instrument
            methods: Optional list of method names to instrument
        """
        if self.tracing_instrumentation:
            self.tracing_instrumentation.instrument_class(cls, methods)
    
    def set_sampling_rate(self, rate: float) -> None:
        """
        Set the tracing sampling rate.
        
        Args:
            rate: Sampling rate between 0.0 and 1.0
        """
        if self.distributed_tracer:
            self.distributed_tracer.set_sampling_rate(rate)
    
    def _start_performance_monitoring(self) -> None:
        """Start background performance monitoring."""
        def monitor():
            while True:
                try:
                    system_metrics = self.get_system_metrics()
                    
                    # Check thresholds and generate alerts
                    if system_metrics.memory_usage_mb > self.config.memory_threshold_mb:
                        self.log_event(
                            "WARN",
                            f"High memory usage: {system_metrics.memory_usage_mb:.1f}MB",
                            memory_usage=system_metrics.memory_usage_mb,
                            threshold=self.config.memory_threshold_mb
                        )
                    
                    if system_metrics.cpu_usage_percent > self.config.cpu_threshold_percent:
                        self.log_event(
                            "WARN",
                            f"High CPU usage: {system_metrics.cpu_usage_percent:.1f}%",
                            cpu_usage=system_metrics.cpu_usage_percent,
                            threshold=self.config.cpu_threshold_percent
                        )
                    
                    # Record system metrics
                    self.record_metric("system.memory_usage_mb", system_metrics.memory_usage_mb)
                    self.record_metric("system.cpu_usage_percent", system_metrics.cpu_usage_percent)
                    self.record_metric("system.disk_usage_mb", system_metrics.disk_usage_mb)
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    self.log_event("ERROR", f"Performance monitoring error: {str(e)}", exception=e)
                    time.sleep(60)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def shutdown(self) -> None:
        """Shutdown observability manager and cleanup resources."""
        self.log_event("INFO", "Shutting down observability manager")
        
        # Clear contexts
        self.clear_log_context()
        
        # Final health check
        final_health = self.health_check()
        self.log_event("INFO", f"Final health status: {final_health.status}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()