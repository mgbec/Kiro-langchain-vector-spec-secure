"""
Unit tests for ObservabilityManager and related components.
"""

import pytest
import json
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from langchain_vector_db.services.observability import (
    ObservabilityManager,
    StructuredLogger,
    MetricsCollector,
    TracingManager,
    JsonFormatter
)
from langchain_vector_db.models.config import ObservabilityConfig
from langchain_vector_db.models.observability import (
    LogLevel, LogContext, LogEntry, Metric, TraceContext, TraceStatus,
    HealthStatus, SystemMetrics, create_log_context, create_trace_context
)


class TestLogContext:
    """Test cases for LogContext."""
    
    def test_initialization(self):
        """Test log context initialization."""
        context = LogContext(
            correlation_id="test-123",
            user_id="user-456",
            operation="test_op"
        )
        
        assert context.correlation_id == "test-123"
        assert context.user_id == "user-456"
        assert context.operation == "test_op"
        assert isinstance(context.metadata, dict)
    
    def test_to_dict(self):
        """Test log context dictionary conversion."""
        context = LogContext(
            correlation_id="test-123",
            user_id="user-456",
            operation="test_op",
            metadata={"key": "value"}
        )
        
        result = context.to_dict()
        
        assert result["correlation_id"] == "test-123"
        assert result["user_id"] == "user-456"
        assert result["operation"] == "test_op"
        assert result["metadata"] == {"key": "value"}


class TestLogEntry:
    """Test cases for LogEntry."""
    
    def test_initialization(self):
        """Test log entry initialization."""
        context = LogContext(correlation_id="test-123")
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.INFO,
            message="Test message",
            context=context
        )
        
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
        assert entry.context == context
    
    def test_to_dict(self):
        """Test log entry dictionary conversion."""
        context = LogContext(correlation_id="test-123")
        timestamp = datetime.utcnow()
        entry = LogEntry(
            timestamp=timestamp,
            level=LogLevel.ERROR,
            message="Error message",
            context=context,
            exception="ValueError: test error"
        )
        
        result = entry.to_dict()
        
        assert result["timestamp"] == timestamp.isoformat()
        assert result["level"] == "ERROR"
        assert result["message"] == "Error message"
        assert result["exception"] == "ValueError: test error"
        assert result["correlation_id"] == "test-123"


class TestTraceContext:
    """Test cases for TraceContext."""
    
    def test_initialization(self):
        """Test trace context initialization."""
        context = TraceContext(
            trace_id="trace-123",
            span_id="span-456",
            operation_name="test_operation"
        )
        
        assert context.trace_id == "trace-123"
        assert context.span_id == "span-456"
        assert context.operation_name == "test_operation"
        assert context.status == TraceStatus.SUCCESS
        assert context.start_time is not None
    
    def test_add_tag(self):
        """Test adding tags to trace context."""
        context = TraceContext(
            trace_id="trace-123",
            span_id="span-456",
            operation_name="test_operation"
        )
        
        context.add_tag("component", "test")
        context.add_tag("version", "1.0")
        
        assert context.tags["component"] == "test"
        assert context.tags["version"] == "1.0"
    
    def test_add_log(self):
        """Test adding logs to trace context."""
        context = TraceContext(
            trace_id="trace-123",
            span_id="span-456",
            operation_name="test_operation"
        )
        
        context.add_log("Test log message", "info", extra_field="value")
        
        assert len(context.logs) == 1
        log_entry = context.logs[0]
        assert log_entry["message"] == "Test log message"
        assert log_entry["level"] == "info"
        assert log_entry["extra_field"] == "value"
    
    def test_finish(self):
        """Test finishing a trace context."""
        context = TraceContext(
            trace_id="trace-123",
            span_id="span-456",
            operation_name="test_operation"
        )
        
        # Small delay to ensure duration > 0
        time.sleep(0.001)
        context.finish(TraceStatus.ERROR)
        
        assert context.status == TraceStatus.ERROR
        assert context.end_time is not None
        assert context.get_duration_ms() > 0
    
    def test_to_dict(self):
        """Test trace context dictionary conversion."""
        context = TraceContext(
            trace_id="trace-123",
            span_id="span-456",
            operation_name="test_operation"
        )
        
        context.add_tag("component", "test")
        context.add_log("Test message")
        context.finish()
        
        result = context.to_dict()
        
        assert result["trace_id"] == "trace-123"
        assert result["span_id"] == "span-456"
        assert result["operation_name"] == "test_operation"
        assert result["status"] == "success"
        assert "duration_ms" in result
        assert len(result["tags"]) == 1
        assert len(result["logs"]) == 1


class TestStructuredLogger:
    """Test cases for StructuredLogger."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = StructuredLogger(
            name="test_logger",
            level=LogLevel.DEBUG,
            log_format="json"
        )
    
    def test_initialization(self):
        """Test logger initialization."""
        assert self.logger.name == "test_logger"
        assert self.logger.level == LogLevel.DEBUG
        assert self.logger.log_format == "json"
    
    def test_set_and_get_context(self):
        """Test setting and getting log context."""
        context = create_log_context(
            correlation_id="test-123",
            user_id="user-456"
        )
        
        self.logger.set_context(context)
        retrieved_context = self.logger.get_context()
        
        assert retrieved_context.correlation_id == "test-123"
        assert retrieved_context.user_id == "user-456"
    
    def test_clear_context(self):
        """Test clearing log context."""
        context = create_log_context(correlation_id="test-123")
        self.logger.set_context(context)
        
        assert self.logger.get_context() is not None
        
        self.logger.clear_context()
        assert self.logger.get_context() is None
    
    def test_log_with_context(self):
        """Test logging with context."""
        context = create_log_context(
            correlation_id="test-123",
            operation="test_operation"
        )
        
        self.logger.log(LogLevel.INFO, "Test message", context)
        
        recent_logs = self.logger.get_recent_logs(1)
        assert len(recent_logs) == 1
        
        log_entry = recent_logs[0]
        assert log_entry.message == "Test message"
        assert log_entry.level == LogLevel.INFO
        assert log_entry.context.correlation_id == "test-123"
    
    def test_log_levels(self):
        """Test different log levels."""
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warn("Warning message")
        self.logger.error("Error message")
        self.logger.fatal("Fatal message")
        
        recent_logs = self.logger.get_recent_logs(5)
        assert len(recent_logs) == 5
        
        levels = [log.level for log in recent_logs]
        assert LogLevel.DEBUG in levels
        assert LogLevel.INFO in levels
        assert LogLevel.WARN in levels
        assert LogLevel.ERROR in levels
        assert LogLevel.FATAL in levels
    
    def test_log_with_exception(self):
        """Test logging with exception."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            self.logger.error("Error occurred", exception=e)
        
        recent_logs = self.logger.get_recent_logs(1)
        log_entry = recent_logs[0]
        
        assert log_entry.exception == "Test exception"
        assert log_entry.stack_trace is not None
    
    def test_level_filtering(self):
        """Test log level filtering."""
        logger = StructuredLogger(level=LogLevel.WARN)
        
        logger.debug("Debug message")  # Should be filtered
        logger.info("Info message")   # Should be filtered
        logger.warn("Warning message")  # Should be logged
        logger.error("Error message")   # Should be logged
        
        recent_logs = logger.get_recent_logs(10)
        assert len(recent_logs) == 2
        
        levels = [log.level for log in recent_logs]
        assert LogLevel.WARN in levels
        assert LogLevel.ERROR in levels
        assert LogLevel.DEBUG not in levels
        assert LogLevel.INFO not in levels


class TestMetricsCollector:
    """Test cases for MetricsCollector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(retention_hours=1)
    
    def test_record_metric(self):
        """Test recording metrics."""
        self.collector.record_metric(
            "test_metric",
            42.5,
            tags={"component": "test"},
            metric_type="gauge",
            unit="ms"
        )
        
        metrics = self.collector.get_metrics("test_metric")
        assert len(metrics) == 1
        
        metric = metrics[0]
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.tags["component"] == "test"
        assert metric.metric_type == "gauge"
        assert metric.unit == "ms"
    
    def test_increment_counter(self):
        """Test incrementing counter metrics."""
        self.collector.increment_counter("requests", {"endpoint": "/api"})
        self.collector.increment_counter("requests", {"endpoint": "/api"})
        
        metrics = self.collector.get_metrics("requests")
        assert len(metrics) == 2
        
        for metric in metrics:
            assert metric.value == 1
            assert metric.metric_type == "counter"
    
    def test_record_timer(self):
        """Test recording timer metrics."""
        self.collector.record_timer("response_time", 150.5, {"method": "GET"})
        
        metrics = self.collector.get_metrics("response_time")
        assert len(metrics) == 1
        
        metric = metrics[0]
        assert metric.value == 150.5
        assert metric.metric_type == "timer"
        assert metric.unit == "ms"
    
    def test_record_gauge(self):
        """Test recording gauge metrics."""
        self.collector.record_gauge("memory_usage", 1024.0, {"host": "server1"})
        
        metrics = self.collector.get_metrics("memory_usage")
        assert len(metrics) == 1
        
        metric = metrics[0]
        assert metric.value == 1024.0
        assert metric.metric_type == "gauge"
    
    def test_application_metrics(self):
        """Test application-specific metrics."""
        self.collector.record_document_indexed()
        self.collector.record_search_performed()
        self.collector.record_embedding_generated()
        self.collector.record_cache_hit()
        self.collector.record_cache_miss()
        
        system_metrics = self.collector.get_system_metrics()
        
        assert system_metrics.documents_indexed == 1
        assert system_metrics.searches_performed == 1
        assert system_metrics.embeddings_generated == 1
        assert system_metrics.cache_hit_rate == 0.5  # 1 hit, 1 miss
    
    def test_request_tracking(self):
        """Test request tracking."""
        self.collector.record_request(100.0, error=False)
        self.collector.record_request(200.0, error=True)
        self.collector.record_request(150.0, error=False)
        
        system_metrics = self.collector.get_system_metrics()
        
        assert system_metrics.request_count == 3
        assert system_metrics.error_count == 1
        assert system_metrics.avg_response_time_ms == 150.0  # (100 + 200 + 150) / 3
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_get_system_metrics(self, mock_disk, mock_cpu, mock_memory):
        """Test getting system metrics."""
        # Mock system information
        mock_memory.return_value = MagicMock(used=1024 * 1024 * 512)  # 512 MB
        mock_cpu.return_value = 75.5
        mock_disk.return_value = MagicMock(used=1024 * 1024 * 1024 * 10)  # 10 GB
        
        system_metrics = self.collector.get_system_metrics()
        
        assert system_metrics.memory_usage_mb == 512.0
        assert system_metrics.cpu_usage_percent == 75.5
        assert system_metrics.disk_usage_mb == 10240.0  # 10 GB in MB
    
    def test_get_metrics_with_time_filter(self):
        """Test getting metrics with time filter."""
        # Record metrics at different times
        self.collector.record_metric("test_metric", 1.0)
        
        # Mock old metric
        old_metric = Metric(
            name="test_metric",
            value=2.0,
            timestamp=datetime.utcnow() - timedelta(hours=2)
        )
        self.collector._metrics["test_metric"].appendleft(old_metric)
        
        # Get metrics from last hour
        recent_metrics = self.collector.get_metrics("test_metric", hours=1)
        assert len(recent_metrics) == 1
        assert recent_metrics[0].value == 1.0
        
        # Get all metrics
        all_metrics = self.collector.get_metrics("test_metric", hours=24)
        assert len(all_metrics) == 2


class TestTracingManager:
    """Test cases for TracingManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracing = TracingManager(service_name="test-service")
    
    def test_start_trace(self):
        """Test starting a trace."""
        trace_context = self.tracing.start_trace("test_operation")
        
        assert trace_context.operation_name == "test_operation"
        assert trace_context.tags["service.name"] == "test-service"
        assert trace_context.span_id in self.tracing._active_traces
    
    def test_end_trace(self):
        """Test ending a trace."""
        trace_context = self.tracing.start_trace("test_operation")
        span_id = trace_context.span_id
        
        self.tracing.end_trace(trace_context, TraceStatus.SUCCESS)
        
        assert span_id not in self.tracing._active_traces
        assert len(self.tracing._completed_traces) == 1
        assert trace_context.status == TraceStatus.SUCCESS
    
    def test_nested_traces(self):
        """Test nested trace contexts."""
        parent_context = self.tracing.start_trace("parent_operation")
        child_context = self.tracing.start_trace("child_operation", parent_context)
        
        assert child_context.trace_id == parent_context.trace_id
        assert child_context.parent_span_id == parent_context.span_id
        assert child_context.span_id != parent_context.span_id
    
    def test_trace_operation_context_manager(self):
        """Test trace operation context manager."""
        with self.tracing.trace_operation("test_operation") as trace_context:
            assert trace_context.operation_name == "test_operation"
            assert trace_context.span_id in self.tracing._active_traces
        
        # After context manager, trace should be completed
        assert trace_context.span_id not in self.tracing._active_traces
        assert trace_context.status == TraceStatus.SUCCESS
    
    def test_trace_operation_with_exception(self):
        """Test trace operation context manager with exception."""
        with pytest.raises(ValueError):
            with self.tracing.trace_operation("test_operation") as trace_context:
                raise ValueError("Test error")
        
        assert trace_context.status == TraceStatus.ERROR
        assert len(trace_context.logs) > 0
        assert "Error: Test error" in trace_context.logs[0]["message"]
    
    def test_current_trace_context(self):
        """Test getting current trace context."""
        assert self.tracing.get_current_trace() is None
        
        trace_context = self.tracing.start_trace("test_operation")
        current_trace = self.tracing.get_current_trace()
        
        assert current_trace == trace_context
        
        self.tracing.end_trace(trace_context)
        assert self.tracing.get_current_trace() is None
    
    def test_add_trace_log_and_tag(self):
        """Test adding logs and tags to current trace."""
        trace_context = self.tracing.start_trace("test_operation")
        
        self.tracing.add_trace_log("Test log message", "info", extra="data")
        self.tracing.add_trace_tag("component", "test")
        
        assert len(trace_context.logs) == 1
        assert trace_context.logs[0]["message"] == "Test log message"
        assert trace_context.tags["component"] == "test"
        
        self.tracing.end_trace(trace_context)
    
    def test_get_traces(self):
        """Test getting completed traces."""
        # Start and end multiple traces
        for i in range(3):
            trace_context = self.tracing.start_trace(f"operation_{i}")
            self.tracing.end_trace(trace_context)
        
        traces = self.tracing.get_traces()
        assert len(traces) == 3
        
        # Test limit
        traces = self.tracing.get_traces(limit=2)
        assert len(traces) == 2


class TestObservabilityManager:
    """Test cases for ObservabilityManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ObservabilityConfig(
            log_level="DEBUG",
            log_format="json",
            metrics_enabled=True,
            tracing_enabled=True,
            performance_monitoring_enabled=False  # Disable for tests
        )
        self.manager = ObservabilityManager(self.config)
    
    def test_initialization(self):
        """Test observability manager initialization."""
        assert self.manager.config == self.config
        assert self.manager.logger is not None
        assert self.manager.metrics is not None
        assert self.manager.tracing is not None
    
    def test_log_event(self):
        """Test logging events."""
        context = create_log_context(
            correlation_id="test-123",
            operation="test_op"
        )
        
        self.manager.log_event(
            "INFO",
            "Test message",
            context,
            extra_field="value"
        )
        
        recent_logs = self.manager.get_recent_logs(1)
        assert len(recent_logs) == 1
        
        log_entry = recent_logs[0]
        assert log_entry.message == "Test message"
        assert log_entry.level == LogLevel.INFO
        assert log_entry.context.correlation_id == "test-123"
    
    def test_record_metric(self):
        """Test recording metrics."""
        self.manager.record_metric(
            "test_metric",
            42.0,
            tags={"component": "test"}
        )
        
        metrics = self.manager.get_metrics("test_metric")
        assert len(metrics) == 1
        assert metrics[0].value == 42.0
    
    def test_tracing_operations(self):
        """Test tracing operations."""
        trace_context = self.manager.start_trace("test_operation")
        assert trace_context is not None
        assert trace_context.operation_name == "test_operation"
        
        self.manager.end_trace(trace_context, "success")
        assert trace_context.status == TraceStatus.SUCCESS
    
    def test_trace_operation_context_manager(self):
        """Test trace operation context manager."""
        with self.manager.trace_operation("test_operation") as trace_context:
            assert trace_context is not None
            assert trace_context.operation_name == "test_operation"
    
    def test_log_context_management(self):
        """Test log context management."""
        context = create_log_context(
            correlation_id="test-123",
            user_id="user-456"
        )
        
        self.manager.set_log_context(context)
        
        # Log without explicit context - should use thread-local context
        self.manager.log_event("INFO", "Test message")
        
        recent_logs = self.manager.get_recent_logs(1)
        log_entry = recent_logs[0]
        assert log_entry.context.correlation_id == "test-123"
        assert log_entry.context.user_id == "user-456"
        
        self.manager.clear_log_context()
    
    def test_health_check(self):
        """Test health check functionality."""
        # Register a health check
        def test_health_check():
            return True, "Test component is healthy", {"status": "ok"}
        
        self.manager.register_health_check("test_component", test_health_check)
        
        health_status = self.manager.health_check()
        
        assert health_status.overall_health is True
        assert "test_component" in health_status.components
        assert health_status.components["test_component"]["healthy"] is True
    
    def test_health_check_with_failure(self):
        """Test health check with component failure."""
        def failing_health_check():
            return False, "Test component is unhealthy", {"error": "connection_failed"}
        
        self.manager.register_health_check("failing_component", failing_health_check)
        
        health_status = self.manager.health_check()
        
        assert health_status.overall_health is False
        assert health_status.status == "degraded"
        assert "failing_component" in health_status.components
        assert health_status.components["failing_component"]["healthy"] is False
    
    def test_health_check_with_exception(self):
        """Test health check with exception in check function."""
        def exception_health_check():
            raise Exception("Health check failed")
        
        self.manager.register_health_check("exception_component", exception_health_check)
        
        health_status = self.manager.health_check()
        
        assert health_status.overall_health is False
        assert "exception_component" in health_status.components
        assert health_status.components["exception_component"]["healthy"] is False
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_get_system_metrics(self, mock_disk, mock_cpu, mock_memory):
        """Test getting system metrics."""
        # Mock system information
        mock_memory.return_value = MagicMock(used=1024 * 1024 * 256)  # 256 MB
        mock_cpu.return_value = 50.0
        mock_disk.return_value = MagicMock(used=1024 * 1024 * 1024 * 5)  # 5 GB
        
        system_metrics = self.manager.get_system_metrics()
        
        assert system_metrics.memory_usage_mb == 256.0
        assert system_metrics.cpu_usage_percent == 50.0
        assert system_metrics.disk_usage_mb == 5120.0
    
    def test_disabled_tracing(self):
        """Test behavior when tracing is disabled."""
        config = ObservabilityConfig(tracing_enabled=False)
        manager = ObservabilityManager(config)
        
        assert manager.tracing is None
        
        trace_context = manager.start_trace("test_operation")
        assert trace_context is None
        
        # Should not raise exception
        manager.end_trace(None, "success")
        
        with manager.trace_operation("test_operation") as trace_context:
            assert trace_context is None
    
    def test_disabled_metrics(self):
        """Test behavior when metrics are disabled."""
        config = ObservabilityConfig(metrics_enabled=False)
        manager = ObservabilityManager(config)
        
        # Should not raise exception
        manager.record_metric("test_metric", 42.0)
        
        # Metrics should still be available (just not recorded via record_metric)
        metrics = manager.get_metrics()
        assert isinstance(metrics, list)
    
    def test_context_manager(self):
        """Test observability manager as context manager."""
        with ObservabilityManager(self.config) as manager:
            assert manager is not None
            manager.log_event("INFO", "Test message")
        
        # Should complete without errors


class TestJsonFormatter:
    """Test cases for JsonFormatter."""
    
    def test_format_json_message(self):
        """Test formatting JSON messages."""
        formatter = JsonFormatter()
        
        # Create a mock log record with JSON message
        record = MagicMock()
        record.getMessage.return_value = '{"level": "INFO", "message": "test"}'
        
        result = formatter.format(record)
        assert result == '{"level": "INFO", "message": "test"}'
    
    def test_format_regular_message(self):
        """Test formatting regular messages."""
        formatter = JsonFormatter()
        
        # Create a mock log record with regular message
        record = MagicMock()
        record.getMessage.return_value = "Regular log message"
        record.levelname = "INFO"
        record.name = "test_logger"
        
        result = formatter.format(record)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["message"] == "Regular log message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test_logger"


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_log_context(self):
        """Test creating log context."""
        context = create_log_context(
            correlation_id="test-123",
            user_id="user-456",
            operation="test_op"
        )
        
        assert context.correlation_id == "test-123"
        assert context.user_id == "user-456"
        assert context.operation == "test_op"
    
    def test_create_log_context_with_auto_correlation_id(self):
        """Test creating log context with auto-generated correlation ID."""
        context = create_log_context(user_id="user-456")
        
        assert context.correlation_id is not None
        assert len(context.correlation_id) > 0
        assert context.user_id == "user-456"
    
    def test_create_trace_context(self):
        """Test creating trace context."""
        context = create_trace_context(
            operation_name="test_operation",
            trace_id="trace-123",
            parent_span_id="parent-456"
        )
        
        assert context.operation_name == "test_operation"
        assert context.trace_id == "trace-123"
        assert context.parent_span_id == "parent-456"
        assert context.span_id is not None
    
    def test_create_trace_context_with_auto_ids(self):
        """Test creating trace context with auto-generated IDs."""
        context = create_trace_context("test_operation")
        
        assert context.operation_name == "test_operation"
        assert context.trace_id is not None
        assert context.span_id is not None
        assert len(context.trace_id) > 0
        assert len(context.span_id) > 0