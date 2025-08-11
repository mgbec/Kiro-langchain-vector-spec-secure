"""
Unit tests for distributed tracing system.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from langchain_vector_db.services.tracing import (
    Span, SpanEvent, SpanLink, TraceExporter, ConsoleTraceExporter,
    InMemoryTraceExporter, HTTPTraceExporter, SpanProcessor,
    BatchSpanProcessor, DistributedTracer, TracingInstrumentation,
    create_tracer
)
from langchain_vector_db.models.observability import TraceStatus


class TestSpanEvent:
    """Test cases for SpanEvent."""
    
    def test_initialization(self):
        """Test span event initialization."""
        timestamp = datetime.utcnow()
        event = SpanEvent(
            timestamp=timestamp,
            name="test_event",
            attributes={"key": "value"}
        )
        
        assert event.timestamp == timestamp
        assert event.name == "test_event"
        assert event.attributes["key"] == "value"
    
    def test_to_dict(self):
        """Test span event dictionary conversion."""
        timestamp = datetime.utcnow()
        event = SpanEvent(
            timestamp=timestamp,
            name="test_event",
            attributes={"key": "value"}
        )
        
        result = event.to_dict()
        
        assert result["timestamp"] == timestamp.isoformat()
        assert result["name"] == "test_event"
        assert result["attributes"]["key"] == "value"


class TestSpanLink:
    """Test cases for SpanLink."""
    
    def test_initialization(self):
        """Test span link initialization."""
        link = SpanLink(
            trace_id="trace123",
            span_id="span456",
            attributes={"relationship": "child"}
        )
        
        assert link.trace_id == "trace123"
        assert link.span_id == "span456"
        assert link.attributes["relationship"] == "child"
    
    def test_to_dict(self):
        """Test span link dictionary conversion."""
        link = SpanLink(
            trace_id="trace123",
            span_id="span456",
            attributes={"relationship": "child"}
        )
        
        result = link.to_dict()
        
        assert result["trace_id"] == "trace123"
        assert result["span_id"] == "span456"
        assert result["attributes"]["relationship"] == "child"


class TestSpan:
    """Test cases for Span."""
    
    def test_initialization(self):
        """Test span initialization."""
        start_time = datetime.utcnow()
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id="parent789",
            operation_name="test_operation",
            start_time=start_time
        )
        
        assert span.trace_id == "trace123"
        assert span.span_id == "span456"
        assert span.parent_span_id == "parent789"
        assert span.operation_name == "test_operation"
        assert span.start_time == start_time
        assert span.status == TraceStatus.SUCCESS
        assert span.resource["service.name"] == "vector-database"
    
    def test_set_attribute(self):
        """Test setting span attributes."""
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        
        span.set_attribute("key", "value")
        span.set_attributes({"key2": "value2", "key3": "value3"})
        
        assert span.attributes["key"] == "value"
        assert span.attributes["key2"] == "value2"
        assert span.attributes["key3"] == "value3"
    
    def test_add_event(self):
        """Test adding events to span."""
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        
        span.add_event("test_event", {"key": "value"})
        
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "test_event"
        assert event.attributes["key"] == "value"
    
    def test_add_link(self):
        """Test adding links to span."""
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        
        span.add_link("other_trace", "other_span", {"relationship": "follows"})
        
        assert len(span.links) == 1
        link = span.links[0]
        assert link.trace_id == "other_trace"
        assert link.span_id == "other_span"
        assert link.attributes["relationship"] == "follows"
    
    def test_set_status(self):
        """Test setting span status."""
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        
        span.set_status(TraceStatus.ERROR, "Something went wrong")
        
        assert span.status == TraceStatus.ERROR
        assert span.attributes["status.description"] == "Something went wrong"
    
    def test_record_exception(self):
        """Test recording exceptions in span."""
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        
        exception = ValueError("Test error")
        span.record_exception(exception)
        
        assert span.status == TraceStatus.ERROR
        assert len(span.events) == 1
        
        event = span.events[0]
        assert event.name == "exception"
        assert event.attributes["exception.type"] == "ValueError"
        assert event.attributes["exception.message"] == "Test error"
    
    def test_finish(self):
        """Test finishing a span."""
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        
        assert not span.is_finished()
        
        end_time = datetime.utcnow()
        span.finish(end_time)
        
        assert span.is_finished()
        assert span.end_time == end_time
    
    def test_get_duration_ms(self):
        """Test getting span duration."""
        start_time = datetime.utcnow()
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=start_time
        )
        
        # No duration before finishing
        assert span.get_duration_ms() is None
        
        # Duration after finishing
        end_time = start_time + timedelta(milliseconds=100)
        span.finish(end_time)
        
        duration = span.get_duration_ms()
        assert duration == 100.0
    
    def test_to_dict(self):
        """Test span dictionary conversion."""
        start_time = datetime.utcnow()
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id="parent789",
            operation_name="test_operation",
            start_time=start_time
        )
        
        span.set_attribute("key", "value")
        span.add_event("test_event")
        span.finish()
        
        result = span.to_dict()
        
        assert result["trace_id"] == "trace123"
        assert result["span_id"] == "span456"
        assert result["parent_span_id"] == "parent789"
        assert result["name"] == "test_operation"
        assert result["status"]["code"] == "STATUS_CODE_OK"
        assert result["attributes"]["key"] == "value"
        assert len(result["events"]) == 1
        assert "duration_ms" in result


class TestTraceExporters:
    """Test cases for trace exporters."""
    
    def test_console_exporter(self):
        """Test console trace exporter."""
        exporter = ConsoleTraceExporter()
        
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        span.finish()
        
        # Should not raise exception
        result = exporter.export([span])
        assert result is True
    
    def test_in_memory_exporter(self):
        """Test in-memory trace exporter."""
        exporter = InMemoryTraceExporter(max_spans=10)
        
        spans = []
        for i in range(5):
            span = Span(
                trace_id=f"trace{i}",
                span_id=f"span{i}",
                parent_span_id=None,
                operation_name=f"operation{i}",
                start_time=datetime.utcnow()
            )
            span.finish()
            spans.append(span)
        
        result = exporter.export(spans)
        assert result is True
        
        stored_spans = exporter.get_spans()
        assert len(stored_spans) == 5
        assert stored_spans[0].trace_id == "trace0"
        
        exporter.clear()
        assert len(exporter.get_spans()) == 0
    
    @patch('requests.post')
    def test_http_exporter_success(self, mock_post):
        """Test HTTP trace exporter success."""
        mock_post.return_value.status_code = 200
        
        exporter = HTTPTraceExporter(
            endpoint="http://localhost:4318/v1/traces",
            headers={"Authorization": "Bearer token"}
        )
        
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        span.finish()
        
        result = exporter.export([span])
        assert result is True
        
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["headers"]["Authorization"] == "Bearer token"
    
    @patch('requests.post')
    def test_http_exporter_failure(self, mock_post):
        """Test HTTP trace exporter failure."""
        mock_post.return_value.status_code = 500
        
        exporter = HTTPTraceExporter(endpoint="http://localhost:4318/v1/traces")
        
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        span.finish()
        
        result = exporter.export([span])
        assert result is False


class TestBatchSpanProcessor:
    """Test cases for BatchSpanProcessor."""
    
    def test_initialization(self):
        """Test batch span processor initialization."""
        exporter = InMemoryTraceExporter()
        processor = BatchSpanProcessor(
            exporter=exporter,
            max_batch_size=10,
            batch_timeout_ms=1000
        )
        
        assert processor.exporter == exporter
        assert processor.max_batch_size == 10
        assert processor.batch_timeout_ms == 1000
    
    def test_span_processing(self):
        """Test span processing through batch processor."""
        exporter = InMemoryTraceExporter()
        processor = BatchSpanProcessor(
            exporter=exporter,
            max_batch_size=2,
            batch_timeout_ms=100
        )
        
        # Create and finish spans
        spans = []
        for i in range(3):
            span = Span(
                trace_id=f"trace{i}",
                span_id=f"span{i}",
                parent_span_id=None,
                operation_name=f"operation{i}",
                start_time=datetime.utcnow()
            )
            span.finish()
            spans.append(span)
            processor.on_end(span)
        
        # Wait for batch processing
        time.sleep(0.2)
        
        # Check that spans were exported
        exported_spans = exporter.get_spans()
        assert len(exported_spans) >= 2  # At least one batch should be processed
        
        processor.shutdown()


class TestDistributedTracer:
    """Test cases for DistributedTracer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = InMemoryTraceExporter()
        self.processor = BatchSpanProcessor(self.exporter, batch_timeout_ms=50)
        self.tracer = DistributedTracer(
            service_name="test-service",
            service_version="1.0.0",
            processors=[self.processor]
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.tracer.shutdown()
    
    def test_initialization(self):
        """Test tracer initialization."""
        assert self.tracer.service_name == "test-service"
        assert self.tracer.service_version == "1.0.0"
        assert len(self.tracer.processors) == 1
        assert self.tracer.sampling_rate == 1.0
    
    def test_start_span(self):
        """Test starting a span."""
        span = self.tracer.start_span(
            "test_operation",
            attributes={"key": "value"}
        )
        
        assert span.operation_name == "test_operation"
        assert span.attributes["key"] == "value"
        assert span.resource["service.name"] == "test-service"
        assert span.span_id in self.tracer._active_spans
    
    def test_end_span(self):
        """Test ending a span."""
        span = self.tracer.start_span("test_operation")
        span_id = span.span_id
        
        assert span_id in self.tracer._active_spans
        
        self.tracer.end_span(span)
        
        assert span.is_finished()
        assert span_id not in self.tracer._active_spans
    
    def test_nested_spans(self):
        """Test nested span creation."""
        parent_span = self.tracer.start_span("parent_operation")
        child_span = self.tracer.start_span("child_operation", parent_context=parent_span)
        
        assert child_span.trace_id == parent_span.trace_id
        assert child_span.parent_span_id == parent_span.span_id
        assert child_span.span_id != parent_span.span_id
        
        self.tracer.end_span(child_span)
        self.tracer.end_span(parent_span)
    
    def test_current_span_management(self):
        """Test current span management."""
        assert self.tracer.get_current_span() is None
        
        span = self.tracer.start_span("test_operation")
        assert self.tracer.get_current_span() == span
        
        self.tracer.end_span(span)
        assert self.tracer.get_current_span() is None
    
    def test_start_as_current_span_context_manager(self):
        """Test start_as_current_span context manager."""
        with self.tracer.start_as_current_span("test_operation") as span:
            assert span.operation_name == "test_operation"
            assert self.tracer.get_current_span() == span
            assert not span.is_finished()
        
        assert span.is_finished()
        assert self.tracer.get_current_span() is None
    
    def test_context_manager_with_exception(self):
        """Test context manager with exception handling."""
        with pytest.raises(ValueError):
            with self.tracer.start_as_current_span("test_operation") as span:
                raise ValueError("Test error")
        
        assert span.is_finished()
        assert span.status == TraceStatus.ERROR
        assert len(span.events) == 1
        assert span.events[0].name == "exception"
    
    def test_sampling(self):
        """Test trace sampling."""
        # Set low sampling rate
        self.tracer.set_sampling_rate(0.0)
        
        span = self.tracer.start_span("test_operation")
        
        # Should create non-recording span
        assert span.trace_id == "00000000000000000000000000000000"
        assert span.span_id == "0000000000000000"
    
    def test_get_active_spans(self):
        """Test getting active spans."""
        assert len(self.tracer.get_active_spans()) == 0
        
        span1 = self.tracer.start_span("operation1")
        span2 = self.tracer.start_span("operation2")
        
        active_spans = self.tracer.get_active_spans()
        assert len(active_spans) == 2
        
        span_ids = [span.span_id for span in active_spans]
        assert span1.span_id in span_ids
        assert span2.span_id in span_ids
        
        self.tracer.end_span(span1)
        self.tracer.end_span(span2)


class TestTracingInstrumentation:
    """Test cases for TracingInstrumentation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = InMemoryTraceExporter()
        self.processor = BatchSpanProcessor(self.exporter, batch_timeout_ms=50)
        self.tracer = DistributedTracer(processors=[self.processor])
        self.instrumentation = TracingInstrumentation(self.tracer)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.tracer.shutdown()
    
    def test_instrument_function(self):
        """Test function instrumentation."""
        def test_function(x, y):
            return x + y
        
        instrumented_func = self.instrumentation.instrument_function(
            test_function,
            "custom_operation"
        )
        
        result = instrumented_func(2, 3)
        assert result == 5
        
        # Wait for span processing
        time.sleep(0.1)
        
        # Check that span was created
        spans = self.exporter.get_spans()
        assert len(spans) >= 1
        
        span = spans[0]
        assert span.operation_name == "custom_operation"
        assert span.attributes["function.name"] == "test_function"
        assert span.attributes["function.args_count"] == 2
    
    def test_instrument_function_with_exception(self):
        """Test function instrumentation with exception."""
        def failing_function():
            raise ValueError("Test error")
        
        instrumented_func = self.instrumentation.instrument_function(failing_function)
        
        with pytest.raises(ValueError):
            instrumented_func()
        
        # Wait for span processing
        time.sleep(0.1)
        
        # Check that span recorded the exception
        spans = self.exporter.get_spans()
        assert len(spans) >= 1
        
        span = spans[0]
        assert span.status == TraceStatus.ERROR
        assert len(span.events) == 1
        assert span.events[0].name == "exception"
    
    def test_instrument_class(self):
        """Test class instrumentation."""
        class TestClass:
            def method1(self, x):
                return x * 2
            
            def method2(self, x, y):
                return x + y
            
            def _private_method(self):
                return "private"
        
        self.instrumentation.instrument_class(TestClass, ["method1", "method2"])
        
        instance = TestClass()
        result1 = instance.method1(5)
        result2 = instance.method2(3, 4)
        
        assert result1 == 10
        assert result2 == 7
        
        # Wait for span processing
        time.sleep(0.1)
        
        # Check that spans were created
        spans = self.exporter.get_spans()
        assert len(spans) >= 2
        
        operation_names = [span.operation_name for span in spans]
        assert "TestClass.method1" in operation_names
        assert "TestClass.method2" in operation_names


class TestCreateTracer:
    """Test cases for create_tracer function."""
    
    def test_create_tracer_console(self):
        """Test creating tracer with console exporter."""
        tracer = create_tracer(
            service_name="test-service",
            exporter_type="console"
        )
        
        assert tracer.service_name == "test-service"
        assert len(tracer.processors) == 1
        
        tracer.shutdown()
    
    def test_create_tracer_memory(self):
        """Test creating tracer with memory exporter."""
        tracer = create_tracer(
            service_name="test-service",
            exporter_type="memory",
            exporter_config={"max_spans": 100}
        )
        
        assert tracer.service_name == "test-service"
        assert len(tracer.processors) == 1
        
        tracer.shutdown()
    
    def test_create_tracer_http(self):
        """Test creating tracer with HTTP exporter."""
        tracer = create_tracer(
            service_name="test-service",
            exporter_type="http",
            exporter_config={
                "endpoint": "http://localhost:4318/v1/traces",
                "headers": {"Authorization": "Bearer token"}
            }
        )
        
        assert tracer.service_name == "test-service"
        assert len(tracer.processors) == 1
        
        tracer.shutdown()
    
    def test_create_tracer_invalid_type(self):
        """Test creating tracer with invalid exporter type."""
        with pytest.raises(ValueError, match="Unknown exporter type"):
            create_tracer(exporter_type="invalid")


class TestIntegration:
    """Integration tests for distributed tracing."""
    
    def test_end_to_end_tracing(self):
        """Test end-to-end tracing workflow."""
        exporter = InMemoryTraceExporter()
        processor = BatchSpanProcessor(exporter, batch_timeout_ms=50)
        tracer = DistributedTracer(processors=[processor])
        
        try:
            # Create nested spans
            with tracer.start_as_current_span("parent_operation") as parent_span:
                parent_span.set_attribute("operation.type", "parent")
                parent_span.add_event("parent_started")
                
                with tracer.start_as_current_span("child_operation") as child_span:
                    child_span.set_attribute("operation.type", "child")
                    child_span.add_event("child_started")
                    
                    # Simulate some work
                    time.sleep(0.01)
                    
                    child_span.add_event("child_completed")
                
                parent_span.add_event("parent_completed")
            
            # Wait for processing
            time.sleep(0.1)
            
            # Verify spans were exported
            spans = exporter.get_spans()
            assert len(spans) >= 2
            
            # Find parent and child spans
            parent_spans = [s for s in spans if s.operation_name == "parent_operation"]
            child_spans = [s for s in spans if s.operation_name == "child_operation"]
            
            assert len(parent_spans) == 1
            assert len(child_spans) == 1
            
            parent_span = parent_spans[0]
            child_span = child_spans[0]
            
            # Verify relationship
            assert child_span.trace_id == parent_span.trace_id
            assert child_span.parent_span_id == parent_span.span_id
            
            # Verify attributes and events
            assert parent_span.attributes["operation.type"] == "parent"
            assert child_span.attributes["operation.type"] == "child"
            assert len(parent_span.events) == 2
            assert len(child_span.events) == 2
            
        finally:
            tracer.shutdown()