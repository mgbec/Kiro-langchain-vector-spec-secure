"""
OpenTelemetry-compatible distributed tracing system.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from ..models.observability import TraceContext, TraceStatus, generate_trace_id, generate_span_id


@dataclass
class SpanEvent:
    """Represents an event within a span."""
    
    timestamp: datetime
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span event to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "attributes": self.attributes
        }


@dataclass
class SpanLink:
    """Represents a link between spans."""
    
    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span link to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "attributes": self.attributes
        }


@dataclass
class Span:
    """OpenTelemetry-compatible span implementation."""
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TraceStatus = TraceStatus.SUCCESS
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    resource: Dict[str, Any] = field(default_factory=dict)
    instrumentation_scope: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize span with default values."""
        if not self.resource:
            self.resource = {
                "service.name": "vector-database",
                "service.version": "1.0.0"
            }
        
        if not self.instrumentation_scope:
            self.instrumentation_scope = {
                "name": "langchain_vector_db",
                "version": "1.0.0"
            }
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value
    
    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set multiple span attributes."""
        self.attributes.update(attributes)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        event = SpanEvent(
            timestamp=datetime.utcnow(),
            name=name,
            attributes=attributes or {}
        )
        self.events.append(event)
    
    def add_link(self, trace_id: str, span_id: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add a link to another span."""
        link = SpanLink(
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes or {}
        )
        self.links.append(link)
    
    def set_status(self, status: TraceStatus, description: Optional[str] = None) -> None:
        """Set the span status."""
        self.status = status
        if description:
            self.set_attribute("status.description", description)
    
    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the span."""
        self.set_status(TraceStatus.ERROR, str(exception))
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.stacktrace": self._get_exception_stacktrace(exception)
            }
        )
    
    def finish(self, end_time: Optional[datetime] = None) -> None:
        """Finish the span."""
        self.end_time = end_time or datetime.utcnow()
    
    def get_duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return None
    
    def is_finished(self) -> bool:
        """Check if span is finished."""
        return self.end_time is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to OpenTelemetry-compatible dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.operation_name,
            "kind": "SPAN_KIND_INTERNAL",
            "start_time_unix_nano": int(self.start_time.timestamp() * 1_000_000_000),
            "end_time_unix_nano": int(self.end_time.timestamp() * 1_000_000_000) if self.end_time else None,
            "status": {
                "code": self._status_to_otel_code(),
                "message": self.attributes.get("status.description", "")
            },
            "attributes": self.attributes,
            "events": [event.to_dict() for event in self.events],
            "links": [link.to_dict() for link in self.links],
            "resource": self.resource,
            "instrumentation_scope": self.instrumentation_scope,
            "duration_ms": self.get_duration_ms()
        }
    
    def _get_exception_stacktrace(self, exception: Exception) -> str:
        """Get exception stacktrace."""
        import traceback
        return ''.join(traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ))
    
    def _status_to_otel_code(self) -> str:
        """Convert TraceStatus to OpenTelemetry status code."""
        status_mapping = {
            TraceStatus.SUCCESS: "STATUS_CODE_OK",
            TraceStatus.ERROR: "STATUS_CODE_ERROR",
            TraceStatus.TIMEOUT: "STATUS_CODE_ERROR",
            TraceStatus.CANCELLED: "STATUS_CODE_ERROR"
        }
        return status_mapping.get(self.status, "STATUS_CODE_UNSET")


class TraceExporter:
    """Base class for trace exporters."""
    
    def export(self, spans: List[Span]) -> bool:
        """
        Export spans to external system.
        
        Args:
            spans: List of spans to export
            
        Returns:
            True if export was successful
        """
        raise NotImplementedError
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleTraceExporter(TraceExporter):
    """Console trace exporter for debugging."""
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans to console."""
        for span in spans:
            print(f"TRACE: {json.dumps(span.to_dict(), indent=2)}")
        return True


class InMemoryTraceExporter(TraceExporter):
    """In-memory trace exporter for testing."""
    
    def __init__(self, max_spans: int = 1000):
        """Initialize in-memory exporter."""
        self.max_spans = max_spans
        self.spans: deque = deque(maxlen=max_spans)
        self._lock = threading.Lock()
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans to memory."""
        with self._lock:
            self.spans.extend(spans)
        return True
    
    def get_spans(self) -> List[Span]:
        """Get all stored spans."""
        with self._lock:
            return list(self.spans)
    
    def clear(self) -> None:
        """Clear all stored spans."""
        with self._lock:
            self.spans.clear()


class HTTPTraceExporter(TraceExporter):
    """HTTP trace exporter for OpenTelemetry collectors."""
    
    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ):
        """
        Initialize HTTP trace exporter.
        
        Args:
            endpoint: HTTP endpoint for trace export
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans via HTTP."""
        try:
            import requests
            
            # Convert spans to OpenTelemetry format
            otel_data = {
                "resourceSpans": [{
                    "resource": spans[0].resource if spans else {},
                    "scopeSpans": [{
                        "scope": spans[0].instrumentation_scope if spans else {},
                        "spans": [span.to_dict() for span in spans]
                    }]
                }]
            }
            
            response = requests.post(
                self.endpoint,
                json=otel_data,
                headers=self.headers,
                timeout=self.timeout
            )
            
            return response.status_code == 200
            
        except Exception:
            return False


class SpanProcessor:
    """Base class for span processors."""
    
    def on_start(self, span: Span) -> None:
        """Called when a span starts."""
        pass
    
    def on_end(self, span: Span) -> None:
        """Called when a span ends."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass


class BatchSpanProcessor(SpanProcessor):
    """Batch span processor that exports spans in batches."""
    
    def __init__(
        self,
        exporter: TraceExporter,
        max_batch_size: int = 512,
        batch_timeout_ms: int = 5000,
        max_queue_size: int = 2048
    ):
        """
        Initialize batch span processor.
        
        Args:
            exporter: Trace exporter to use
            max_batch_size: Maximum batch size
            batch_timeout_ms: Batch timeout in milliseconds
            max_queue_size: Maximum queue size
        """
        self.exporter = exporter
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_queue_size = max_queue_size
        
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._shutdown = False
        
        # Start background export thread
        self._export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self._export_thread.start()
    
    def on_end(self, span: Span) -> None:
        """Add finished span to export queue."""
        if self._shutdown:
            return
        
        with self._lock:
            if len(self._queue) < self.max_queue_size:
                self._queue.append(span)
    
    def _export_worker(self) -> None:
        """Background worker that exports spans in batches."""
        while not self._shutdown:
            try:
                batch = []
                
                # Collect batch
                with self._lock:
                    while len(batch) < self.max_batch_size and self._queue:
                        batch.append(self._queue.popleft())
                
                # Export batch if not empty
                if batch:
                    self.exporter.export(batch)
                
                # Wait before next batch
                time.sleep(self.batch_timeout_ms / 1000.0)
                
            except Exception:
                # Continue on export errors
                pass
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        self._shutdown = True
        
        # Export remaining spans
        with self._lock:
            if self._queue:
                remaining_spans = list(self._queue)
                self.exporter.export(remaining_spans)
                self._queue.clear()
        
        self.exporter.shutdown()


class DistributedTracer:
    """OpenTelemetry-compatible distributed tracer."""
    
    def __init__(
        self,
        service_name: str = "vector-database",
        service_version: str = "1.0.0",
        processors: Optional[List[SpanProcessor]] = None
    ):
        """
        Initialize distributed tracer.
        
        Args:
            service_name: Name of the service
            service_version: Version of the service
            processors: List of span processors
        """
        self.service_name = service_name
        self.service_version = service_version
        self.processors = processors or []
        
        # Active spans storage
        self._active_spans: Dict[str, Span] = {}
        self._lock = threading.Lock()
        
        # Thread-local storage for current span
        self._local = threading.local()
        
        # Sampling configuration
        self.sampling_rate = 1.0  # Sample all traces by default
        
        # Resource attributes
        self.resource_attributes = {
            "service.name": service_name,
            "service.version": service_version,
            "telemetry.sdk.name": "langchain_vector_db",
            "telemetry.sdk.version": "1.0.0",
            "telemetry.sdk.language": "python"
        }
    
    def start_span(
        self,
        operation_name: str,
        parent_context: Optional[Union[Span, TraceContext]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanLink]] = None
    ) -> Span:
        """
        Start a new span.
        
        Args:
            operation_name: Name of the operation
            parent_context: Parent span or trace context
            attributes: Initial span attributes
            links: Span links
            
        Returns:
            New span instance
        """
        # Check sampling
        if not self._should_sample():
            return self._create_non_recording_span(operation_name)
        
        # Determine trace and parent span IDs
        if isinstance(parent_context, Span):
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        elif isinstance(parent_context, TraceContext):
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            # Check for current span in thread-local storage
            current_span = self.get_current_span()
            if current_span:
                trace_id = current_span.trace_id
                parent_span_id = current_span.span_id
            else:
                trace_id = generate_trace_id()
                parent_span_id = None
        
        # Create new span
        span = Span(
            trace_id=trace_id,
            span_id=generate_span_id(),
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            resource=self.resource_attributes.copy()
        )
        
        # Set initial attributes
        if attributes:
            span.set_attributes(attributes)
        
        # Add links
        if links:
            span.links.extend(links)
        
        # Store as active span
        with self._lock:
            self._active_spans[span.span_id] = span
        
        # Set as current span
        self._local.current_span = span
        
        # Notify processors
        for processor in self.processors:
            try:
                processor.on_start(span)
            except Exception:
                pass  # Don't let processor errors affect tracing
        
        return span
    
    def end_span(self, span: Span) -> None:
        """
        End a span.
        
        Args:
            span: Span to end
        """
        if span.is_finished():
            return
        
        # Finish the span
        span.finish()
        
        # Remove from active spans
        with self._lock:
            self._active_spans.pop(span.span_id, None)
        
        # Clear current span if it matches
        current_span = getattr(self._local, 'current_span', None)
        if current_span and current_span.span_id == span.span_id:
            self._local.current_span = None
        
        # Notify processors
        for processor in self.processors:
            try:
                processor.on_end(span)
            except Exception:
                pass
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span for this thread."""
        return getattr(self._local, 'current_span', None)
    
    def set_current_span(self, span: Optional[Span]) -> None:
        """Set the current active span for this thread."""
        self._local.current_span = span
    
    @contextmanager
    def start_as_current_span(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanLink]] = None
    ):
        """
        Context manager that starts a span and sets it as current.
        
        Args:
            operation_name: Name of the operation
            attributes: Initial span attributes
            links: Span links
        """
        span = self.start_span(operation_name, attributes=attributes, links=links)
        previous_span = self.get_current_span()
        
        try:
            self.set_current_span(span)
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            self.end_span(span)
            self.set_current_span(previous_span)
    
    def add_span_processor(self, processor: SpanProcessor) -> None:
        """Add a span processor."""
        self.processors.append(processor)
    
    def get_active_spans(self) -> List[Span]:
        """Get all currently active spans."""
        with self._lock:
            return list(self._active_spans.values())
    
    def set_sampling_rate(self, rate: float) -> None:
        """
        Set the sampling rate.
        
        Args:
            rate: Sampling rate between 0.0 and 1.0
        """
        self.sampling_rate = max(0.0, min(1.0, rate))
    
    def _should_sample(self) -> bool:
        """Determine if a trace should be sampled."""
        import random
        return random.random() < self.sampling_rate
    
    def _create_non_recording_span(self, operation_name: str) -> Span:
        """Create a non-recording span for unsampled traces."""
        return Span(
            trace_id="00000000000000000000000000000000",
            span_id="0000000000000000",
            parent_span_id=None,
            operation_name=operation_name,
            start_time=datetime.utcnow()
        )
    
    def shutdown(self) -> None:
        """Shutdown the tracer."""
        # End all active spans
        with self._lock:
            active_spans = list(self._active_spans.values())
        
        for span in active_spans:
            self.end_span(span)
        
        # Shutdown processors
        for processor in self.processors:
            try:
                processor.shutdown()
            except Exception:
                pass


class TracingInstrumentation:
    """Automatic instrumentation for common operations."""
    
    def __init__(self, tracer: DistributedTracer):
        """Initialize tracing instrumentation."""
        self.tracer = tracer
    
    def instrument_function(self, func: Callable, operation_name: Optional[str] = None):
        """
        Instrument a function with tracing.
        
        Args:
            func: Function to instrument
            operation_name: Optional operation name (defaults to function name)
            
        Returns:
            Instrumented function
        """
        import functools
        
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span(op_name) as span:
                # Add function metadata
                span.set_attributes({
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                    "function.args_count": len(args),
                    "function.kwargs_count": len(kwargs)
                })
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.result_type", type(result).__name__)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        return wrapper
    
    def instrument_class(self, cls: type, methods: Optional[List[str]] = None):
        """
        Instrument a class with tracing.
        
        Args:
            cls: Class to instrument
            methods: Optional list of method names to instrument
        """
        if methods is None:
            # Instrument all public methods
            methods = [
                name for name in dir(cls)
                if not name.startswith('_') and callable(getattr(cls, name))
            ]
        
        for method_name in methods:
            if hasattr(cls, method_name):
                original_method = getattr(cls, method_name)
                if callable(original_method):
                    instrumented_method = self.instrument_function(
                        original_method,
                        f"{cls.__name__}.{method_name}"
                    )
                    setattr(cls, method_name, instrumented_method)


def create_tracer(
    service_name: str = "vector-database",
    service_version: str = "1.0.0",
    exporter_type: str = "console",
    exporter_config: Optional[Dict[str, Any]] = None
) -> DistributedTracer:
    """
    Create a configured distributed tracer.
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        exporter_type: Type of exporter ("console", "memory", "http")
        exporter_config: Configuration for the exporter
        
    Returns:
        Configured distributed tracer
    """
    # Create exporter
    exporter_config = exporter_config or {}
    
    if exporter_type == "console":
        exporter = ConsoleTraceExporter()
    elif exporter_type == "memory":
        exporter = InMemoryTraceExporter(**exporter_config)
    elif exporter_type == "http":
        exporter = HTTPTraceExporter(**exporter_config)
    else:
        raise ValueError(f"Unknown exporter type: {exporter_type}")
    
    # Create processor
    processor = BatchSpanProcessor(exporter)
    
    # Create tracer
    tracer = DistributedTracer(
        service_name=service_name,
        service_version=service_version,
        processors=[processor]
    )
    
    return tracer