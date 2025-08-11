"""
Enhanced metrics collection and monitoring system.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import statistics

from ..models.observability import Metric, SystemMetrics


@dataclass
class PerformanceMetric:
    """Performance metric with statistical analysis."""
    
    name: str
    values: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a new value to the metric."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.values.append(value)
        self.timestamps.append(timestamp)
        
        # Keep only recent values (last 1000)
        if len(self.values) > 1000:
            self.values = self.values[-1000:]
            self.timestamps = self.timestamps[-1000:]
    
    def get_statistics(self, window_minutes: int = 60) -> Dict[str, float]:
        """Get statistical analysis of the metric values."""
        if not self.values:
            return {}
        
        # Filter values within time window
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_values = [
            value for value, timestamp in zip(self.values, self.timestamps)
            if timestamp > cutoff
        ]
        
        if not recent_values:
            return {}
        
        return {
            "count": len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "std_dev": statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0,
            "p95": self._percentile(recent_values, 95),
            "p99": self._percentile(recent_values, 99)
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


@dataclass
class BusinessMetric:
    """Business-specific metric tracking."""
    
    name: str
    description: str
    value: Union[int, float]
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


class PerformanceMonitor:
    """Monitor performance of operations and system resources."""
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize performance monitor.
        
        Args:
            retention_hours: How long to retain performance data
        """
        self.retention_hours = retention_hours
        self._performance_metrics: Dict[str, PerformanceMetric] = {}
        self._operation_timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            "embedding_generation_ms": 5000,  # 5 seconds
            "search_operation_ms": 1000,      # 1 second
            "document_processing_ms": 10000,  # 10 seconds
            "vector_store_operation_ms": 2000, # 2 seconds
            "memory_usage_mb": 1000,          # 1 GB
            "cpu_usage_percent": 80.0         # 80%
        }
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def record_operation_time(
        self,
        operation: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record the execution time of an operation.
        
        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            tags: Optional tags for the metric
        """
        with self._lock:
            # Add to performance metrics
            if operation not in self._performance_metrics:
                self._performance_metrics[operation] = PerformanceMetric(
                    name=operation,
                    tags=tags or {}
                )
            
            self._performance_metrics[operation].add_value(duration_ms)
            
            # Add to operation timers for quick access
            self._operation_timers[operation].append(duration_ms)
            
            # Keep only recent timers
            if len(self._operation_timers[operation]) > 1000:
                self._operation_timers[operation] = self._operation_timers[operation][-1000:]
            
            # Check thresholds and trigger alerts
            self._check_performance_threshold(operation, duration_ms, tags or {})
    
    @contextmanager
    def time_operation(self, operation: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager to time an operation.
        
        Args:
            operation: Name of the operation
            tags: Optional tags for the metric
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_operation_time(operation, duration_ms, tags)
    
    def get_operation_statistics(
        self,
        operation: str,
        window_minutes: int = 60
    ) -> Dict[str, float]:
        """
        Get performance statistics for an operation.
        
        Args:
            operation: Name of the operation
            window_minutes: Time window for statistics
            
        Returns:
            Dictionary with performance statistics
        """
        with self._lock:
            if operation not in self._performance_metrics:
                return {}
            
            return self._performance_metrics[operation].get_statistics(window_minutes)
    
    def get_all_operation_statistics(
        self,
        window_minutes: int = 60
    ) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all operations."""
        with self._lock:
            return {
                operation: metric.get_statistics(window_minutes)
                for operation, metric in self._performance_metrics.items()
            }
    
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
            List of slow operations with statistics
        """
        slow_operations = []
        
        with self._lock:
            for operation, metric in self._performance_metrics.items():
                stats = metric.get_statistics(window_minutes)
                if stats and stats.get("mean", 0) > threshold_ms:
                    slow_operations.append({
                        "operation": operation,
                        "mean_duration_ms": stats["mean"],
                        "p95_duration_ms": stats["p95"],
                        "count": stats["count"],
                        "tags": metric.tags
                    })
        
        return sorted(slow_operations, key=lambda x: x["mean_duration_ms"], reverse=True)
    
    def register_alert_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Register a callback for performance alerts."""
        self._alert_callbacks.append(callback)
    
    def _check_performance_threshold(
        self,
        operation: str,
        duration_ms: float,
        tags: Dict[str, str]
    ) -> None:
        """Check if operation exceeds performance thresholds."""
        threshold_key = f"{operation}_ms"
        if threshold_key in self.thresholds:
            threshold = self.thresholds[threshold_key]
            if duration_ms > threshold:
                alert_data = {
                    "operation": operation,
                    "duration_ms": duration_ms,
                    "threshold_ms": threshold,
                    "tags": tags,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                for callback in self._alert_callbacks:
                    try:
                        callback("performance_threshold_exceeded", alert_data)
                    except Exception:
                        # Don't let alert callback failures affect the main operation
                        pass
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update performance thresholds."""
        with self._lock:
            self.thresholds.update(new_thresholds)
    
    def cleanup_old_data(self) -> None:
        """Clean up old performance data."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            for metric in self._performance_metrics.values():
                # Filter out old data
                recent_indices = [
                    i for i, timestamp in enumerate(metric.timestamps)
                    if timestamp > cutoff
                ]
                
                if recent_indices:
                    metric.values = [metric.values[i] for i in recent_indices]
                    metric.timestamps = [metric.timestamps[i] for i in recent_indices]
                else:
                    metric.values.clear()
                    metric.timestamps.clear()


class BusinessMetricsCollector:
    """Collect and manage business-specific metrics."""
    
    def __init__(self):
        """Initialize business metrics collector."""
        self._metrics: Dict[str, BusinessMetric] = {}
        self._metric_history: Dict[str, List[BusinessMetric]] = defaultdict(list)
        self._lock = threading.Lock()
    
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
        metric = BusinessMetric(
            name=name,
            description=description,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        with self._lock:
            self._metrics[name] = metric
            self._metric_history[name].append(metric)
            
            # Keep only recent history (last 1000 entries)
            if len(self._metric_history[name]) > 1000:
                self._metric_history[name] = self._metric_history[name][-1000:]
    
    def get_current_metrics(self) -> Dict[str, BusinessMetric]:
        """Get current business metrics."""
        with self._lock:
            return dict(self._metrics)
    
    def get_metric_history(
        self,
        name: str,
        hours: int = 24
    ) -> List[BusinessMetric]:
        """
        Get historical data for a metric.
        
        Args:
            name: Metric name
            hours: Hours of history to retrieve
            
        Returns:
            List of historical metric values
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            return [
                metric for metric in self._metric_history.get(name, [])
                if metric.timestamp > cutoff
            ]
    
    def get_metric_trend(
        self,
        name: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get trend analysis for a metric.
        
        Args:
            name: Metric name
            hours: Hours of history to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        history = self.get_metric_history(name, hours)
        
        if len(history) < 2:
            return {"trend": "insufficient_data", "data_points": len(history)}
        
        values = [metric.value for metric in history]
        
        # Calculate trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            trend = "increasing"
        elif second_avg < first_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "data_points": len(history),
            "first_half_avg": first_avg,
            "second_half_avg": second_avg,
            "change_percent": ((second_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0,
            "min_value": min(values),
            "max_value": max(values),
            "current_value": values[-1]
        }


class HealthCheckEndpoint:
    """Health check endpoint for monitoring system status."""
    
    def __init__(self):
        """Initialize health check endpoint."""
        self._health_checks: Dict[str, Callable[[], tuple]] = {}
        self._last_check_results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], tuple],
        description: str = ""
    ) -> None:
        """
        Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns (healthy: bool, message: str, details: dict)
            description: Description of the health check
        """
        with self._lock:
            self._health_checks[name] = check_func
            self._last_check_results[name] = {
                "description": description,
                "last_checked": None,
                "healthy": None,
                "message": "",
                "details": {}
            }
    
    def run_health_checks(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run all health checks and return results.
        
        Args:
            force_refresh: Force refresh of all health checks
            
        Returns:
            Dictionary with health check results
        """
        results = {
            "overall_health": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        with self._lock:
            for name, check_func in self._health_checks.items():
                try:
                    # Check if we need to refresh this check
                    last_result = self._last_check_results[name]
                    should_refresh = (
                        force_refresh or
                        last_result["last_checked"] is None or
                        (datetime.utcnow() - last_result["last_checked"]).seconds > 30
                    )
                    
                    if should_refresh:
                        healthy, message, details = check_func()
                        
                        self._last_check_results[name].update({
                            "last_checked": datetime.utcnow(),
                            "healthy": healthy,
                            "message": message,
                            "details": details
                        })
                    
                    # Use cached result
                    check_result = self._last_check_results[name].copy()
                    check_result["last_checked"] = check_result["last_checked"].isoformat() if check_result["last_checked"] else None
                    
                    results["checks"][name] = check_result
                    
                    # Update overall health
                    if not check_result["healthy"]:
                        if results["overall_health"] == "healthy":
                            results["overall_health"] = "degraded"
                        elif results["overall_health"] == "degraded":
                            results["overall_health"] = "unhealthy"
                
                except Exception as e:
                    results["checks"][name] = {
                        "healthy": False,
                        "message": f"Health check failed: {str(e)}",
                        "details": {"error": str(e)},
                        "last_checked": datetime.utcnow().isoformat()
                    }
                    results["overall_health"] = "unhealthy"
        
        return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of health check status."""
        results = self.run_health_checks()
        
        healthy_count = sum(1 for check in results["checks"].values() if check["healthy"])
        total_count = len(results["checks"])
        
        return {
            "overall_health": results["overall_health"],
            "healthy_checks": healthy_count,
            "total_checks": total_count,
            "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0,
            "timestamp": results["timestamp"]
        }


class EnhancedMetricsCollector:
    """Enhanced metrics collector with performance monitoring and business metrics."""
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize enhanced metrics collector.
        
        Args:
            retention_hours: How long to retain metrics data
        """
        self.retention_hours = retention_hours
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(retention_hours)
        self.business_metrics = BusinessMetricsCollector()
        self.health_endpoint = HealthCheckEndpoint()
        
        # Basic metrics storage (from original implementation)
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
        
        # Register performance alert callback
        self.performance_monitor.register_alert_callback(self._handle_performance_alert)
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        metric_type: str = "gauge",
        unit: Optional[str] = None
    ) -> None:
        """Record a basic metric (maintains compatibility with existing code)."""
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
    
    def record_performance_metric(
        self,
        operation: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a performance metric."""
        self.performance_monitor.record_operation_time(operation, duration_ms, tags)
    
    def record_business_metric(
        self,
        name: str,
        value: Union[int, float],
        description: str = "",
        unit: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a business metric."""
        self.business_metrics.record_business_metric(name, value, description, unit, tags)
    
    @contextmanager
    def time_operation(self, operation: str, tags: Optional[Dict[str, str]] = None):
        """Context manager to time an operation."""
        with self.performance_monitor.time_operation(operation, tags):
            yield
    
    def get_performance_statistics(
        self,
        operation: Optional[str] = None,
        window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get performance statistics."""
        if operation:
            return self.performance_monitor.get_operation_statistics(operation, window_minutes)
        else:
            return self.performance_monitor.get_all_operation_statistics(window_minutes)
    
    def get_slow_operations(
        self,
        threshold_ms: float = 1000,
        window_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Get operations that are performing slowly."""
        return self.performance_monitor.get_slow_operations(threshold_ms, window_minutes)
    
    def get_business_metrics(self) -> Dict[str, BusinessMetric]:
        """Get current business metrics."""
        return self.business_metrics.get_current_metrics()
    
    def get_business_metric_trend(
        self,
        name: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get trend analysis for a business metric."""
        return self.business_metrics.get_metric_trend(name, hours)
    
    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], tuple],
        description: str = ""
    ) -> None:
        """Register a health check function."""
        self.health_endpoint.register_health_check(name, check_func, description)
    
    def get_health_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return self.health_endpoint.run_health_checks(force_refresh)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health status summary."""
        return self.health_endpoint.get_health_summary()
    
    def _handle_performance_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """Handle performance alerts by recording them as metrics."""
        self.record_metric(
            f"alert_{alert_type}",
            1,
            tags={
                "operation": alert_data.get("operation", "unknown"),
                "severity": "warning"
            },
            metric_type="counter"
        )
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics beyond retention period."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        for name, metric_list in self._metrics.items():
            while metric_list and metric_list[0].timestamp < cutoff:
                metric_list.popleft()
    
    # Maintain compatibility with existing methods
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, 1, tags, "counter")
    
    def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric."""
        self.record_metric(name, duration_ms, tags, "timer", "ms")
        # Also record as performance metric
        self.record_performance_metric(name, duration_ms, tags)
    
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
        
        # Record as performance metric
        self.record_performance_metric("request", response_time_ms, {"error": str(error)})
    
    def record_document_indexed(self) -> None:
        """Record a document indexing operation."""
        with self._lock:
            self._documents_indexed += 1
        self.increment_counter("documents_indexed")
        self.record_business_metric("total_documents_indexed", self._documents_indexed, "Total documents indexed", "count")
    
    def record_search_performed(self) -> None:
        """Record a search operation."""
        with self._lock:
            self._searches_performed += 1
        self.increment_counter("searches_performed")
        self.record_business_metric("total_searches_performed", self._searches_performed, "Total searches performed", "count")
    
    def record_embedding_generated(self) -> None:
        """Record an embedding generation."""
        with self._lock:
            self._embeddings_generated += 1
        self.increment_counter("embeddings_generated")
        self.record_business_metric("total_embeddings_generated", self._embeddings_generated, "Total embeddings generated", "count")
    
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
        """Get metrics for a specific name or all metrics."""
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
        import psutil
        
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