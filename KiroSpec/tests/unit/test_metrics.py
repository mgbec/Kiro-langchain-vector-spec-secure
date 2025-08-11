"""
Unit tests for enhanced metrics collection and monitoring.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from langchain_vector_db.services.metrics import (
    PerformanceMetric,
    BusinessMetric,
    PerformanceMonitor,
    BusinessMetricsCollector,
    HealthCheckEndpoint,
    EnhancedMetricsCollector
)


class TestPerformanceMetric:
    """Test cases for PerformanceMetric."""
    
    def test_initialization(self):
        """Test performance metric initialization."""
        metric = PerformanceMetric(
            name="test_operation",
            tags={"component": "test"}
        )
        
        assert metric.name == "test_operation"
        assert metric.tags["component"] == "test"
        assert len(metric.values) == 0
        assert len(metric.timestamps) == 0
    
    def test_add_value(self):
        """Test adding values to performance metric."""
        metric = PerformanceMetric(name="test_operation")
        
        metric.add_value(100.0)
        metric.add_value(200.0)
        
        assert len(metric.values) == 2
        assert metric.values == [100.0, 200.0]
        assert len(metric.timestamps) == 2
    
    def test_get_statistics(self):
        """Test getting statistics from performance metric."""
        metric = PerformanceMetric(name="test_operation")
        
        # Add test values
        values = [100.0, 150.0, 200.0, 250.0, 300.0]
        for value in values:
            metric.add_value(value)
        
        stats = metric.get_statistics()
        
        assert stats["count"] == 5
        assert stats["min"] == 100.0
        assert stats["max"] == 300.0
        assert stats["mean"] == 200.0
        assert stats["median"] == 200.0
        assert "std_dev" in stats
        assert "p95" in stats
        assert "p99" in stats
    
    def test_get_statistics_empty(self):
        """Test getting statistics from empty metric."""
        metric = PerformanceMetric(name="test_operation")
        
        stats = metric.get_statistics()
        assert stats == {}
    
    def test_get_statistics_with_time_window(self):
        """Test getting statistics with time window filtering."""
        metric = PerformanceMetric(name="test_operation")
        
        # Add old value
        old_timestamp = datetime.utcnow() - timedelta(hours=2)
        metric.add_value(100.0, old_timestamp)
        
        # Add recent value
        metric.add_value(200.0)
        
        # Get statistics for last hour (should only include recent value)
        stats = metric.get_statistics(window_minutes=60)
        
        assert stats["count"] == 1
        assert stats["mean"] == 200.0
    
    def test_percentile_calculation(self):
        """Test percentile calculation."""
        metric = PerformanceMetric(name="test_operation")
        
        # Add values 1-100
        for i in range(1, 101):
            metric.add_value(float(i))
        
        stats = metric.get_statistics()
        
        # P95 should be around 95
        assert 94 <= stats["p95"] <= 96
        # P99 should be around 99
        assert 98 <= stats["p99"] <= 100


class TestBusinessMetric:
    """Test cases for BusinessMetric."""
    
    def test_initialization(self):
        """Test business metric initialization."""
        metric = BusinessMetric(
            name="total_users",
            description="Total number of users",
            value=1000,
            unit="count",
            tags={"category": "users"}
        )
        
        assert metric.name == "total_users"
        assert metric.description == "Total number of users"
        assert metric.value == 1000
        assert metric.unit == "count"
        assert metric.tags["category"] == "users"
    
    def test_to_dict(self):
        """Test business metric dictionary conversion."""
        metric = BusinessMetric(
            name="revenue",
            description="Monthly revenue",
            value=50000.0,
            unit="USD"
        )
        
        result = metric.to_dict()
        
        assert result["name"] == "revenue"
        assert result["description"] == "Monthly revenue"
        assert result["value"] == 50000.0
        assert result["unit"] == "USD"
        assert "timestamp" in result


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(retention_hours=1)
    
    def test_initialization(self):
        """Test performance monitor initialization."""
        assert self.monitor.retention_hours == 1
        assert len(self.monitor._performance_metrics) == 0
        assert len(self.monitor._operation_timers) == 0
    
    def test_record_operation_time(self):
        """Test recording operation times."""
        self.monitor.record_operation_time("test_op", 150.0, {"component": "test"})
        
        assert "test_op" in self.monitor._performance_metrics
        assert "test_op" in self.monitor._operation_timers
        
        metric = self.monitor._performance_metrics["test_op"]
        assert len(metric.values) == 1
        assert metric.values[0] == 150.0
        assert metric.tags["component"] == "test"
    
    def test_time_operation_context_manager(self):
        """Test timing operations with context manager."""
        with self.monitor.time_operation("test_op", {"component": "test"}):
            time.sleep(0.01)  # Small delay
        
        assert "test_op" in self.monitor._performance_metrics
        metric = self.monitor._performance_metrics["test_op"]
        assert len(metric.values) == 1
        assert metric.values[0] > 0  # Should have recorded some time
    
    def test_get_operation_statistics(self):
        """Test getting operation statistics."""
        # Record multiple operations
        for i in range(5):
            self.monitor.record_operation_time("test_op", 100.0 + i * 10)
        
        stats = self.monitor.get_operation_statistics("test_op")
        
        assert stats["count"] == 5
        assert stats["min"] == 100.0
        assert stats["max"] == 140.0
        assert stats["mean"] == 120.0
    
    def test_get_all_operation_statistics(self):
        """Test getting statistics for all operations."""
        self.monitor.record_operation_time("op1", 100.0)
        self.monitor.record_operation_time("op2", 200.0)
        
        all_stats = self.monitor.get_all_operation_statistics()
        
        assert "op1" in all_stats
        assert "op2" in all_stats
        assert all_stats["op1"]["mean"] == 100.0
        assert all_stats["op2"]["mean"] == 200.0
    
    def test_get_slow_operations(self):
        """Test getting slow operations."""
        # Record fast operation
        self.monitor.record_operation_time("fast_op", 50.0)
        
        # Record slow operation
        self.monitor.record_operation_time("slow_op", 1500.0)
        
        slow_ops = self.monitor.get_slow_operations(threshold_ms=1000.0)
        
        assert len(slow_ops) == 1
        assert slow_ops[0]["operation"] == "slow_op"
        assert slow_ops[0]["mean_duration_ms"] == 1500.0
    
    def test_performance_threshold_alerts(self):
        """Test performance threshold alerts."""
        alert_data = []
        
        def alert_callback(alert_type, data):
            alert_data.append((alert_type, data))
        
        self.monitor.register_alert_callback(alert_callback)
        
        # Set a low threshold for testing
        self.monitor.update_thresholds({"test_op_ms": 100.0})
        
        # Record operation that exceeds threshold
        self.monitor.record_operation_time("test_op", 150.0)
        
        assert len(alert_data) == 1
        assert alert_data[0][0] == "performance_threshold_exceeded"
        assert alert_data[0][1]["operation"] == "test_op"
        assert alert_data[0][1]["duration_ms"] == 150.0
    
    def test_update_thresholds(self):
        """Test updating performance thresholds."""
        original_threshold = self.monitor.thresholds.get("embedding_generation_ms", 0)
        
        self.monitor.update_thresholds({"embedding_generation_ms": 2000.0})
        
        assert self.monitor.thresholds["embedding_generation_ms"] == 2000.0
        assert self.monitor.thresholds["embedding_generation_ms"] != original_threshold


class TestBusinessMetricsCollector:
    """Test cases for BusinessMetricsCollector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = BusinessMetricsCollector()
    
    def test_record_business_metric(self):
        """Test recording business metrics."""
        self.collector.record_business_metric(
            "total_users",
            1000,
            "Total number of users",
            "count",
            {"category": "users"}
        )
        
        metrics = self.collector.get_current_metrics()
        
        assert "total_users" in metrics
        metric = metrics["total_users"]
        assert metric.value == 1000
        assert metric.description == "Total number of users"
        assert metric.unit == "count"
        assert metric.tags["category"] == "users"
    
    def test_get_metric_history(self):
        """Test getting metric history."""
        # Record multiple values
        for i in range(5):
            self.collector.record_business_metric(
                "user_count",
                1000 + i * 100,
                "User count"
            )
        
        history = self.collector.get_metric_history("user_count")
        
        assert len(history) == 5
        assert history[0].value == 1000
        assert history[-1].value == 1400
    
    def test_get_metric_trend(self):
        """Test getting metric trend analysis."""
        # Record increasing values
        for i in range(10):
            self.collector.record_business_metric("revenue", 1000 + i * 100)
        
        trend = self.collector.get_metric_trend("revenue")
        
        assert trend["trend"] == "increasing"
        assert trend["data_points"] == 10
        assert trend["change_percent"] > 0
    
    def test_get_metric_trend_decreasing(self):
        """Test getting decreasing metric trend."""
        # Record decreasing values
        for i in range(10):
            self.collector.record_business_metric("errors", 1000 - i * 100)
        
        trend = self.collector.get_metric_trend("errors")
        
        assert trend["trend"] == "decreasing"
        assert trend["change_percent"] < 0
    
    def test_get_metric_trend_stable(self):
        """Test getting stable metric trend."""
        # Record stable values
        for i in range(10):
            self.collector.record_business_metric("stable_metric", 1000 + (i % 2))
        
        trend = self.collector.get_metric_trend("stable_metric")
        
        assert trend["trend"] == "stable"
    
    def test_get_metric_trend_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        self.collector.record_business_metric("single_metric", 100)
        
        trend = self.collector.get_metric_trend("single_metric")
        
        assert trend["trend"] == "insufficient_data"
        assert trend["data_points"] == 1


class TestHealthCheckEndpoint:
    """Test cases for HealthCheckEndpoint."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.endpoint = HealthCheckEndpoint()
    
    def test_register_health_check(self):
        """Test registering health checks."""
        def test_check():
            return True, "Test is healthy", {"status": "ok"}
        
        self.endpoint.register_health_check("test_component", test_check, "Test component")
        
        assert "test_component" in self.endpoint._health_checks
        assert "test_component" in self.endpoint._last_check_results
    
    def test_run_health_checks_all_healthy(self):
        """Test running health checks when all are healthy."""
        def healthy_check():
            return True, "Component is healthy", {"status": "ok"}
        
        self.endpoint.register_health_check("component1", healthy_check)
        self.endpoint.register_health_check("component2", healthy_check)
        
        results = self.endpoint.run_health_checks()
        
        assert results["overall_health"] == "healthy"
        assert len(results["checks"]) == 2
        assert results["checks"]["component1"]["healthy"] is True
        assert results["checks"]["component2"]["healthy"] is True
    
    def test_run_health_checks_with_failure(self):
        """Test running health checks with component failure."""
        def healthy_check():
            return True, "Component is healthy", {}
        
        def unhealthy_check():
            return False, "Component is unhealthy", {"error": "connection_failed"}
        
        self.endpoint.register_health_check("healthy_component", healthy_check)
        self.endpoint.register_health_check("unhealthy_component", unhealthy_check)
        
        results = self.endpoint.run_health_checks()
        
        assert results["overall_health"] == "degraded"
        assert results["checks"]["healthy_component"]["healthy"] is True
        assert results["checks"]["unhealthy_component"]["healthy"] is False
    
    def test_run_health_checks_with_exception(self):
        """Test running health checks with exception."""
        def exception_check():
            raise Exception("Health check failed")
        
        self.endpoint.register_health_check("exception_component", exception_check)
        
        results = self.endpoint.run_health_checks()
        
        assert results["overall_health"] == "unhealthy"
        assert results["checks"]["exception_component"]["healthy"] is False
        assert "Health check failed" in results["checks"]["exception_component"]["message"]
    
    def test_get_health_summary(self):
        """Test getting health summary."""
        def healthy_check():
            return True, "Healthy", {}
        
        def unhealthy_check():
            return False, "Unhealthy", {}
        
        self.endpoint.register_health_check("healthy", healthy_check)
        self.endpoint.register_health_check("unhealthy", unhealthy_check)
        
        summary = self.endpoint.get_health_summary()
        
        assert summary["overall_health"] == "degraded"
        assert summary["healthy_checks"] == 1
        assert summary["total_checks"] == 2
        assert summary["health_percentage"] == 50.0
    
    def test_health_check_caching(self):
        """Test health check result caching."""
        call_count = 0
        
        def counting_check():
            nonlocal call_count
            call_count += 1
            return True, f"Called {call_count} times", {}
        
        self.endpoint.register_health_check("counting", counting_check)
        
        # First call should execute the check
        results1 = self.endpoint.run_health_checks()
        assert call_count == 1
        
        # Second call within 30 seconds should use cached result
        results2 = self.endpoint.run_health_checks()
        assert call_count == 1
        
        # Force refresh should execute the check again
        results3 = self.endpoint.run_health_checks(force_refresh=True)
        assert call_count == 2


class TestEnhancedMetricsCollector:
    """Test cases for EnhancedMetricsCollector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = EnhancedMetricsCollector(retention_hours=1)
    
    def test_initialization(self):
        """Test enhanced metrics collector initialization."""
        assert self.collector.retention_hours == 1
        assert self.collector.performance_monitor is not None
        assert self.collector.business_metrics is not None
        assert self.collector.health_endpoint is not None
    
    def test_record_performance_metric(self):
        """Test recording performance metrics."""
        self.collector.record_performance_metric("test_op", 150.0, {"component": "test"})
        
        stats = self.collector.get_performance_statistics("test_op")
        assert stats["count"] == 1
        assert stats["mean"] == 150.0
    
    def test_record_business_metric(self):
        """Test recording business metrics."""
        self.collector.record_business_metric(
            "total_users",
            1000,
            "Total users",
            "count",
            {"category": "users"}
        )
        
        metrics = self.collector.get_business_metrics()
        assert "total_users" in metrics
        assert metrics["total_users"]["value"] == 1000
    
    def test_time_operation_context_manager(self):
        """Test timing operations with context manager."""
        with self.collector.time_operation("test_op", {"component": "test"}):
            time.sleep(0.01)
        
        stats = self.collector.get_performance_statistics("test_op")
        assert stats["count"] == 1
        assert stats["mean"] > 0
    
    def test_get_slow_operations(self):
        """Test getting slow operations."""
        self.collector.record_performance_metric("fast_op", 50.0)
        self.collector.record_performance_metric("slow_op", 1500.0)
        
        slow_ops = self.collector.get_slow_operations(threshold_ms=1000.0)
        
        assert len(slow_ops) == 1
        assert slow_ops[0]["operation"] == "slow_op"
    
    def test_register_health_check(self):
        """Test registering health checks."""
        def test_check():
            return True, "Test is healthy", {}
        
        self.collector.register_health_check("test", test_check, "Test component")
        
        health_status = self.collector.get_health_status()
        assert "test" in health_status["checks"]
        assert health_status["checks"]["test"]["healthy"] is True
    
    def test_get_health_summary(self):
        """Test getting health summary."""
        def healthy_check():
            return True, "Healthy", {}
        
        self.collector.register_health_check("test", healthy_check)
        
        summary = self.collector.get_health_summary()
        assert summary["overall_health"] == "healthy"
        assert summary["healthy_checks"] == 1
        assert summary["total_checks"] == 1
    
    def test_business_metric_trend(self):
        """Test business metric trend analysis."""
        # Record increasing values
        for i in range(10):
            self.collector.record_business_metric("revenue", 1000 + i * 100)
        
        trend = self.collector.get_business_metric_trend("revenue")
        assert trend["trend"] == "increasing"
        assert trend["change_percent"] > 0
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_get_system_metrics(self, mock_disk, mock_cpu, mock_memory):
        """Test getting system metrics."""
        # Mock system information
        mock_memory.return_value = MagicMock(used=1024 * 1024 * 512)  # 512 MB
        mock_cpu.return_value = 75.5
        mock_disk.return_value = MagicMock(used=1024 * 1024 * 1024 * 10)  # 10 GB
        
        # Record some application metrics
        self.collector.record_document_indexed()
        self.collector.record_search_performed()
        self.collector.record_embedding_generated()
        
        system_metrics = self.collector.get_system_metrics()
        
        assert system_metrics.memory_usage_mb == 512.0
        assert system_metrics.cpu_usage_percent == 75.5
        assert system_metrics.disk_usage_mb == 10240.0
        assert system_metrics.documents_indexed == 1
        assert system_metrics.searches_performed == 1
        assert system_metrics.embeddings_generated == 1
    
    def test_compatibility_methods(self):
        """Test compatibility with existing metrics methods."""
        # Test counter
        self.collector.increment_counter("test_counter", {"component": "test"})
        
        # Test timer
        self.collector.record_timer("test_timer", 150.0, {"component": "test"})
        
        # Test gauge
        self.collector.record_gauge("test_gauge", 42.0, {"component": "test"})
        
        # Test request tracking
        self.collector.record_request(100.0, error=False)
        self.collector.record_request(200.0, error=True)
        
        # Verify metrics were recorded
        metrics = self.collector.get_metrics()
        assert len(metrics) > 0
        
        # Verify performance metrics were also recorded
        stats = self.collector.get_performance_statistics("test_timer")
        assert stats["count"] == 1
        assert stats["mean"] == 150.0
    
    def test_application_metrics_recording(self):
        """Test application-specific metrics recording."""
        # Record various application metrics
        self.collector.record_document_indexed()
        self.collector.record_search_performed()
        self.collector.record_embedding_generated()
        self.collector.record_cache_hit()
        self.collector.record_cache_miss()
        
        # Check business metrics were created
        business_metrics = self.collector.get_business_metrics()
        
        assert "total_documents_indexed" in business_metrics
        assert "total_searches_performed" in business_metrics
        assert "total_embeddings_generated" in business_metrics
        
        # Check system metrics
        system_metrics = self.collector.get_system_metrics()
        assert system_metrics.documents_indexed == 1
        assert system_metrics.searches_performed == 1
        assert system_metrics.embeddings_generated == 1
        assert system_metrics.cache_hit_rate == 0.5  # 1 hit, 1 miss