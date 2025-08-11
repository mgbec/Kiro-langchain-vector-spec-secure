# Observability and Monitoring Guide

This guide covers the comprehensive observability features of the LangChain Vector Database, including logging, metrics, tracing, and monitoring.

## Table of Contents

1. [Observability Overview](#observability-overview)
2. [Logging Configuration](#logging-configuration)
3. [Metrics Collection](#metrics-collection)
4. [Distributed Tracing](#distributed-tracing)
5. [Performance Monitoring](#performance-monitoring)
6. [Health Checks](#health-checks)
7. [Alerting and Notifications](#alerting-and-notifications)
8. [Dashboard Setup](#dashboard-setup)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Observability Overview

The LangChain Vector Database provides comprehensive observability features:

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Metrics Collection**: System and business metrics
- **Distributed Tracing**: Request flow visibility with OpenTelemetry
- **Performance Monitoring**: Real-time performance analysis
- **Health Checks**: Component and system health monitoring
- **Resource Monitoring**: CPU, memory, and disk usage tracking

## Logging Configuration

### Basic Logging Setup

```python
from langchain_vector_db.models.config import VectorDBConfig, ObservabilityConfig

observability_config = ObservabilityConfig(
    log_level="INFO",
    log_format="json",
    log_output="file",
    log_file_path="./logs/vector_db.log",
    log_rotation_size_mb=100,
    log_retention_days=30
)

config = VectorDBConfig(
    storage_type="local",
    observability=observability_config
)
```

### Advanced Logging Configuration

```python
observability_config = ObservabilityConfig(
    log_level="DEBUG",
    log_format="json",
    log_output="both",  # file and console
    log_file_path="./logs/vector_db.log",
    log_rotation_size_mb=50,
    log_retention_days=90,
    
    # Structured logging options
    include_correlation_id=True,
    include_request_id=True,
    include_user_context=True,
    include_performance_data=True,
    
    # Log filtering
    log_filters=["embedding_service", "vector_store"],
    exclude_patterns=["health_check", "metrics_collection"]
)
```

### Using Structured Logging

```python
manager = VectorDatabaseManager(config)
observability_manager = manager.observability_manager

# Set logging context
from langchain_vector_db.services.observability import create_log_context

log_context = create_log_context(
    user_id="john_doe",
    operation="document_ingestion",
    component="VectorDatabaseManager",
    correlation_id="req_12345"
)

observability_manager.set_log_context(log_context)

# Log events with context
observability_manager.log_event("INFO", "Starting document processing")
observability_manager.log_event("DEBUG", "Generated embeddings", extra_data={"count": 5})
observability_manager.log_event("ERROR", "Processing failed", exception=exception)
```

### Retrieving Logs

```python
# Get recent logs
recent_logs = observability_manager.get_recent_logs(limit=100)

for log_entry in recent_logs:
    log_dict = log_entry.to_dict()
    print(f"[{log_dict['timestamp']}] {log_dict['level']}: {log_dict['message']}")
    
    if log_dict.get('correlation_id'):
        print(f"  Correlation ID: {log_dict['correlation_id']}")
    
    if log_dict.get('user_id'):
        print(f"  User: {log_dict['user_id']}")

# Filter logs by criteria
filtered_logs = observability_manager.get_logs_by_filter(
    level="ERROR",
    component="VectorStore",
    start_time=datetime.utcnow() - timedelta(hours=1)
)
```

## Metrics Collection

### System Metrics

```python
observability_config = ObservabilityConfig(
    metrics_enabled=True,
    metrics_collection_interval=30,  # seconds
    system_metrics_enabled=True,
    business_metrics_enabled=True,
    custom_metrics_enabled=True
)

# Get system metrics
system_metrics = observability_manager.get_system_metrics()

print(f"Documents indexed: {system_metrics.documents_indexed}")
print(f"Searches performed: {system_metrics.searches_performed}")
print(f"Embeddings generated: {system_metrics.embeddings_generated}")
print(f"Request count: {system_metrics.request_count}")
print(f"Error count: {system_metrics.error_count}")
print(f"Average response time: {system_metrics.avg_response_time_ms}ms")
```

### Business Metrics

```python
# Record custom business metrics
observability_manager.record_business_metric(
    name="documents_processed_per_hour",
    value=150,
    description="Number of documents processed per hour",
    unit="count/hour",
    tags={"department": "research", "priority": "high"}
)

observability_manager.record_business_metric(
    name="search_accuracy_score",
    value=94.5,
    description="Search result accuracy percentage",
    unit="percentage",
    tags={"model": "text-embedding-ada-002"}
)

# Get business metrics
business_metrics = observability_manager.get_business_metrics()
for metric_name, metric_data in business_metrics.items():
    print(f"{metric_name}: {metric_data['value']} {metric_data['unit']}")
```

### Performance Metrics

```python
# Get performance statistics
perf_stats = observability_manager.get_performance_statistics()

for operation, stats in perf_stats.items():
    if isinstance(stats, dict) and 'count' in stats:
        print(f"{operation}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95: {stats.get('p95', 'N/A'):.2f}ms")
        print(f"  P99: {stats.get('p99', 'N/A'):.2f}ms")

# Get slow operations
slow_operations = observability_manager.get_slow_operations(threshold_ms=1000)
for slow_op in slow_operations:
    print(f"Slow operation: {slow_op['operation']}")
    print(f"  Mean duration: {slow_op['mean_duration_ms']:.2f}ms")
    print(f"  Call count: {slow_op['call_count']}")
```

### Resource Metrics

```python
# Monitor system resources
resource_usage = observability_manager.get_resource_usage()

print(f"Memory usage: {resource_usage['memory_usage_mb']} MB")
print(f"CPU usage: {resource_usage['cpu_usage_percent']}%")
print(f"Disk usage: {resource_usage['disk_usage_mb']} MB")
print(f"Network I/O: {resource_usage['network_io_mb']} MB")

# Check for resource alerts
resource_alerts = observability_manager.get_resource_alerts()
for alert in resource_alerts:
    print(f"Resource Alert: {alert['type']}")
    print(f"  Message: {alert['message']}")
    print(f"  Threshold: {alert['threshold']}")
    print(f"  Current: {alert['current_value']}")
```

## Distributed Tracing

### OpenTelemetry Integration

```python
observability_config = ObservabilityConfig(
    tracing_enabled=True,
    tracing_provider="opentelemetry",
    tracing_endpoint="http://jaeger:14268/api/traces",
    tracing_service_name="vector-database",
    tracing_sample_rate=1.0,  # 100% sampling for development
    
    # Trace configuration
    trace_requests=True,
    trace_database_operations=True,
    trace_embedding_generation=True,
    trace_external_calls=True
)
```

### Manual Tracing

```python
# Create custom traces
with observability_manager.trace_distributed_operation(
    "custom_document_processing",
    attributes={
        "document_count": len(documents),
        "embedding_model": "text-embedding-ada-002",
        "operation.type": "batch_processing"
    }
) as span:
    
    # Process documents
    doc_ids = manager.add_documents(documents)
    
    # Add span events
    observability_manager.add_span_event(
        "documents_processed",
        {"processed_count": len(doc_ids)},
        span
    )
    
    # Set span attributes
    observability_manager.set_span_attribute("success", True, span)
```

### Trace Analysis

```python
# Get recent traces
traces = observability_manager.get_traces(limit=50)

for trace in traces:
    trace_dict = trace.to_dict()
    print(f"Trace: {trace_dict['operation_name']}")
    print(f"  Duration: {trace_dict['duration_ms']:.2f}ms")
    print(f"  Status: {trace_dict['status']}")
    print(f"  Trace ID: {trace_dict['trace_id']}")

# Get trace statistics
trace_stats = observability_manager.get_trace_statistics()
print(f"Total traces: {trace_stats['total_traces']}")
print(f"Active spans: {trace_stats['active_spans']}")
print(f"Average trace duration: {trace_stats['avg_duration_ms']:.2f}ms")
```

### Trace Correlation

```python
# Correlate traces with logs
correlation_id = "req_12345"

# Get traces for correlation ID
related_traces = observability_manager.get_traces_by_correlation_id(correlation_id)

# Get logs for correlation ID
related_logs = observability_manager.get_logs_by_correlation_id(correlation_id)

print(f"Found {len(related_traces)} traces and {len(related_logs)} logs for correlation ID: {correlation_id}")
```

## Performance Monitoring

### Real-time Performance Monitoring

```python
observability_config = ObservabilityConfig(
    performance_monitoring_enabled=True,
    performance_collection_interval=10,  # seconds
    memory_threshold_mb=1024,
    cpu_threshold_percent=80.0,
    response_time_threshold_ms=2000.0,
    
    # Performance alerts
    enable_performance_alerts=True,
    alert_on_memory_threshold=True,
    alert_on_cpu_threshold=True,
    alert_on_slow_operations=True
)
```

### Performance Analysis

```python
# Analyze performance trends
performance_trends = observability_manager.get_performance_trends(
    time_window_hours=24,
    metrics=["response_time", "throughput", "error_rate"]
)

for metric, trend_data in performance_trends.items():
    print(f"{metric} trend:")
    print(f"  Current: {trend_data['current']}")
    print(f"  Average: {trend_data['average']}")
    print(f"  Trend: {trend_data['trend']}")  # increasing/decreasing/stable

# Get performance bottlenecks
bottlenecks = observability_manager.identify_performance_bottlenecks()
for bottleneck in bottlenecks:
    print(f"Bottleneck: {bottleneck['component']}")
    print(f"  Issue: {bottleneck['issue']}")
    print(f"  Impact: {bottleneck['impact']}")
    print(f"  Recommendation: {bottleneck['recommendation']}")
```

### Load Testing Metrics

```python
# Monitor during load testing
load_test_metrics = observability_manager.start_load_test_monitoring(
    test_name="similarity_search_load_test",
    expected_rps=100,
    duration_minutes=10
)

# ... perform load test ...

# Get load test results
results = observability_manager.get_load_test_results("similarity_search_load_test")
print(f"Peak RPS: {results['peak_rps']}")
print(f"Average response time: {results['avg_response_time_ms']}ms")
print(f"Error rate: {results['error_rate']}%")
print(f"P95 response time: {results['p95_response_time_ms']}ms")
```

## Health Checks

### Comprehensive Health Monitoring

```python
observability_config = ObservabilityConfig(
    health_checks_enabled=True,
    health_check_interval=30,  # seconds
    health_check_timeout=10,   # seconds
    
    # Component health checks
    check_vector_store=True,
    check_embedding_service=True,
    check_security_manager=True,
    check_external_dependencies=True
)
```

### Health Check Implementation

```python
# Get comprehensive health status
health_status = observability_manager.get_comprehensive_health_status()

print(f"Overall health: {health_status['overall_health']}")
print(f"Health score: {health_status['health_score']}/100")

for component, status in health_status['checks'].items():
    health_icon = "✅" if status['healthy'] else "❌"
    print(f"{health_icon} {component}: {status['message']}")
    
    if not status['healthy'] and 'details' in status:
        print(f"  Details: {status['details']}")

# Get health summary
health_summary = observability_manager.get_health_summary()
print(f"Healthy checks: {health_summary['healthy_checks']}/{health_summary['total_checks']}")
print(f"Health percentage: {health_summary['health_percentage']:.1f}%")
```

### Custom Health Checks

```python
# Add custom health check
def custom_database_health_check():
    try:
        # Perform custom health check logic
        doc_count = manager.get_document_count()
        if doc_count >= 0:
            return {"healthy": True, "message": f"Database accessible with {doc_count} documents"}
        else:
            return {"healthy": False, "message": "Database returned invalid count"}
    except Exception as e:
        return {"healthy": False, "message": f"Database check failed: {str(e)}"}

# Register custom health check
observability_manager.register_health_check(
    name="custom_database_check",
    check_function=custom_database_health_check,
    interval_seconds=60
)
```

### Health Alerting

```python
# Configure health-based alerts
health_alert_config = {
    "overall_health_threshold": 80,  # Alert if health score below 80%
    "component_failure_alert": True,
    "consecutive_failures_threshold": 3,
    "alert_channels": ["email", "slack", "webhook"]
}

observability_manager.configure_health_alerts(health_alert_config)

# Get health alerts
health_alerts = observability_manager.get_health_alerts()
for alert in health_alerts:
    print(f"Health Alert: {alert['component']}")
    print(f"  Status: {alert['status']}")
    print(f"  Message: {alert['message']}")
    print(f"  Duration: {alert['duration_minutes']} minutes")
```

## Alerting and Notifications

### Alert Configuration

```python
observability_config = ObservabilityConfig(
    alerting_enabled=True,
    alert_channels=["email", "slack", "webhook"],
    
    # Alert thresholds
    error_rate_threshold=5.0,  # 5% error rate
    response_time_threshold=2000.0,  # 2 seconds
    memory_usage_threshold=80.0,  # 80% memory usage
    cpu_usage_threshold=85.0,  # 85% CPU usage
    
    # Alert configuration
    alert_cooldown_minutes=15,
    alert_escalation_minutes=60,
    alert_severity_levels=["low", "medium", "high", "critical"]
)
```

### Setting Up Alerts

```python
# Configure email alerts
observability_manager.configure_email_alerts(
    smtp_server="smtp.company.com",
    smtp_port=587,
    username="alerts@company.com",
    password="password",
    recipients=["admin@company.com", "ops@company.com"]
)

# Configure Slack alerts
observability_manager.configure_slack_alerts(
    webhook_url="https://hooks.slack.com/services/...",
    channel="#vector-db-alerts",
    username="VectorDB Monitor"
)

# Configure webhook alerts
observability_manager.configure_webhook_alerts(
    webhook_url="https://api.company.com/alerts",
    headers={"Authorization": "Bearer token"},
    payload_template={
        "service": "vector-database",
        "alert": "{{alert_type}}",
        "message": "{{message}}",
        "severity": "{{severity}}"
    }
)
```

### Custom Alerts

```python
# Create custom alert rules
custom_alert_rules = [
    {
        "name": "high_search_latency",
        "condition": "avg(search_response_time_ms) > 3000",
        "severity": "high",
        "message": "Search response time is above 3 seconds",
        "cooldown_minutes": 10
    },
    {
        "name": "embedding_service_errors",
        "condition": "rate(embedding_errors) > 0.1",
        "severity": "critical",
        "message": "Embedding service error rate is above 10%",
        "cooldown_minutes": 5
    }
]

observability_manager.add_alert_rules(custom_alert_rules)
```

## Dashboard Setup

### Grafana Integration

```python
# Export metrics for Grafana
grafana_config = {
    "prometheus_endpoint": "http://prometheus:9090",
    "grafana_url": "http://grafana:3000",
    "dashboard_refresh_interval": "30s",
    "retention_period": "30d"
}

# Generate Grafana dashboard configuration
dashboard_config = observability_manager.generate_grafana_dashboard(
    dashboard_name="Vector Database Monitoring",
    panels=[
        "system_metrics",
        "performance_metrics",
        "business_metrics",
        "health_status",
        "error_rates",
        "response_times"
    ]
)

# Save dashboard configuration
with open("grafana_dashboard.json", "w") as f:
    json.dump(dashboard_config, f, indent=2)
```

### Custom Dashboard Metrics

```python
# Define custom dashboard metrics
dashboard_metrics = {
    "documents_per_minute": {
        "query": "rate(documents_indexed_total[1m])",
        "title": "Documents Indexed per Minute",
        "type": "graph",
        "unit": "docs/min"
    },
    "search_success_rate": {
        "query": "rate(searches_successful_total[5m]) / rate(searches_total[5m]) * 100",
        "title": "Search Success Rate",
        "type": "stat",
        "unit": "percent"
    },
    "embedding_generation_time": {
        "query": "histogram_quantile(0.95, embedding_generation_duration_seconds)",
        "title": "Embedding Generation Time (P95)",
        "type": "graph",
        "unit": "seconds"
    }
}

observability_manager.register_dashboard_metrics(dashboard_metrics)
```

## Best Practices

### 1. Logging Best Practices

- Use structured logging (JSON format)
- Include correlation IDs for request tracing
- Set appropriate log levels for different environments
- Implement log rotation and retention policies
- Avoid logging sensitive information

### 2. Metrics Best Practices

- Collect both system and business metrics
- Use appropriate metric types (counters, gauges, histograms)
- Include relevant tags and labels
- Monitor key performance indicators (KPIs)
- Set up alerting on critical metrics

### 3. Tracing Best Practices

- Trace critical operations and workflows
- Include relevant context in spans
- Use sampling to manage overhead
- Correlate traces with logs and metrics
- Monitor trace performance impact

### 4. Performance Monitoring

- Monitor response times and throughput
- Track resource utilization
- Identify performance bottlenecks
- Set up performance alerts
- Regular performance testing

### 5. Health Checks

- Implement comprehensive health checks
- Check all critical components
- Set appropriate timeouts
- Monitor health check performance
- Alert on health degradation

## Troubleshooting

### Common Issues

#### High Memory Usage

```python
# Investigate memory usage
memory_stats = observability_manager.get_memory_statistics()
print(f"Total memory: {memory_stats['total_mb']} MB")
print(f"Used memory: {memory_stats['used_mb']} MB")
print(f"Available memory: {memory_stats['available_mb']} MB")

# Check for memory leaks
memory_trends = observability_manager.get_memory_trends(hours=24)
if memory_trends['trend'] == 'increasing':
    print("Potential memory leak detected")
    
    # Get top memory consumers
    memory_consumers = observability_manager.get_top_memory_consumers()
    for consumer in memory_consumers:
        print(f"  {consumer['component']}: {consumer['memory_mb']} MB")
```

#### Slow Performance

```python
# Analyze slow operations
slow_ops = observability_manager.get_slow_operations(threshold_ms=1000)
for op in slow_ops:
    print(f"Slow operation: {op['operation']}")
    print(f"  Average duration: {op['mean_duration_ms']}ms")
    print(f"  Call count: {op['call_count']}")
    
    # Get operation details
    op_details = observability_manager.get_operation_details(op['operation'])
    print(f"  Bottleneck: {op_details['bottleneck']}")
    print(f"  Recommendation: {op_details['recommendation']}")
```

#### Missing Metrics

```python
# Validate metrics collection
metrics_status = observability_manager.validate_metrics_collection()
if not metrics_status['is_collecting']:
    print("Metrics collection issues:")
    for issue in metrics_status['issues']:
        print(f"  - {issue}")

# Check metric endpoints
endpoints = observability_manager.get_metrics_endpoints()
for endpoint in endpoints:
    status = observability_manager.check_endpoint_health(endpoint)
    print(f"Endpoint {endpoint}: {'✅' if status else '❌'}")
```

#### Trace Collection Issues

```python
# Debug tracing issues
tracing_status = observability_manager.get_tracing_status()
print(f"Tracing enabled: {tracing_status['enabled']}")
print(f"Active spans: {tracing_status['active_spans']}")
print(f"Traces collected: {tracing_status['traces_collected']}")

if tracing_status['errors']:
    print("Tracing errors:")
    for error in tracing_status['errors']:
        print(f"  - {error}")
```

### Performance Optimization

```python
# Optimize observability overhead
optimization_config = ObservabilityConfig(
    # Reduce sampling for high-volume operations
    tracing_sample_rate=0.1,  # 10% sampling
    
    # Batch metrics collection
    metrics_batch_size=100,
    metrics_flush_interval=30,
    
    # Optimize log output
    log_buffer_size=1000,
    log_async_writing=True,
    
    # Reduce health check frequency
    health_check_interval=60
)

# Monitor observability overhead
overhead_stats = observability_manager.get_observability_overhead()
print(f"CPU overhead: {overhead_stats['cpu_percent']}%")
print(f"Memory overhead: {overhead_stats['memory_mb']} MB")
print(f"Network overhead: {overhead_stats['network_kb']} KB/s")
```

This observability guide provides comprehensive coverage of all monitoring and observability features. Use these configurations and examples to set up effective monitoring for your vector database deployment.