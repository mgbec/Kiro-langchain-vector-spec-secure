# Performance Tuning and Best Practices

This guide provides comprehensive recommendations for optimizing the performance of your LangChain Vector Database deployment.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Configuration Optimization](#configuration-optimization)
3. [Storage Backend Optimization](#storage-backend-optimization)
4. [Embedding Service Optimization](#embedding-service-optimization)
5. [Memory Management](#memory-management)
6. [Indexing Strategies](#indexing-strategies)
7. [Search Optimization](#search-optimization)
8. [Batch Processing](#batch-processing)
9. [Caching Strategies](#caching-strategies)
10. [Monitoring and Profiling](#monitoring-and-profiling)

## Performance Overview

### Key Performance Metrics

Monitor these critical metrics for optimal performance:

- **Indexing Throughput**: Documents indexed per second
- **Search Latency**: Time to return search results (P95, P99)
- **Memory Usage**: RAM consumption during operations
- **CPU Utilization**: Processing overhead
- **Storage I/O**: Read/write operations per second
- **Network Latency**: For cloud storage backends

### Performance Targets

Recommended performance targets for different deployment sizes:

| Deployment Size | Documents | Search Latency (P95) | Indexing Rate | Memory Usage |
|----------------|-----------|---------------------|---------------|--------------|
| Small          | < 10K     | < 100ms            | > 100 docs/s  | < 2GB        |
| Medium         | 10K-100K  | < 200ms            | > 50 docs/s   | < 8GB        |
| Large          | 100K-1M   | < 500ms            | > 20 docs/s   | < 32GB       |
| Enterprise     | > 1M      | < 1000ms           | > 10 docs/s   | < 128GB      |

## Configuration Optimization

### Basic Performance Configuration

```python
from langchain_vector_db.models.config import VectorDBConfig, ObservabilityConfig

# Optimized configuration for performance
performance_config = VectorDBConfig(
    storage_type="local",  # or "s3" for cloud
    embedding_provider="openai",
    embedding_model="text-embedding-ada-002",
    
    # Batch processing optimization
    embedding_batch_size=64,  # Increase for better throughput
    document_batch_size=100,  # Process documents in batches
    
    # Memory optimization
    max_memory_usage_mb=8192,  # Set appropriate memory limit
    use_memory_mapped_storage=True,  # Reduce memory footprint
    
    # I/O optimization
    io_thread_pool_size=8,  # Parallel I/O operations
    async_processing=True,  # Enable async operations
    
    # Caching
    enable_embedding_cache=True,
    cache_size=10000,  # Cache frequently used embeddings
    cache_ttl_seconds=3600,  # 1 hour cache TTL
    
    # Observability (minimal for production)
    observability=ObservabilityConfig(
        log_level="WARN",  # Reduce logging overhead
        metrics_enabled=True,
        tracing_enabled=False,  # Disable in production for performance
        performance_monitoring_enabled=True
    )
)
```

### Environment-Specific Configurations

#### Development Environment

```python
dev_config = VectorDBConfig(
    # Smaller batch sizes for faster feedback
    embedding_batch_size=16,
    document_batch_size=25,
    
    # More detailed observability
    observability=ObservabilityConfig(
        log_level="DEBUG",
        tracing_enabled=True,
        performance_monitoring_enabled=True
    )
)
```

#### Production Environment

```python
prod_config = VectorDBConfig(
    # Larger batch sizes for throughput
    embedding_batch_size=128,
    document_batch_size=500,
    
    # Optimized for performance
    max_memory_usage_mb=16384,
    io_thread_pool_size=16,
    
    # Minimal observability overhead
    observability=ObservabilityConfig(
        log_level="ERROR",
        metrics_enabled=True,
        tracing_enabled=False
    )
)
```

## Storage Backend Optimization

### Local Storage (FAISS) Optimization

```python
# Optimized local storage configuration
local_config = VectorDBConfig(
    storage_type="local",
    local_storage_path="/fast/ssd/vector_db",  # Use SSD storage
    
    # FAISS index optimization
    faiss_index_type="IVF",  # Use inverted file index
    faiss_nlist=1024,  # Number of clusters (sqrt(N) is good)
    faiss_nprobe=64,   # Search clusters (balance speed vs accuracy)
    
    # Memory mapping for large indices
    use_memory_mapped_storage=True,
    memory_map_threshold_mb=1024,
    
    # Parallel processing
    faiss_omp_threads=8,  # OpenMP threads for FAISS
)

# Advanced FAISS configuration
def configure_advanced_faiss():
    import faiss
    
    # Create optimized index factory
    dimension = 1536  # OpenAI embedding dimension
    
    if dimension <= 768:
        # For smaller dimensions, use HNSW
        index_factory = f"HNSW{dimension},Flat"
    else:
        # For larger dimensions, use IVF with PQ compression
        nlist = min(4096, max(64, int(np.sqrt(num_documents))))
        index_factory = f"IVF{nlist},PQ{dimension//8}x8"
    
    return index_factory
```

### S3 Storage Optimization

```python
# Optimized S3 configuration
s3_config = VectorDBConfig(
    storage_type="s3",
    s3_bucket_name="my-vector-db-bucket",
    s3_key_prefix="vector_db/",
    aws_region="us-east-1",  # Choose region close to your application
    
    # S3 optimization
    s3_batch_size=1000,  # Larger batches for S3
    s3_multipart_threshold=64 * 1024 * 1024,  # 64MB
    s3_multipart_chunksize=16 * 1024 * 1024,  # 16MB chunks
    s3_max_concurrency=10,  # Parallel uploads
    s3_use_threads=True,
    
    # Connection pooling
    s3_max_pool_connections=50,
    
    # Retry configuration
    s3_retry_attempts=3,
    s3_retry_backoff_factor=2,
    
    # Caching for S3
    enable_s3_cache=True,
    s3_cache_size_mb=1024,  # 1GB local cache
)

# S3 Transfer Acceleration
def enable_s3_acceleration():
    import boto3
    
    s3_client = boto3.client('s3')
    s3_client.put_bucket_accelerate_configuration(
        Bucket='my-vector-db-bucket',
        AccelerateConfiguration={'Status': 'Enabled'}
    )
    
    # Use accelerated endpoint
    accelerated_config = s3_config.copy()
    accelerated_config.s3_endpoint_url = 'https://s3-accelerate.amazonaws.com'
    return accelerated_config
```

## Embedding Service Optimization

### OpenAI API Optimization

```python
# Optimized embedding service configuration
embedding_config = {
    # Batch processing
    "batch_size": 100,  # OpenAI allows up to 2048 inputs
    "max_tokens_per_batch": 8000,  # Stay within token limits
    
    # Connection optimization
    "max_retries": 3,
    "timeout": 30,
    "connection_pool_size": 20,
    
    # Rate limiting
    "requests_per_minute": 3000,  # Adjust based on your OpenAI plan
    "tokens_per_minute": 1000000,
}

# Implement smart batching
class OptimizedEmbeddingService:
    def __init__(self, config):
        self.config = config
        self.batch_size = config.get("batch_size", 100)
        self.max_tokens = config.get("max_tokens_per_batch", 8000)
    
    def embed_documents_optimized(self, texts):
        """Optimized batch embedding with token counting."""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            # Estimate tokens (rough approximation)
            text_tokens = len(text.split()) * 1.3
            
            if (len(current_batch) >= self.batch_size or 
                current_tokens + text_tokens > self.max_tokens):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = [text]
                    current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        # Process batches in parallel
        embeddings = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self._embed_batch, batch) 
                for batch in batches
            ]
            
            for future in futures:
                batch_embeddings = future.result()
                embeddings.extend(batch_embeddings)
        
        return embeddings
```

### Alternative Embedding Providers

```python
# HuggingFace local embeddings for better performance
huggingface_config = VectorDBConfig(
    embedding_provider="huggingface",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    
    # Local processing - no API calls
    device="cuda",  # Use GPU if available
    batch_size=256,  # Larger batches for local processing
    
    # Model optimization
    use_fp16=True,  # Half precision for speed
    normalize_embeddings=True,
)

# Cohere embeddings for multilingual content
cohere_config = VectorDBConfig(
    embedding_provider="cohere",
    embedding_model="embed-multilingual-v2.0",
    
    # Cohere-specific optimizations
    batch_size=96,  # Cohere's batch limit
    truncate="END",  # Handle long texts
)
```

## Memory Management

### Memory-Efficient Configuration

```python
# Memory-optimized configuration
memory_config = VectorDBConfig(
    # Limit memory usage
    max_memory_usage_mb=4096,  # 4GB limit
    memory_growth_limit_mb=512,  # Alert if growth exceeds 512MB
    
    # Use memory mapping
    use_memory_mapped_storage=True,
    memory_map_threshold_mb=100,
    
    # Garbage collection optimization
    gc_threshold=1000,  # Trigger GC after 1000 operations
    enable_memory_profiling=True,
    
    # Streaming for large datasets
    enable_streaming_mode=True,
    stream_chunk_size=1000,
)

# Implement memory monitoring
class MemoryMonitor:
    def __init__(self, threshold_mb=1024):
        self.threshold_mb = threshold_mb
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self):
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def check_memory_usage(self):
        current_memory = self.get_memory_usage()
        memory_increase = current_memory - self.initial_memory
        
        if memory_increase > self.threshold_mb:
            print(f"⚠️  Memory usage increased by {memory_increase:.2f} MB")
            return False
        return True
    
    def cleanup_if_needed(self):
        if not self.check_memory_usage():
            import gc
            gc.collect()
            print("🧹 Performed garbage collection")

# Use memory monitor
memory_monitor = MemoryMonitor(threshold_mb=512)
```

### Memory-Efficient Document Processing

```python
def process_documents_memory_efficient(documents, manager, batch_size=50):
    """Process documents with memory efficiency."""
    total_docs = len(documents)
    processed = 0
    
    # Process in smaller batches
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        
        # Process batch
        doc_ids = manager.add_documents(batch)
        processed += len(doc_ids)
        
        # Memory cleanup after each batch
        if processed % (batch_size * 5) == 0:  # Every 5 batches
            import gc
            gc.collect()
            
            # Persist to free memory
            manager.persist()
            
            print(f"Processed {processed}/{total_docs} documents, "
                  f"Memory: {memory_monitor.get_memory_usage():.2f} MB")
    
    return processed
```

## Indexing Strategies

### Optimized Index Configuration

```python
# Choose index type based on dataset size and requirements
def get_optimal_index_config(num_documents, dimension, accuracy_requirement="high"):
    """Get optimal FAISS index configuration."""
    
    if num_documents < 1000:
        # Small dataset - use flat index
        return {
            "index_type": "Flat",
            "metric": "L2"
        }
    
    elif num_documents < 100000:
        # Medium dataset - use IVF
        nlist = max(64, min(4096, int(np.sqrt(num_documents))))
        return {
            "index_type": "IVF",
            "nlist": nlist,
            "nprobe": nlist // 8 if accuracy_requirement == "high" else nlist // 16,
            "metric": "L2"
        }
    
    else:
        # Large dataset - use IVF with compression
        nlist = max(1024, min(16384, int(np.sqrt(num_documents))))
        
        if accuracy_requirement == "high":
            return {
                "index_type": "IVF",
                "nlist": nlist,
                "nprobe": nlist // 4,
                "metric": "L2"
            }
        else:
            # Use product quantization for memory efficiency
            m = min(64, dimension // 8)  # Number of subquantizers
            return {
                "index_type": "IVFPQ",
                "nlist": nlist,
                "nprobe": nlist // 8,
                "m": m,
                "nbits": 8,
                "metric": "L2"
            }

# Apply optimal configuration
optimal_config = get_optimal_index_config(
    num_documents=50000,
    dimension=1536,
    accuracy_requirement="high"
)

print(f"Optimal index configuration: {optimal_config}")
```

### Index Building Strategies

```python
# Progressive index building for large datasets
class ProgressiveIndexBuilder:
    def __init__(self, manager, target_size=10000):
        self.manager = manager
        self.target_size = target_size
        self.current_size = 0
    
    def add_documents_progressive(self, documents):
        """Add documents with progressive index optimization."""
        
        for i, doc in enumerate(documents):
            self.manager.add_documents([doc])
            self.current_size += 1
            
            # Rebuild index when reaching target size
            if self.current_size % self.target_size == 0:
                print(f"Rebuilding index at {self.current_size} documents...")
                self.rebuild_index_optimized()
    
    def rebuild_index_optimized(self):
        """Rebuild index with optimal parameters for current size."""
        
        # Get optimal configuration for current size
        config = get_optimal_index_config(
            self.current_size, 
            1536,  # OpenAI embedding dimension
            "high"
        )
        
        # Apply new configuration
        self.manager.vector_store.rebuild_index(config)
        print(f"Index rebuilt with config: {config}")

# Use progressive builder for large datasets
# builder = ProgressiveIndexBuilder(manager)
# builder.add_documents_progressive(large_document_list)
```

## Search Optimization

### Search Performance Tuning

```python
# Optimized search configuration
search_config = {
    # Result caching
    "enable_result_cache": True,
    "cache_size": 10000,
    "cache_ttl_seconds": 300,  # 5 minutes
    
    # Search parameters
    "default_k": 10,  # Reasonable default
    "max_k": 100,     # Prevent excessive results
    
    # Parallel search
    "enable_parallel_search": True,
    "search_threads": 4,
    
    # Early termination
    "enable_early_termination": True,
    "min_score_threshold": 0.1,
}

# Implement search result caching
from functools import lru_cache
import hashlib

class OptimizedSearchManager:
    def __init__(self, manager, cache_size=1000):
        self.manager = manager
        self.cache_size = cache_size
        self._search_cache = {}
    
    def _get_cache_key(self, query, k, filters=None):
        """Generate cache key for search query."""
        key_data = f"{query}:{k}:{filters}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def similarity_search_cached(self, query, k=10, filters=None):
        """Cached similarity search."""
        cache_key = self._get_cache_key(query, k, filters)
        
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        
        # Perform search
        results = self.manager.similarity_search(query, k=k)
        
        # Cache results
        self._search_cache[cache_key] = results
        
        # Limit cache size
        if len(self._search_cache) > self.cache_size:
            # Remove oldest entries
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]
        
        return results
```

### Query Optimization

```python
# Query preprocessing for better performance
class QueryOptimizer:
    def __init__(self):
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'])
        self.min_query_length = 3
        self.max_query_length = 500
    
    def optimize_query(self, query):
        """Optimize query for better search performance."""
        
        # Basic cleaning
        query = query.strip().lower()
        
        # Length validation
        if len(query) < self.min_query_length:
            return None
        
        if len(query) > self.max_query_length:
            query = query[:self.max_query_length]
        
        # Remove excessive whitespace
        query = ' '.join(query.split())
        
        # Optional: Remove stop words for certain use cases
        # words = [w for w in query.split() if w not in self.stop_words]
        # query = ' '.join(words)
        
        return query
    
    def should_use_semantic_search(self, query):
        """Determine if semantic search is beneficial."""
        
        # Use semantic search for longer, more complex queries
        word_count = len(query.split())
        
        if word_count >= 3:
            return True
        
        # Check for question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        if any(word in query.lower() for word in question_words):
            return True
        
        return False

# Use query optimizer
query_optimizer = QueryOptimizer()

def optimized_search(manager, query, k=10):
    """Perform optimized search with query preprocessing."""
    
    # Optimize query
    optimized_query = query_optimizer.optimize_query(query)
    if not optimized_query:
        return []
    
    # Choose search strategy
    if query_optimizer.should_use_semantic_search(optimized_query):
        return manager.similarity_search(optimized_query, k=k)
    else:
        # Use simpler search for short queries
        return manager.similarity_search(optimized_query, k=min(k, 5))
```

## Batch Processing

### Optimized Batch Operations

```python
# Batch processing configuration
batch_config = {
    "document_batch_size": 100,
    "embedding_batch_size": 50,
    "search_batch_size": 20,
    "max_concurrent_batches": 5,
    "batch_timeout_seconds": 300,
}

class BatchProcessor:
    def __init__(self, manager, config=None):
        self.manager = manager
        self.config = config or batch_config
        self.executor = ThreadPoolExecutor(
            max_workers=self.config["max_concurrent_batches"]
        )
    
    def process_documents_batch(self, documents):
        """Process documents in optimized batches."""
        
        batch_size = self.config["document_batch_size"]
        batches = [
            documents[i:i + batch_size] 
            for i in range(0, len(documents), batch_size)
        ]
        
        # Process batches in parallel
        futures = []
        for batch in batches:
            future = self.executor.submit(self._process_single_batch, batch)
            futures.append(future)
        
        # Collect results
        all_doc_ids = []
        for future in futures:
            try:
                doc_ids = future.result(timeout=self.config["batch_timeout_seconds"])
                all_doc_ids.extend(doc_ids)
            except Exception as e:
                print(f"Batch processing error: {e}")
        
        return all_doc_ids
    
    def _process_single_batch(self, batch):
        """Process a single batch of documents."""
        return self.manager.add_documents(batch, generate_embeddings=True)
    
    def search_batch(self, queries, k=10):
        """Perform batch search operations."""
        
        # Group queries into batches
        batch_size = self.config["search_batch_size"]
        query_batches = [
            queries[i:i + batch_size]
            for i in range(0, len(queries), batch_size)
        ]
        
        # Process search batches
        all_results = []
        for batch in query_batches:
            batch_results = []
            for query in batch:
                results = self.manager.similarity_search(query, k=k)
                batch_results.append(results)
            all_results.extend(batch_results)
        
        return all_results

# Use batch processor
batch_processor = BatchProcessor(manager, batch_config)

# Process large document sets
# doc_ids = batch_processor.process_documents_batch(large_document_list)

# Batch search
# search_results = batch_processor.search_batch(query_list, k=5)
```

## Caching Strategies

### Multi-Level Caching

```python
# Comprehensive caching configuration
cache_config = {
    # Embedding cache
    "embedding_cache_size": 10000,
    "embedding_cache_ttl": 3600,  # 1 hour
    
    # Search result cache
    "search_cache_size": 5000,
    "search_cache_ttl": 300,  # 5 minutes
    
    # Document cache
    "document_cache_size": 1000,
    "document_cache_ttl": 1800,  # 30 minutes
    
    # Index cache
    "index_cache_enabled": True,
    "index_cache_size_mb": 512,
}

class MultiLevelCache:
    def __init__(self, config):
        self.config = config
        
        # Initialize caches
        self.embedding_cache = {}
        self.search_cache = {}
        self.document_cache = {}
        
        # Cache statistics
        self.cache_stats = {
            "embedding_hits": 0,
            "embedding_misses": 0,
            "search_hits": 0,
            "search_misses": 0,
            "document_hits": 0,
            "document_misses": 0,
        }
    
    def get_embedding_cached(self, text):
        """Get embedding with caching."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.embedding_cache:
            self.cache_stats["embedding_hits"] += 1
            return self.embedding_cache[cache_key]
        
        # Generate embedding
        embedding = self.manager.embedding_service.embed_query(text)
        
        # Cache result
        self.embedding_cache[cache_key] = embedding
        self.cache_stats["embedding_misses"] += 1
        
        # Limit cache size
        if len(self.embedding_cache) > self.config["embedding_cache_size"]:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        return embedding
    
    def get_cache_statistics(self):
        """Get cache performance statistics."""
        stats = self.cache_stats.copy()
        
        # Calculate hit rates
        total_embedding = stats["embedding_hits"] + stats["embedding_misses"]
        total_search = stats["search_hits"] + stats["search_misses"]
        total_document = stats["document_hits"] + stats["document_misses"]
        
        if total_embedding > 0:
            stats["embedding_hit_rate"] = stats["embedding_hits"] / total_embedding
        
        if total_search > 0:
            stats["search_hit_rate"] = stats["search_hits"] / total_search
        
        if total_document > 0:
            stats["document_hit_rate"] = stats["document_hits"] / total_document
        
        return stats

# Implement cache warming
def warm_cache(manager, common_queries):
    """Pre-populate cache with common queries."""
    
    print("Warming up caches...")
    
    for query in common_queries:
        # Warm embedding cache
        manager.embedding_service.embed_query(query)
        
        # Warm search cache
        manager.similarity_search(query, k=5)
    
    print(f"Cache warmed with {len(common_queries)} queries")

# Common queries for cache warming
common_queries = [
    "artificial intelligence",
    "machine learning",
    "natural language processing",
    "deep learning",
    "neural networks"
]

# warm_cache(manager, common_queries)
```

## Monitoring and Profiling

### Performance Monitoring Setup

```python
# Performance monitoring configuration
monitoring_config = ObservabilityConfig(
    performance_monitoring_enabled=True,
    metrics_collection_interval=10,  # seconds
    
    # Performance thresholds
    slow_operation_threshold_ms=1000,
    memory_threshold_mb=2048,
    cpu_threshold_percent=80,
    
    # Profiling
    enable_profiling=True,
    profiling_sample_rate=0.1,  # 10% sampling
    profile_memory_usage=True,
    profile_cpu_usage=True,
)

# Performance profiler
class PerformanceProfiler:
    def __init__(self, manager):
        self.manager = manager
        self.operation_times = {}
        self.memory_usage = []
    
    def profile_operation(self, operation_name):
        """Decorator to profile operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                import time
                import psutil
                
                # Record start state
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                try:
                    # Execute operation
                    result = func(*args, **kwargs)
                    
                    # Record end state
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    # Calculate metrics
                    duration = (end_time - start_time) * 1000  # ms
                    memory_delta = end_memory - start_memory
                    
                    # Store metrics
                    if operation_name not in self.operation_times:
                        self.operation_times[operation_name] = []
                    
                    self.operation_times[operation_name].append({
                        "duration_ms": duration,
                        "memory_delta_mb": memory_delta,
                        "timestamp": end_time
                    })
                    
                    # Log slow operations
                    if duration > 1000:  # > 1 second
                        print(f"⚠️  Slow operation: {operation_name} took {duration:.2f}ms")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ Operation failed: {operation_name} - {e}")
                    raise
            
            return wrapper
        return decorator
    
    def get_performance_report(self):
        """Generate performance report."""
        report = {}
        
        for operation, times in self.operation_times.items():
            if times:
                durations = [t["duration_ms"] for t in times]
                memory_deltas = [t["memory_delta_mb"] for t in times]
                
                report[operation] = {
                    "count": len(times),
                    "avg_duration_ms": sum(durations) / len(durations),
                    "max_duration_ms": max(durations),
                    "min_duration_ms": min(durations),
                    "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
                    "total_memory_delta_mb": sum(memory_deltas)
                }
        
        return report

# Use profiler
profiler = PerformanceProfiler(manager)

# Profile specific operations
@profiler.profile_operation("document_addition")
def add_documents_profiled(documents):
    return manager.add_documents(documents)

@profiler.profile_operation("similarity_search")
def search_profiled(query, k=10):
    return manager.similarity_search(query, k=k)

# Generate performance report
# performance_report = profiler.get_performance_report()
# print(json.dumps(performance_report, indent=2))
```

### Continuous Performance Monitoring

```python
# Automated performance monitoring
class ContinuousMonitor:
    def __init__(self, manager, alert_thresholds=None):
        self.manager = manager
        self.thresholds = alert_thresholds or {
            "search_latency_ms": 1000,
            "memory_usage_mb": 4096,
            "cpu_usage_percent": 85,
            "error_rate_percent": 5
        }
        self.monitoring = True
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        import threading
        import time
        
        def monitor_loop():
            while self.monitoring:
                try:
                    self.check_performance_metrics()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    print(f"Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print("Performance monitoring started")
    
    def check_performance_metrics(self):
        """Check current performance metrics."""
        
        # Get system metrics
        if hasattr(self.manager, 'observability_manager'):
            metrics = self.manager.observability_manager.get_system_metrics()
            
            # Check search latency
            if metrics.avg_response_time_ms > self.thresholds["search_latency_ms"]:
                self.alert("High search latency", 
                          f"{metrics.avg_response_time_ms:.2f}ms")
            
            # Check error rate
            if metrics.request_count > 0:
                error_rate = (metrics.error_count / metrics.request_count) * 100
                if error_rate > self.thresholds["error_rate_percent"]:
                    self.alert("High error rate", f"{error_rate:.2f}%")
        
        # Check memory usage
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_mb > self.thresholds["memory_usage_mb"]:
            self.alert("High memory usage", f"{memory_mb:.2f}MB")
        
        # Check CPU usage
        cpu_percent = psutil.Process().cpu_percent()
        if cpu_percent > self.thresholds["cpu_usage_percent"]:
            self.alert("High CPU usage", f"{cpu_percent:.2f}%")
    
    def alert(self, alert_type, message):
        """Send performance alert."""
        timestamp = datetime.utcnow().isoformat()
        print(f"🚨 PERFORMANCE ALERT [{timestamp}]: {alert_type} - {message}")
        
        # Here you could integrate with alerting systems like:
        # - Slack notifications
        # - Email alerts
        # - PagerDuty
        # - Custom webhooks
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False
        print("Performance monitoring stopped")

# Start continuous monitoring
# monitor = ContinuousMonitor(manager)
# monitor.start_monitoring()
```

This performance tuning guide provides comprehensive strategies for optimizing your vector database deployment. Apply these optimizations incrementally and measure the impact to achieve the best performance for your specific use case.