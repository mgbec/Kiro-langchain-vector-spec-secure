# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the LangChain Vector Database.

## Table of Contents

1. [General Troubleshooting](#general-troubleshooting)
2. [Configuration Issues](#configuration-issues)
3. [Authentication and Security](#authentication-and-security)
4. [Storage Backend Issues](#storage-backend-issues)
5. [Embedding Service Issues](#embedding-service-issues)
6. [Performance Issues](#performance-issues)
7. [Memory and Resource Issues](#memory-and-resource-issues)
8. [Network and Connectivity](#network-and-connectivity)
9. [Data Integrity Issues](#data-integrity-issues)
10. [Monitoring and Observability](#monitoring-and-observability)

## General Troubleshooting

### Enable Debug Logging

First step in troubleshooting is to enable debug logging:

```python
from langchain_vector_db.models.config import VectorDBConfig, ObservabilityConfig

config = VectorDBConfig(
    # ... other config ...
    observability=ObservabilityConfig(
        log_level="DEBUG",
        log_format="json",
        log_output="both"  # console and file
    )
)
```

### Health Check

Perform a comprehensive health check:

```python
manager = VectorDatabaseManager(config)

# System health check
health_status = manager.health_check()
print(f"System healthy: {health_status}")

# Detailed health status
if hasattr(manager, 'observability_manager'):
    detailed_health = manager.observability_manager.get_comprehensive_health_status()
    print(f"Overall health: {detailed_health['overall_health']}")
    
    for component, status in detailed_health['checks'].items():
        icon = "✅" if status['healthy'] else "❌"
        print(f"{icon} {component}: {status['message']}")
```

### Check System Resources

```python
# Check resource usage
if hasattr(manager, 'observability_manager'):
    resources = manager.observability_manager.get_resource_usage()
    print(f"Memory: {resources.get('memory_usage_mb', 'N/A')} MB")
    print(f"CPU: {resources.get('cpu_usage_percent', 'N/A')}%")
    print(f"Disk: {resources.get('disk_usage_mb', 'N/A')} MB")
```

## Configuration Issues

### Invalid Configuration

**Problem**: Configuration validation errors

**Symptoms**:
- `ConfigurationException` on startup
- Invalid parameter values
- Missing required configuration

**Solution**:

```python
# Validate configuration before use
try:
    config = VectorDBConfig(
        storage_type="local",
        embedding_provider="openai",
        # ... other config
    )
    
    # Validate configuration
    validation_result = config.validate()
    if not validation_result.is_valid:
        print("Configuration errors:")
        for error in validation_result.errors:
            print(f"  - {error}")
    
except Exception as e:
    print(f"Configuration error: {e}")
```

### Environment Variables

**Problem**: Missing or incorrect environment variables

**Solution**:

```python
import os

# Check required environment variables
required_vars = {
    "OPENAI_API_KEY": "OpenAI API key for embeddings",
    "AWS_ACCESS_KEY_ID": "AWS access key (for S3 storage)",
    "AWS_SECRET_ACCESS_KEY": "AWS secret key (for S3 storage)"
}

missing_vars = []
for var, description in required_vars.items():
    if not os.getenv(var):
        missing_vars.append(f"{var}: {description}")

if missing_vars:
    print("Missing environment variables:")
    for var in missing_vars:
        print(f"  - {var}")
```

### Storage Path Issues

**Problem**: Invalid or inaccessible storage paths

**Solution**:

```python
from pathlib import Path

storage_path = "./vector_db_data"

# Check if path exists and is writable
path = Path(storage_path)
try:
    path.mkdir(parents=True, exist_ok=True)
    
    # Test write permissions
    test_file = path / "test_write.tmp"
    test_file.write_text("test")
    test_file.unlink()
    
    print(f"Storage path {storage_path} is accessible")
    
except PermissionError:
    print(f"Permission denied: Cannot write to {storage_path}")
except Exception as e:
    print(f"Storage path error: {e}")
```

## Authentication and Security

### Authentication Failures

**Problem**: API key authentication fails

**Symptoms**:
- `AuthenticationException`
- "Invalid API key" errors
- Authentication timeouts

**Diagnosis**:

```python
try:
    # Test API key authentication
    security_manager = manager.security_manager
    
    # Check if API key exists
    api_key = "your_api_key_here"
    is_valid = security_manager.validate_api_key(api_key)
    print(f"API key valid: {is_valid}")
    
    # Get API key info
    key_info = security_manager.get_api_key_info(api_key)
    print(f"Key info: {key_info}")
    
    # Test authentication
    auth_token = security_manager.authenticate_api_key(api_key, "user_id")
    print(f"Authentication successful: {auth_token.user_id}")
    
except AuthenticationException as e:
    print(f"Authentication failed: {e}")
    
    # Check common issues
    if "expired" in str(e).lower():
        print("  Issue: API key has expired")
        print("  Solution: Generate a new API key")
    elif "invalid" in str(e).lower():
        print("  Issue: API key is invalid")
        print("  Solution: Check API key format and regenerate if needed")
```

### Permission Denied

**Problem**: Authorization failures

**Solution**:

```python
# Check user permissions
auth_token = security_manager.authenticate_api_key(api_key, user_id)

print(f"User roles: {auth_token.roles}")
print(f"User permissions: {auth_token.permissions}")

# Check specific permission
has_permission = security_manager.authorize_operation(
    auth_token, "documents", "create"
)

if not has_permission:
    print("Permission denied for document creation")
    
    # Get required permissions
    required_perms = security_manager.get_required_permissions("documents", "create")
    print(f"Required permissions: {required_perms}")
    
    # Check what permissions user has
    user_perms = set(auth_token.permissions)
    required_perms_set = set(required_perms)
    missing_perms = required_perms_set - user_perms
    
    if missing_perms:
        print(f"Missing permissions: {list(missing_perms)}")
```

### Rate Limiting Issues

**Problem**: Requests being rate limited

**Solution**:

```python
# Check rate limit status
try:
    # Attempt operation
    manager.add_documents(documents, auth_token=auth_token)
    
except VectorDBException as e:
    if "rate limit" in str(e).lower():
        # Get rate limit status
        status = security_manager.get_rate_limit_status(
            user_id=auth_token.user_id,
            ip_address="client_ip"
        )
        
        print(f"Rate limit exceeded:")
        print(f"  Requests remaining: {status['requests_remaining']}")
        print(f"  Reset time: {status['reset_time']}")
        print(f"  Window: {status['window_size']} seconds")
        
        # Wait for reset or request limit increase
        reset_in = status['reset_time'] - datetime.utcnow()
        print(f"  Reset in: {reset_in.total_seconds()} seconds")
```

## Storage Backend Issues

### Local Storage Issues

**Problem**: Local FAISS storage errors

**Symptoms**:
- File permission errors
- Corrupted index files
- Disk space issues

**Diagnosis**:

```python
import os
from pathlib import Path

storage_path = config.local_storage_path

# Check disk space
stat = os.statvfs(storage_path)
free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
print(f"Free disk space: {free_space_gb:.2f} GB")

if free_space_gb < 1.0:
    print("Warning: Low disk space")

# Check for corrupted files
index_files = list(Path(storage_path).glob("*.faiss"))
metadata_files = list(Path(storage_path).glob("*.json"))

print(f"Found {len(index_files)} index files")
print(f"Found {len(metadata_files)} metadata files")

# Validate file integrity
for index_file in index_files:
    try:
        # Try to load FAISS index
        import faiss
        index = faiss.read_index(str(index_file))
        print(f"✅ {index_file.name}: {index.ntotal} vectors")
    except Exception as e:
        print(f"❌ {index_file.name}: Corrupted ({e})")
```

**Solution**:

```python
# Repair corrupted storage
def repair_local_storage(storage_path):
    backup_path = Path(storage_path + "_backup")
    
    # Create backup
    if Path(storage_path).exists():
        shutil.copytree(storage_path, backup_path, dirs_exist_ok=True)
        print(f"Created backup at {backup_path}")
    
    # Clear corrupted files
    corrupted_files = []  # List of corrupted files from diagnosis
    for file_path in corrupted_files:
        file_path.unlink()
        print(f"Removed corrupted file: {file_path}")
    
    # Reinitialize storage
    manager = VectorDatabaseManager(config)
    print("Storage reinitialized")

# repair_local_storage(config.local_storage_path)
```

### S3 Storage Issues

**Problem**: AWS S3 connectivity or permission issues

**Symptoms**:
- Connection timeouts
- Access denied errors
- Bucket not found errors

**Diagnosis**:

```python
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

try:
    # Test S3 connection
    s3_client = boto3.client(
        's3',
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name=config.aws_region
    )
    
    # Test bucket access
    bucket_name = config.s3_bucket_name
    
    # Check if bucket exists
    s3_client.head_bucket(Bucket=bucket_name)
    print(f"✅ Bucket {bucket_name} is accessible")
    
    # Test write permissions
    test_key = f"{config.s3_key_prefix}test_write.txt"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=test_key,
        Body=b"test"
    )
    
    # Clean up test object
    s3_client.delete_object(Bucket=bucket_name, Key=test_key)
    print("✅ Write permissions confirmed")
    
except NoCredentialsError:
    print("❌ AWS credentials not found")
    print("  Solution: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    
except ClientError as e:
    error_code = e.response['Error']['Code']
    
    if error_code == 'NoSuchBucket':
        print(f"❌ Bucket {bucket_name} does not exist")
        print(f"  Solution: Create bucket or check bucket name")
    elif error_code == 'AccessDenied':
        print(f"❌ Access denied to bucket {bucket_name}")
        print("  Solution: Check IAM permissions")
    else:
        print(f"❌ S3 error: {error_code} - {e.response['Error']['Message']}")

except Exception as e:
    print(f"❌ S3 connection error: {e}")
```

**Solution for S3 Issues**:

```python
# Create S3 bucket if it doesn't exist
def create_s3_bucket(bucket_name, region):
    s3_client = boto3.client('s3', region_name=region)
    
    try:
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"Created bucket: {bucket_name}")
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyExists':
            print(f"Bucket {bucket_name} already exists")
        else:
            raise

# Set up proper IAM policy
iam_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                f"arn:aws:s3:::{config.s3_bucket_name}",
                f"arn:aws:s3:::{config.s3_bucket_name}/*"
            ]
        }
    ]
}

print("Required IAM policy:")
print(json.dumps(iam_policy, indent=2))
```

## Embedding Service Issues

### OpenAI API Issues

**Problem**: OpenAI embedding service failures

**Symptoms**:
- API key errors
- Rate limiting from OpenAI
- Network timeouts

**Diagnosis**:

```python
import openai

# Test OpenAI API key
try:
    openai.api_key = config.api_key
    
    # Test with a simple embedding request
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input="test embedding"
    )
    
    print("✅ OpenAI API key is valid")
    print(f"Embedding dimension: {len(response['data'][0]['embedding'])}")
    
except openai.error.AuthenticationError:
    print("❌ Invalid OpenAI API key")
    print("  Solution: Check OPENAI_API_KEY environment variable")
    
except openai.error.RateLimitError:
    print("❌ OpenAI rate limit exceeded")
    print("  Solution: Wait or upgrade OpenAI plan")
    
except openai.error.APIError as e:
    print(f"❌ OpenAI API error: {e}")
    
except Exception as e:
    print(f"❌ Embedding service error: {e}")
```

### Embedding Dimension Mismatch

**Problem**: Embedding dimension inconsistencies

**Solution**:

```python
# Check embedding dimensions
try:
    # Get expected dimension
    expected_dim = manager.get_embedding_dimension()
    print(f"Expected embedding dimension: {expected_dim}")
    
    # Test embedding generation
    test_text = "test document for dimension check"
    embedding = manager.embedding_service.embed_query(test_text)
    actual_dim = len(embedding)
    
    print(f"Actual embedding dimension: {actual_dim}")
    
    if expected_dim != actual_dim:
        print("❌ Dimension mismatch detected")
        print("  Solution: Recreate vector store with correct dimensions")
        
        # Clear and reinitialize
        manager.clear()
        print("Vector store cleared and reinitialized")
    
except Exception as e:
    print(f"Embedding dimension check failed: {e}")
```

## Performance Issues

### Slow Search Performance

**Problem**: Similarity searches are slow

**Diagnosis**:

```python
import time

# Measure search performance
def benchmark_search():
    queries = [
        "artificial intelligence",
        "machine learning algorithms",
        "natural language processing",
        "vector database search",
        "embedding similarity"
    ]
    
    times = []
    for query in queries:
        start_time = time.time()
        results = manager.similarity_search(query, k=5)
        duration = time.time() - start_time
        times.append(duration)
        
        print(f"Query: '{query}' - {duration:.3f}s ({len(results)} results)")
    
    avg_time = sum(times) / len(times)
    print(f"Average search time: {avg_time:.3f}s")
    
    if avg_time > 2.0:
        print("⚠️  Search performance is slow")
        return False
    return True

is_fast = benchmark_search()
```

**Solutions**:

```python
# Optimize vector store configuration
optimized_config = VectorDBConfig(
    storage_type="local",
    # Use faster FAISS index type
    faiss_index_type="IVF",  # Instead of Flat
    faiss_nlist=100,  # Number of clusters
    
    # Optimize embedding batch size
    embedding_batch_size=32,
    
    # Enable caching
    enable_embedding_cache=True,
    cache_size=1000
)

# Rebuild index with optimization
manager_optimized = VectorDatabaseManager(optimized_config)

# For existing data, rebuild index
if not is_fast:
    print("Rebuilding index for better performance...")
    
    # Export existing documents
    all_docs = []
    doc_count = manager.get_document_count()
    
    # Get all documents (implement pagination if needed)
    for i in range(doc_count):
        try:
            doc = manager.get_document(f"doc_{i}")
            if doc:
                all_docs.append(doc)
        except:
            continue
    
    # Add to optimized manager
    if all_docs:
        manager_optimized.add_documents(all_docs)
        print(f"Rebuilt index with {len(all_docs)} documents")
```

### High Memory Usage

**Problem**: Excessive memory consumption

**Diagnosis**:

```python
import psutil
import gc

# Check memory usage
process = psutil.Process()
memory_info = process.memory_info()

print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
print(f"Virtual memory: {memory_info.vms / 1024 / 1024:.2f} MB")

# Check for memory leaks
gc.collect()  # Force garbage collection

# Monitor memory during operations
def monitor_memory_usage():
    initial_memory = process.memory_info().rss
    
    # Perform operations
    documents = [Document(page_content=f"Test document {i}") for i in range(100)]
    manager.add_documents(documents)
    
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / 1024 / 1024
    
    print(f"Memory increase: {memory_increase:.2f} MB")
    
    if memory_increase > 100:  # More than 100MB increase
        print("⚠️  Potential memory leak detected")

monitor_memory_usage()
```

**Solutions**:

```python
# Optimize memory usage
memory_optimized_config = VectorDBConfig(
    # Reduce batch sizes
    embedding_batch_size=16,
    document_batch_size=50,
    
    # Enable memory optimization
    optimize_memory_usage=True,
    max_memory_usage_mb=1024,
    
    # Use memory-efficient storage
    use_memory_mapped_storage=True
)

# Implement memory cleanup
def cleanup_memory():
    # Clear caches
    if hasattr(manager, 'embedding_service'):
        manager.embedding_service.clear_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Clear temporary data
    if hasattr(manager, 'vector_store'):
        manager.vector_store.cleanup_temp_data()

# Call cleanup periodically
cleanup_memory()
```

## Memory and Resource Issues

### Out of Memory Errors

**Problem**: System runs out of memory

**Solution**:

```python
# Implement batch processing for large datasets
def process_large_dataset(documents, batch_size=50):
    total_docs = len(documents)
    processed = 0
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        
        try:
            doc_ids = manager.add_documents(batch)
            processed += len(doc_ids)
            
            print(f"Processed {processed}/{total_docs} documents")
            
            # Cleanup after each batch
            gc.collect()
            
            # Persist periodically
            if processed % (batch_size * 10) == 0:
                manager.persist()
                
        except MemoryError:
            print(f"Memory error at batch {i//batch_size + 1}")
            print("Reducing batch size...")
            
            # Reduce batch size and retry
            smaller_batch_size = max(1, batch_size // 2)
            return process_large_dataset(documents[i:], smaller_batch_size)
    
    return processed

# Use for large datasets
# processed_count = process_large_dataset(large_document_list)
```

### Disk Space Issues

**Problem**: Running out of disk space

**Solution**:

```python
import shutil

def check_and_cleanup_disk_space(storage_path, min_free_gb=1.0):
    # Check available space
    total, used, free = shutil.disk_usage(storage_path)
    free_gb = free / (1024**3)
    
    print(f"Disk space - Total: {total/(1024**3):.2f} GB, "
          f"Used: {used/(1024**3):.2f} GB, "
          f"Free: {free_gb:.2f} GB")
    
    if free_gb < min_free_gb:
        print(f"⚠️  Low disk space: {free_gb:.2f} GB remaining")
        
        # Cleanup old files
        cleanup_old_files(storage_path)
        
        # Compress data
        compress_storage_data(storage_path)
        
        # Check space again
        _, _, free_after = shutil.disk_usage(storage_path)
        free_after_gb = free_after / (1024**3)
        
        print(f"After cleanup: {free_after_gb:.2f} GB free")
        
        if free_after_gb < min_free_gb:
            print("❌ Still insufficient disk space")
            print("  Solution: Add more storage or move to cloud storage")
            return False
    
    return True

def cleanup_old_files(storage_path):
    # Remove temporary files
    temp_files = Path(storage_path).glob("*.tmp")
    for temp_file in temp_files:
        temp_file.unlink()
        print(f"Removed temp file: {temp_file}")
    
    # Remove old backup files
    backup_files = Path(storage_path).glob("*.backup")
    for backup_file in backup_files:
        if backup_file.stat().st_mtime < time.time() - 7*24*3600:  # 7 days old
            backup_file.unlink()
            print(f"Removed old backup: {backup_file}")

# check_and_cleanup_disk_space(config.local_storage_path)
```

## Network and Connectivity

### Connection Timeouts

**Problem**: Network timeouts when accessing external services

**Solution**:

```python
# Configure timeouts and retries
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def configure_robust_http_client():
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set timeouts
    session.timeout = (10, 30)  # (connect, read) timeouts
    
    return session

# Use robust HTTP client for API calls
robust_session = configure_robust_http_client()

# Test connectivity
def test_connectivity():
    endpoints = [
        "https://api.openai.com/v1/models",
        "https://s3.amazonaws.com",
    ]
    
    for endpoint in endpoints:
        try:
            response = robust_session.get(endpoint, timeout=10)
            print(f"✅ {endpoint}: {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"❌ {endpoint}: Timeout")
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint}: Connection error")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")

test_connectivity()
```

### Proxy Configuration

**Problem**: Network requests fail due to proxy settings

**Solution**:

```python
import os

# Configure proxy settings
proxy_config = {
    'http': os.getenv('HTTP_PROXY'),
    'https': os.getenv('HTTPS_PROXY'),
}

# Remove None values
proxy_config = {k: v for k, v in proxy_config.items() if v}

if proxy_config:
    print(f"Using proxy configuration: {proxy_config}")
    
    # Apply to requests
    import requests
    requests.adapters.DEFAULT_RETRIES = 3
    
    # Test with proxy
    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            proxies=proxy_config,
            timeout=30
        )
        print("✅ Proxy connection successful")
    except Exception as e:
        print(f"❌ Proxy connection failed: {e}")
```

## Data Integrity Issues

### Corrupted Vector Index

**Problem**: Vector index becomes corrupted

**Symptoms**:
- Search results are inconsistent
- Index loading fails
- Dimension mismatch errors

**Solution**:

```python
def repair_vector_index():
    print("Repairing vector index...")
    
    # Backup current data
    backup_path = f"{config.local_storage_path}_backup_{int(time.time())}"
    if Path(config.local_storage_path).exists():
        shutil.copytree(config.local_storage_path, backup_path)
        print(f"Created backup at: {backup_path}")
    
    # Extract documents from metadata
    documents = []
    metadata_files = Path(config.local_storage_path).glob("*.json")
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Reconstruct documents
            for doc_data in metadata.get('documents', []):
                doc = Document(
                    doc_id=doc_data['doc_id'],
                    page_content=doc_data['page_content'],
                    metadata=doc_data.get('metadata', {})
                )
                documents.append(doc)
                
        except Exception as e:
            print(f"Error reading {metadata_file}: {e}")
    
    print(f"Recovered {len(documents)} documents")
    
    # Clear corrupted index
    manager.clear()
    
    # Rebuild index
    if documents:
        print("Rebuilding vector index...")
        doc_ids = manager.add_documents(documents, generate_embeddings=True)
        print(f"Rebuilt index with {len(doc_ids)} documents")
        
        # Persist the repaired index
        manager.persist()
        print("Index repair completed")
    
    return len(documents)

# Detect corruption
def detect_index_corruption():
    try:
        # Test basic operations
        count = manager.get_document_count()
        test_results = manager.similarity_search("test", k=1)
        
        # Check for dimension consistency
        if hasattr(manager, 'vector_store'):
            expected_dim = manager.get_embedding_dimension()
            # Additional corruption checks...
            
        print("✅ Index appears to be healthy")
        return False
        
    except Exception as e:
        print(f"❌ Index corruption detected: {e}")
        return True

# if detect_index_corruption():
#     repair_vector_index()
```

### Missing Documents

**Problem**: Documents appear to be missing

**Solution**:

```python
def audit_document_integrity():
    print("Auditing document integrity...")
    
    # Get document count
    reported_count = manager.get_document_count()
    print(f"Reported document count: {reported_count}")
    
    # Count actual documents
    actual_documents = []
    missing_documents = []
    
    # Check metadata files
    metadata_files = Path(config.local_storage_path).glob("*.json")
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for doc_data in metadata.get('documents', []):
                doc_id = doc_data['doc_id']
                
                # Try to retrieve document
                try:
                    doc = manager.get_document(doc_id)
                    if doc:
                        actual_documents.append(doc_id)
                    else:
                        missing_documents.append(doc_id)
                except:
                    missing_documents.append(doc_id)
                    
        except Exception as e:
            print(f"Error reading metadata: {e}")
    
    print(f"Actual documents found: {len(actual_documents)}")
    print(f"Missing documents: {len(missing_documents)}")
    
    if missing_documents:
        print("Missing document IDs:")
        for doc_id in missing_documents[:10]:  # Show first 10
            print(f"  - {doc_id}")
        
        if len(missing_documents) > 10:
            print(f"  ... and {len(missing_documents) - 10} more")
    
    return {
        'reported_count': reported_count,
        'actual_count': len(actual_documents),
        'missing_count': len(missing_documents),
        'missing_ids': missing_documents
    }

# audit_result = audit_document_integrity()
```

## Monitoring and Observability

### Missing Metrics

**Problem**: Metrics are not being collected

**Solution**:

```python
def diagnose_metrics_collection():
    if not hasattr(manager, 'observability_manager'):
        print("❌ Observability manager not initialized")
        print("  Solution: Enable observability in configuration")
        return
    
    obs_manager = manager.observability_manager
    
    # Check if metrics are enabled
    if not obs_manager.metrics_enabled:
        print("❌ Metrics collection is disabled")
        print("  Solution: Set metrics_enabled=True in ObservabilityConfig")
        return
    
    # Test metrics collection
    try:
        system_metrics = obs_manager.get_system_metrics()
        print("✅ System metrics available:")
        print(f"  Request count: {system_metrics.request_count}")
        print(f"  Error count: {system_metrics.error_count}")
        
    except Exception as e:
        print(f"❌ Error collecting system metrics: {e}")
    
    # Test custom metrics
    try:
        obs_manager.record_business_metric(
            "test_metric", 1, "Test metric", "count"
        )
        print("✅ Custom metrics recording works")
        
    except Exception as e:
        print(f"❌ Error recording custom metrics: {e}")

diagnose_metrics_collection()
```

### Log Analysis Issues

**Problem**: Logs are not helpful for debugging

**Solution**:

```python
def improve_logging():
    # Enable more detailed logging
    detailed_config = ObservabilityConfig(
        log_level="DEBUG",
        log_format="json",
        include_correlation_id=True,
        include_request_id=True,
        include_user_context=True,
        include_performance_data=True,
        include_stack_trace=True
    )
    
    # Update configuration
    config.observability = detailed_config
    
    # Reinitialize manager
    new_manager = VectorDatabaseManager(config)
    
    print("Enhanced logging configuration applied")
    return new_manager

# Analyze recent logs for patterns
def analyze_log_patterns():
    if hasattr(manager, 'observability_manager'):
        recent_logs = manager.observability_manager.get_recent_logs(100)
        
        # Count log levels
        level_counts = {}
        error_patterns = {}
        
        for log_entry in recent_logs:
            log_dict = log_entry.to_dict()
            level = log_dict.get('level', 'UNKNOWN')
            level_counts[level] = level_counts.get(level, 0) + 1
            
            # Analyze error patterns
            if level == 'ERROR':
                message = log_dict.get('message', '')
                # Extract error type
                error_type = message.split(':')[0] if ':' in message else message
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        print("Log level distribution:")
        for level, count in level_counts.items():
            print(f"  {level}: {count}")
        
        if error_patterns:
            print("\\nError patterns:")
            for error_type, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count} occurrences")

analyze_log_patterns()
```

## Emergency Recovery Procedures

### Complete System Recovery

**Problem**: System is completely non-functional

**Solution**:

```python
def emergency_recovery():
    print("Starting emergency recovery procedure...")
    
    # Step 1: Backup current state
    timestamp = int(time.time())
    backup_dir = f"emergency_backup_{timestamp}"
    
    if Path(config.local_storage_path).exists():
        shutil.copytree(config.local_storage_path, backup_dir)
        print(f"Created emergency backup: {backup_dir}")
    
    # Step 2: Reset to minimal configuration
    minimal_config = VectorDBConfig(
        storage_type="local",
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002",
        local_storage_path=f"./recovery_db_{timestamp}",
        api_key=os.getenv("OPENAI_API_KEY"),
        
        # Minimal observability
        observability=ObservabilityConfig(
            log_level="INFO",
            metrics_enabled=False,
            tracing_enabled=False
        ),
        
        # Minimal security
        security=SecurityConfig(
            auth_enabled=False,
            encryption_enabled=False
        )
    )
    
    # Step 3: Initialize with minimal config
    try:
        recovery_manager = VectorDatabaseManager(minimal_config)
        print("✅ Recovery manager initialized")
        
        # Test basic functionality
        test_doc = Document(page_content="Recovery test document")
        doc_ids = recovery_manager.add_documents([test_doc])
        
        if doc_ids:
            print("✅ Basic functionality confirmed")
            return recovery_manager
        else:
            print("❌ Basic functionality test failed")
            return None
            
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        return None

# Use only in emergency situations
# recovery_manager = emergency_recovery()
```

This troubleshooting guide covers the most common issues you might encounter. For additional help, check the logs, enable debug mode, and consult the API documentation for specific error codes and solutions.