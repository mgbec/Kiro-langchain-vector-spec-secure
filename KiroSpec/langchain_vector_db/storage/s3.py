"""
S3 vector store implementation using AWS S3 for cloud-based vector storage.
"""

import json
import pickle
import time
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
except ImportError:
    raise ImportError(
        "boto3 is required for S3VectorStore. Install with: pip install boto3"
    )

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is required for S3VectorStore. Install with: pip install faiss-cpu"
    )

from .interface import VectorStoreInterface
from ..models.document import Document
from ..exceptions import StorageException, ConfigurationException, S3Exception


class S3VectorStore(VectorStoreInterface):
    """S3-based vector store implementation using FAISS with S3 persistence."""
    
    def __init__(
        self,
        bucket_name: str,
        s3_prefix: str = "vectors/",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = "us-east-1",
        dimension: Optional[int] = None,
        index_type: str = "flat",
        metric: str = "l2",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the S3 vector store.
        
        Args:
            bucket_name: S3 bucket name for storage
            s3_prefix: Prefix for S3 keys (should end with /)
            aws_access_key_id: AWS access key ID (optional if using IAM/env)
            aws_secret_access_key: AWS secret access key (optional if using IAM/env)
            aws_region: AWS region
            dimension: Dimension of the vectors (auto-detected if None)
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            metric: Distance metric ("l2", "ip" for inner product)
            max_retries: Maximum number of retry attempts for S3 operations
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix if s3_prefix.endswith('/') else s3_prefix + '/'
        self.aws_region = aws_region
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # S3 key paths
        self.index_key = f"{self.s3_prefix}faiss_index.bin"
        self.metadata_key = f"{self.s3_prefix}metadata.json"
        self.documents_key = f"{self.s3_prefix}documents.pkl"
        
        # Initialize data structures
        self.index: Optional[faiss.Index] = None
        self.documents: Dict[str, Document] = {}
        self.id_to_index: Dict[str, int] = {}  # Map doc_id to FAISS index
        self.index_to_id: Dict[int, str] = {}  # Map FAISS index to doc_id
        self.next_index = 0
        
        # Initialize S3 client
        self.s3_client = self._create_s3_client(aws_access_key_id, aws_secret_access_key)
        
        # Validate parameters and S3 connection
        self._validate_parameters()
        self._validate_s3_connection()
        
        # Try to load existing data
        self.load()
    
    def _create_s3_client(
        self, 
        aws_access_key_id: Optional[str], 
        aws_secret_access_key: Optional[str]
    ) -> boto3.client:
        """Create S3 client with proper credentials."""
        try:
            session_kwargs = {"region_name": self.aws_region}
            
            if aws_access_key_id and aws_secret_access_key:
                session_kwargs.update({
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key
                })
            
            session = boto3.Session(**session_kwargs)
            return session.client('s3')
            
        except Exception as e:
            raise S3Exception(f"Failed to create S3 client: {str(e)}")
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if not self.bucket_name:
            raise ConfigurationException("bucket_name cannot be empty")
        
        valid_index_types = ["flat", "ivf", "hnsw"]
        if self.index_type not in valid_index_types:
            raise ConfigurationException(
                f"Invalid index_type '{self.index_type}'. Must be one of: {valid_index_types}"
            )
        
        valid_metrics = ["l2", "ip"]
        if self.metric not in valid_metrics:
            raise ConfigurationException(
                f"Invalid metric '{self.metric}'. Must be one of: {valid_metrics}"
            )
        
        if self.dimension is not None and self.dimension <= 0:
            raise ConfigurationException("dimension must be greater than 0")
        
        if self.max_retries < 0:
            raise ConfigurationException("max_retries cannot be negative")
        
        if self.retry_delay < 0:
            raise ConfigurationException("retry_delay cannot be negative")
    
    def _validate_s3_connection(self) -> None:
        """Validate S3 connection and bucket access."""
        try:
            # Check if bucket exists and is accessible
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise S3Exception(f"S3 bucket '{self.bucket_name}' does not exist")
            elif error_code == '403':
                raise S3Exception(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                raise S3Exception(f"S3 bucket validation failed: {str(e)}")
        except NoCredentialsError:
            raise S3Exception("AWS credentials not found. Please configure credentials.")
        except Exception as e:
            raise S3Exception(f"Failed to validate S3 connection: {str(e)}")
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """Create a FAISS index based on configuration."""
        if self.index_type == "flat":
            if self.metric == "l2":
                return faiss.IndexFlatL2(dimension)
            else:  # ip
                return faiss.IndexFlatIP(dimension)
        
        elif self.index_type == "ivf":
            # IVF (Inverted File) index for faster search on large datasets
            nlist = min(100, max(1, int(np.sqrt(1000))))  # Reasonable default
            quantizer = faiss.IndexFlatL2(dimension) if self.metric == "l2" else faiss.IndexFlatIP(dimension)
            return faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        elif self.index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World) for fast approximate search
            return faiss.IndexHNSWFlat(dimension, 32)  # 32 is a reasonable M value
        
        else:
            raise ConfigurationException(f"Unsupported index type: {self.index_type}")
    
    def _ensure_index_initialized(self, dimension: int) -> None:
        """Ensure the FAISS index is initialized with the correct dimension."""
        if self.index is None:
            if self.dimension is None:
                self.dimension = dimension
            elif self.dimension != dimension:
                raise StorageException(
                    f"Vector dimension mismatch. Expected {self.dimension}, got {dimension}"
                )
            
            self.index = self._create_index(self.dimension)
            
            # Train the index if necessary (for IVF)
            if self.index_type == "ivf" and not self.index.is_trained:
                # We'll train when we have enough vectors
                pass
    
    def _s3_operation_with_retry(self, operation_func, *args, **kwargs):
        """Execute S3 operation with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation_func(*args, **kwargs)
            except (ClientError, BotoCoreError) as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                # Calculate delay with exponential backoff
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
        
        # If we get here, all retries failed
        raise S3Exception(f"S3 operation failed after {self.max_retries + 1} attempts: {str(last_exception)}")
    
    def _upload_to_s3(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Upload data to S3 with retry logic."""
        def upload_operation():
            return self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                ContentType=content_type
            )
        
        self._s3_operation_with_retry(upload_operation)
    
    def _download_from_s3(self, key: str) -> bytes:
        """Download data from S3 with retry logic."""
        def download_operation():
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        
        return self._s3_operation_with_retry(download_operation)
    
    def _key_exists_in_s3(self, key: str) -> bool:
        """Check if a key exists in S3."""
        try:
            def head_operation():
                return self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            
            self._s3_operation_with_retry(head_operation)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise S3Exception(f"Failed to check if key exists: {str(e)}")
    
    def add_vectors(
        self, 
        vectors: List[List[float]], 
        documents: List[Document]
    ) -> List[str]:
        """
        Add vectors and associated documents to the store.
        
        Args:
            vectors: List of vector embeddings
            documents: List of documents with metadata
            
        Returns:
            List of document IDs for the added vectors
        """
        if not vectors or not documents:
            return []
        
        if len(vectors) != len(documents):
            raise StorageException(
                f"Number of vectors ({len(vectors)}) must match number of documents ({len(documents)})"
            )
        
        try:
            # Convert to numpy array
            vector_array = np.array(vectors, dtype=np.float32)
            
            # Ensure index is initialized
            self._ensure_index_initialized(vector_array.shape[1])
            
            # Train index if necessary
            if self.index_type == "ivf" and not self.index.is_trained:
                if len(vectors) >= 100:  # Need enough vectors to train
                    self.index.train(vector_array)
                else:
                    # Not enough vectors to train IVF, fall back to flat index
                    self.index = self._create_index(self.dimension)
            
            # Add vectors to FAISS index
            start_index = self.next_index
            self.index.add(vector_array)
            
            # Store documents and maintain mappings
            doc_ids = []
            for i, document in enumerate(documents):
                doc_id = document.doc_id
                faiss_index = start_index + i
                
                # Store document
                self.documents[doc_id] = document
                
                # Maintain mappings
                self.id_to_index[doc_id] = faiss_index
                self.index_to_id[faiss_index] = doc_id
                
                doc_ids.append(doc_id)
            
            self.next_index += len(vectors)
            
            # Auto-persist to S3
            self.persist()
            
            return doc_ids
            
        except Exception as e:
            if isinstance(e, (StorageException, S3Exception)):
                raise
            raise StorageException(f"Failed to add vectors: {str(e)}")
    
    def search_vectors(
        self, 
        query_vector: List[float], 
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector embedding
            k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.index is None or self.get_vector_count() == 0:
            return []
        
        try:
            # Convert query to numpy array
            query_array = np.array([query_vector], dtype=np.float32)
            
            # Validate dimension
            if query_array.shape[1] != self.dimension:
                raise StorageException(
                    f"Query vector dimension ({query_array.shape[1]}) "
                    f"doesn't match index dimension ({self.dimension})"
                )
            
            # Search in FAISS index
            k = min(k, self.get_vector_count())  # Don't search for more than available
            distances, indices = self.index.search(query_array, k)
            
            # Convert results to documents with scores
            results = []
            for distance, faiss_idx in zip(distances[0], indices[0]):
                if faiss_idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                doc_id = self.index_to_id.get(faiss_idx)
                if doc_id and doc_id in self.documents:
                    document = self.documents[doc_id]
                    
                    # Convert distance to similarity score
                    # For L2: smaller distance = higher similarity
                    # For IP: larger score = higher similarity
                    if self.metric == "l2":
                        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    else:  # ip
                        similarity = float(distance)  # Inner product is already a similarity
                    
                    results.append((document, similarity))
            
            return results
            
        except Exception as e:
            if isinstance(e, StorageException):
                raise
            raise StorageException(f"Failed to search vectors: {str(e)}")
    
    def update_vector(
        self, 
        doc_id: str, 
        vector: List[float], 
        document: Document
    ) -> bool:
        """
        Update an existing vector and document.
        
        Args:
            doc_id: Document ID to update
            vector: New vector embedding
            document: Updated document
            
        Returns:
            True if update was successful
        """
        if doc_id not in self.documents:
            return False
        
        try:
            # Update the document in our storage
            self.documents[doc_id] = document
            
            # Note: The vector in FAISS index is not updated here
            # This is a limitation of the current implementation
            # A full implementation would need to rebuild the index or use a different approach
            
            # Persist changes to S3
            self.persist()
            
            return True
            
        except Exception as e:
            raise StorageException(f"Failed to update vector: {str(e)}")
    
    def delete_vectors(self, doc_ids: List[str]) -> bool:
        """
        Delete vectors by document IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Remove documents and mappings
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    # Remove from documents
                    del self.documents[doc_id]
                    
                    # Remove from mappings
                    if doc_id in self.id_to_index:
                        faiss_idx = self.id_to_index[doc_id]
                        del self.id_to_index[doc_id]
                        if faiss_idx in self.index_to_id:
                            del self.index_to_id[faiss_idx]
            
            # Note: FAISS doesn't support direct deletion of vectors
            # The vectors remain in the index but are no longer accessible
            # In a production system, you might want to rebuild the index periodically
            
            # Persist changes to S3
            self.persist()
            
            return True
            
        except Exception as e:
            raise StorageException(f"Failed to delete vectors: {str(e)}")
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        return self.documents.get(doc_id)
    
    def persist(self) -> bool:
        """
        Persist the vector store to S3.
        
        Returns:
            True if persistence was successful
        """
        try:
            # Save FAISS index
            if self.index is not None:
                index_buffer = BytesIO()
                faiss.write_index(self.index, faiss.BufferedIOWriter(faiss.PyCallbackIOWriter(index_buffer.write)))
                index_buffer.seek(0)
                self._upload_to_s3(self.index_key, index_buffer.read(), "application/octet-stream")
            
            # Save metadata
            metadata = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
                "next_index": self.next_index,
                "id_to_index": self.id_to_index,
                "index_to_id": {str(k): v for k, v in self.index_to_id.items()}  # JSON serializable
            }
            
            metadata_json = json.dumps(metadata, indent=2)
            self._upload_to_s3(self.metadata_key, metadata_json.encode('utf-8'), "application/json")
            
            # Save documents
            documents_buffer = BytesIO()
            pickle.dump(self.documents, documents_buffer)
            documents_buffer.seek(0)
            self._upload_to_s3(self.documents_key, documents_buffer.read(), "application/octet-stream")
            
            return True
            
        except Exception as e:
            raise S3Exception(f"Failed to persist vector store to S3: {str(e)}")
    
    def load(self) -> bool:
        """
        Load the vector store from S3.
        
        Returns:
            True if loading was successful
        """
        try:
            # Load FAISS index
            if self._key_exists_in_s3(self.index_key):
                index_data = self._download_from_s3(self.index_key)
                index_buffer = BytesIO(index_data)
                self.index = faiss.read_index(faiss.BufferedIOReader(faiss.PyCallbackIOReader(index_buffer.read)))
            
            # Load metadata
            if self._key_exists_in_s3(self.metadata_key):
                metadata_data = self._download_from_s3(self.metadata_key)
                metadata = json.loads(metadata_data.decode('utf-8'))
                
                self.dimension = metadata.get("dimension")
                self.next_index = metadata.get("next_index", 0)
                self.id_to_index = metadata.get("id_to_index", {})
                self.index_to_id = {int(k): v for k, v in metadata.get("index_to_id", {}).items()}
            
            # Load documents
            if self._key_exists_in_s3(self.documents_key):
                documents_data = self._download_from_s3(self.documents_key)
                documents_buffer = BytesIO(documents_data)
                self.documents = pickle.load(documents_buffer)
            
            return True
            
        except Exception as e:
            # Don't raise exception on load failure - just return False
            # This allows for graceful handling of missing or corrupted files
            return False
    
    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the store.
        
        Returns:
            Number of vectors stored
        """
        return len(self.documents)
    
    def health_check(self) -> bool:
        """
        Check if the vector store is healthy and accessible.
        
        Returns:
            True if the store is healthy
        """
        try:
            # Check S3 connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            # Check if we can perform basic S3 operations
            test_key = f"{self.s3_prefix}.health_check"
            test_data = b"health_check"
            
            # Try to upload and delete a test object
            self._upload_to_s3(test_key, test_data)
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=test_key)
            
            # Check if index is in a valid state
            if self.index is not None and self.get_vector_count() > 0:
                # Try a search with a dummy vector
                dummy_vector = [0.0] * self.dimension
                self.search_vectors(dummy_vector, k=1)
            
            return True
            
        except Exception:
            return False
    
    def get_s3_info(self) -> Dict[str, Any]:
        """
        Get information about S3 storage.
        
        Returns:
            Dictionary containing S3 storage information
        """
        info = {
            "bucket_name": self.bucket_name,
            "s3_prefix": self.s3_prefix,
            "aws_region": self.aws_region,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "vector_count": self.get_vector_count(),
            "next_index": self.next_index,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }
        
        # Add S3 object information if available
        try:
            for key_name, s3_key in [
                ("index_key", self.index_key),
                ("metadata_key", self.metadata_key),
                ("documents_key", self.documents_key)
            ]:
                if self._key_exists_in_s3(s3_key):
                    response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                    info[f"{key_name}_size"] = response.get('ContentLength', 0)
                    info[f"{key_name}_last_modified"] = response.get('LastModified')
        except Exception:
            # Don't fail if we can't get object info
            pass
        
        return info
    
    def list_s3_objects(self) -> List[Dict[str, Any]]:
        """
        List all objects in the S3 prefix.
        
        Returns:
            List of S3 object information
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.s3_prefix
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag']
                })
            
            return objects
            
        except Exception as e:
            raise S3Exception(f"Failed to list S3 objects: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the S3 vector store."""
        return (
            f"S3VectorStore(bucket={self.bucket_name}, "
            f"prefix={self.s3_prefix}, "
            f"dimension={self.dimension}, "
            f"count={self.get_vector_count()})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the S3 vector store."""
        return (
            f"S3VectorStore(bucket_name='{self.bucket_name}', "
            f"s3_prefix='{self.s3_prefix}', "
            f"aws_region='{self.aws_region}', "
            f"dimension={self.dimension}, "
            f"index_type='{self.index_type}', "
            f"metric='{self.metric}', "
            f"vector_count={self.get_vector_count()})"
        )