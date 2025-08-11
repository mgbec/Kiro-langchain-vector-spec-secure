"""
Local vector store implementation using FAISS for efficient similarity search.
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is required for LocalVectorStore. Install with: pip install faiss-cpu"
    )

from .interface import VectorStoreInterface
from ..models.document import Document
from ..models.observability import create_log_context
from ..exceptions import StorageException, ConfigurationException


class LocalVectorStore(VectorStoreInterface):
    """Local vector store implementation using FAISS for similarity search."""
    
    def __init__(
        self,
        storage_path: str,
        dimension: Optional[int] = None,
        index_type: str = "flat",
        metric: str = "l2",
        observability_manager=None
    ):
        """
        Initialize the local vector store.
        
        Args:
            storage_path: Path to store the vector index and metadata
            dimension: Dimension of the vectors (auto-detected if None)
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            metric: Distance metric ("l2", "ip" for inner product)
            observability_manager: Optional observability manager for logging and metrics
        """
        self.storage_path = Path(storage_path)
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.observability_manager = observability_manager
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Set up observability context
        if self.observability_manager:
            log_context = create_log_context(
                component="LocalVectorStore",
                operation="initialization"
            )
            self.observability_manager.set_log_context(log_context)
            self.observability_manager.log_event("INFO", f"Initializing LocalVectorStore at {storage_path}")
        
        # File paths
        self.index_file = self.storage_path / "faiss_index.bin"
        self.metadata_file = self.storage_path / "metadata.json"
        self.documents_file = self.storage_path / "documents.pkl"
        
        # Initialize data structures
        self.index: Optional[faiss.Index] = None
        self.documents: Dict[str, Document] = {}
        self.id_to_index: Dict[str, int] = {}  # Map doc_id to FAISS index
        self.index_to_id: Dict[int, str] = {}  # Map FAISS index to doc_id
        self.next_index = 0
        
        # Validate parameters
        self._validate_parameters()
        
        # Try to load existing data
        self.load()
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
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
        
        # Observability logging
        if self.observability_manager:
            self.observability_manager.log_event("DEBUG", f"Adding {len(vectors)} vectors to local store")
            self.observability_manager.record_metric("local_vectors_added", len(vectors))
        
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
            
            # Add vectors to FAISS index with performance monitoring
            if self.observability_manager:
                with self.observability_manager.trace_distributed_operation(
                    "faiss_index_add",
                    attributes={"vector_count": len(vectors), "dimension": vector_array.shape[1]}
                ):
                    start_index = self.next_index
                    self.index.add(vector_array)
            else:
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
            
            if self.observability_manager:
                self.observability_manager.log_event("INFO", f"Successfully added {len(doc_ids)} vectors to local store")
                self.observability_manager.record_business_metric(
                    "local_store_total_vectors",
                    self.next_index,
                    "Total vectors in local store",
                    "count"
                )
            
            return doc_ids
            
        except Exception as e:
            if self.observability_manager:
                self.observability_manager.log_event("ERROR", f"Failed to add vectors: {str(e)}", exception=e)
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
        
        # Observability logging
        if self.observability_manager:
            self.observability_manager.log_event("DEBUG", f"Searching local store for {k} similar vectors")
            self.observability_manager.record_metric("local_searches_performed", 1)
        
        try:
            # Convert query to numpy array
            query_array = np.array([query_vector], dtype=np.float32)
            
            # Validate dimension
            if query_array.shape[1] != self.dimension:
                raise StorageException(
                    f"Query vector dimension ({query_array.shape[1]}) "
                    f"doesn't match index dimension ({self.dimension})"
                )
            
            # Search in FAISS index with performance monitoring
            k = min(k, self.get_vector_count())  # Don't search for more than available
            
            if self.observability_manager:
                with self.observability_manager.trace_distributed_operation(
                    "faiss_search",
                    attributes={"k": k, "total_vectors": self.get_vector_count()}
                ):
                    distances, indices = self.index.search(query_array, k)
            else:
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
            # Get the FAISS index for this document
            faiss_idx = self.id_to_index[doc_id]
            
            # FAISS doesn't support direct updates, so we need to rebuild
            # For now, we'll mark this as a limitation and suggest delete + add
            # In a production system, you might want to implement a more sophisticated approach
            
            # Update the document in our storage
            self.documents[doc_id] = document
            
            # Note: The vector in FAISS index is not updated here
            # This is a limitation of the current implementation
            # A full implementation would need to rebuild the index or use a different approach
            
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
        Persist the vector store to storage.
        
        Returns:
            True if persistence was successful
        """
        try:
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            metadata = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
                "next_index": self.next_index,
                "id_to_index": self.id_to_index,
                "index_to_id": {str(k): v for k, v in self.index_to_id.items()}  # JSON serializable
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save documents
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            return True
            
        except Exception as e:
            raise StorageException(f"Failed to persist vector store: {str(e)}")
    
    def load(self) -> bool:
        """
        Load the vector store from storage.
        
        Returns:
            True if loading was successful
        """
        try:
            # Load FAISS index
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.dimension = metadata.get("dimension")
                self.next_index = metadata.get("next_index", 0)
                self.id_to_index = metadata.get("id_to_index", {})
                self.index_to_id = {int(k): v for k, v in metadata.get("index_to_id", {}).items()}
            
            # Load documents
            if self.documents_file.exists():
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
            
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
            # Check if storage path is accessible
            if not self.storage_path.exists():
                return False
            
            # Check if we can write to the storage path
            test_file = self.storage_path / ".health_check"
            test_file.write_text("test")
            test_file.unlink()
            
            # Check if index is in a valid state
            if self.index is not None:
                # Try a simple operation
                if self.get_vector_count() > 0:
                    # Try a search with a dummy vector
                    dummy_vector = [0.0] * self.dimension
                    self.search_vectors(dummy_vector, k=1)
            
            return True
            
        except Exception:
            return False
    
    def rebuild_index(self) -> bool:
        """
        Rebuild the FAISS index from stored documents.
        This can be useful after many deletions or updates.
        
        Returns:
            True if rebuild was successful
        """
        try:
            if not self.documents:
                return True
            
            # Extract vectors from documents (if they have embeddings)
            vectors = []
            docs_with_embeddings = []
            
            for doc in self.documents.values():
                if doc.embedding:
                    vectors.append(doc.embedding)
                    docs_with_embeddings.append(doc)
            
            if not vectors:
                return True
            
            # Create new index
            vector_array = np.array(vectors, dtype=np.float32)
            self.index = self._create_index(vector_array.shape[1])
            
            # Train if necessary
            if self.index_type == "ivf":
                self.index.train(vector_array)
            
            # Add vectors
            self.index.add(vector_array)
            
            # Rebuild mappings
            self.id_to_index = {}
            self.index_to_id = {}
            
            for i, doc in enumerate(docs_with_embeddings):
                self.id_to_index[doc.doc_id] = i
                self.index_to_id[i] = doc.doc_id
            
            self.next_index = len(docs_with_embeddings)
            
            return True
            
        except Exception as e:
            raise StorageException(f"Failed to rebuild index: {str(e)}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage.
        
        Returns:
            Dictionary containing storage information
        """
        info = {
            "storage_path": str(self.storage_path),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "vector_count": self.get_vector_count(),
            "next_index": self.next_index,
        }
        
        # Add file sizes if files exist
        if self.index_file.exists():
            info["index_file_size"] = self.index_file.stat().st_size
        
        if self.metadata_file.exists():
            info["metadata_file_size"] = self.metadata_file.stat().st_size
        
        if self.documents_file.exists():
            info["documents_file_size"] = self.documents_file.stat().st_size
        
        return info
    
    def __str__(self) -> str:
        """String representation of the local vector store."""
        return (
            f"LocalVectorStore(path={self.storage_path}, "
            f"dimension={self.dimension}, "
            f"count={self.get_vector_count()})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the local vector store."""
        return (
            f"LocalVectorStore(storage_path='{self.storage_path}', "
            f"dimension={self.dimension}, "
            f"index_type='{self.index_type}', "
            f"metric='{self.metric}', "
            f"vector_count={self.get_vector_count()})"
        )