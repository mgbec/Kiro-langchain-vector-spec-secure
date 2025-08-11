"""
Unit tests for LocalVectorStore.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from langchain_vector_db.storage.local import LocalVectorStore
from langchain_vector_db.models.document import Document
from langchain_vector_db.exceptions import StorageException, ConfigurationException


class TestLocalVectorStore:
    """Test cases for LocalVectorStore."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_store"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_with_defaults(self):
        """Test store initialization with default parameters."""
        store = LocalVectorStore(str(self.storage_path))
        
        assert store.storage_path == self.storage_path
        assert store.dimension is None
        assert store.index_type == "flat"
        assert store.metric == "l2"
        assert store.storage_path.exists()
    
    def test_initialization_with_custom_parameters(self):
        """Test store initialization with custom parameters."""
        store = LocalVectorStore(
            str(self.storage_path),
            dimension=384,
            index_type="ivf",
            metric="ip"
        )
        
        assert store.dimension == 384
        assert store.index_type == "ivf"
        assert store.metric == "ip"
    
    def test_invalid_index_type(self):
        """Test initialization with invalid index type."""
        with pytest.raises(ConfigurationException) as exc_info:
            LocalVectorStore(str(self.storage_path), index_type="invalid")
        
        assert "Invalid index_type" in str(exc_info.value)
    
    def test_invalid_metric(self):
        """Test initialization with invalid metric."""
        with pytest.raises(ConfigurationException) as exc_info:
            LocalVectorStore(str(self.storage_path), metric="invalid")
        
        assert "Invalid metric" in str(exc_info.value)
    
    def test_invalid_dimension(self):
        """Test initialization with invalid dimension."""
        with pytest.raises(ConfigurationException) as exc_info:
            LocalVectorStore(str(self.storage_path), dimension=0)
        
        assert "dimension must be greater than 0" in str(exc_info.value)
    
    def test_add_vectors_basic(self):
        """Test basic vector addition."""
        store = LocalVectorStore(str(self.storage_path))
        
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2")
        ]
        
        doc_ids = store.add_vectors(vectors, documents)
        
        assert len(doc_ids) == 2
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert store.get_vector_count() == 2
        assert store.dimension == 3
    
    def test_add_vectors_empty_lists(self):
        """Test adding empty vector lists."""
        store = LocalVectorStore(str(self.storage_path))
        
        doc_ids = store.add_vectors([], [])
        
        assert doc_ids == []
        assert store.get_vector_count() == 0
    
    def test_add_vectors_mismatched_lengths(self):
        """Test adding vectors with mismatched document count."""
        store = LocalVectorStore(str(self.storage_path))
        
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        documents = [Document(page_content="Document 1", doc_id="doc1")]
        
        with pytest.raises(StorageException) as exc_info:
            store.add_vectors(vectors, documents)
        
        assert "must match number of documents" in str(exc_info.value)
    
    def test_add_vectors_dimension_mismatch(self):
        """Test adding vectors with different dimensions."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Add first batch
        vectors1 = [[1.0, 2.0, 3.0]]
        documents1 = [Document(page_content="Document 1", doc_id="doc1")]
        store.add_vectors(vectors1, documents1)
        
        # Try to add vectors with different dimension
        vectors2 = [[1.0, 2.0]]  # Different dimension
        documents2 = [Document(page_content="Document 2", doc_id="doc2")]
        
        with pytest.raises(StorageException) as exc_info:
            store.add_vectors(vectors2, documents2)
        
        assert "Vector dimension mismatch" in str(exc_info.value)
    
    def test_search_vectors_basic(self):
        """Test basic vector search."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Add some vectors
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2"),
            Document(page_content="Document 3", doc_id="doc3")
        ]
        store.add_vectors(vectors, documents)
        
        # Search with a query vector
        query_vector = [1.0, 0.0, 0.0]
        results = store.search_vectors(query_vector, k=2)
        
        assert len(results) == 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        assert all(isinstance(result[0], Document) for result in results)
        assert all(isinstance(result[1], (int, float)) for result in results)
    
    def test_search_vectors_empty_store(self):
        """Test searching in empty store."""
        store = LocalVectorStore(str(self.storage_path))
        
        query_vector = [1.0, 2.0, 3.0]
        results = store.search_vectors(query_vector, k=5)
        
        assert results == []
    
    def test_search_vectors_dimension_mismatch(self):
        """Test searching with wrong dimension."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Add vectors
        vectors = [[1.0, 2.0, 3.0]]
        documents = [Document(page_content="Document 1", doc_id="doc1")]
        store.add_vectors(vectors, documents)
        
        # Search with wrong dimension
        query_vector = [1.0, 2.0]  # Wrong dimension
        
        with pytest.raises(StorageException) as exc_info:
            store.search_vectors(query_vector, k=1)
        
        assert "doesn't match index dimension" in str(exc_info.value)
    
    def test_get_document(self):
        """Test retrieving documents by ID."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Add a vector
        vectors = [[1.0, 2.0, 3.0]]
        documents = [Document(page_content="Test document", doc_id="doc1")]
        store.add_vectors(vectors, documents)
        
        # Retrieve the document
        retrieved_doc = store.get_document("doc1")
        
        assert retrieved_doc is not None
        assert retrieved_doc.page_content == "Test document"
        assert retrieved_doc.doc_id == "doc1"
    
    def test_get_nonexistent_document(self):
        """Test retrieving non-existent document."""
        store = LocalVectorStore(str(self.storage_path))
        
        retrieved_doc = store.get_document("nonexistent")
        
        assert retrieved_doc is None
    
    def test_update_vector(self):
        """Test updating an existing vector."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Add a vector
        vectors = [[1.0, 2.0, 3.0]]
        documents = [Document(page_content="Original", doc_id="doc1")]
        store.add_vectors(vectors, documents)
        
        # Update the vector
        new_vector = [4.0, 5.0, 6.0]
        new_document = Document(page_content="Updated", doc_id="doc1")
        
        result = store.update_vector("doc1", new_vector, new_document)
        
        assert result is True
        
        # Verify the document was updated
        retrieved_doc = store.get_document("doc1")
        assert retrieved_doc.page_content == "Updated"
    
    def test_update_nonexistent_vector(self):
        """Test updating non-existent vector."""
        store = LocalVectorStore(str(self.storage_path))
        
        new_vector = [1.0, 2.0, 3.0]
        new_document = Document(page_content="New", doc_id="nonexistent")
        
        result = store.update_vector("nonexistent", new_vector, new_document)
        
        assert result is False
    
    def test_delete_vectors(self):
        """Test deleting vectors."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Add some vectors
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2")
        ]
        store.add_vectors(vectors, documents)
        
        # Delete one vector
        result = store.delete_vectors(["doc1"])
        
        assert result is True
        assert store.get_vector_count() == 1
        assert store.get_document("doc1") is None
        assert store.get_document("doc2") is not None
    
    def test_delete_multiple_vectors(self):
        """Test deleting multiple vectors."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Add some vectors
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2"),
            Document(page_content="Document 3", doc_id="doc3")
        ]
        store.add_vectors(vectors, documents)
        
        # Delete multiple vectors
        result = store.delete_vectors(["doc1", "doc3"])
        
        assert result is True
        assert store.get_vector_count() == 1
        assert store.get_document("doc2") is not None
    
    def test_delete_nonexistent_vectors(self):
        """Test deleting non-existent vectors."""
        store = LocalVectorStore(str(self.storage_path))
        
        result = store.delete_vectors(["nonexistent"])
        
        assert result is True  # Should not fail
    
    def test_persist_and_load(self):
        """Test persisting and loading the store."""
        # Create and populate store
        store1 = LocalVectorStore(str(self.storage_path))
        
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2")
        ]
        store1.add_vectors(vectors, documents)
        
        # Persist the store
        result = store1.persist()
        assert result is True
        
        # Create new store and load
        store2 = LocalVectorStore(str(self.storage_path))
        
        # Verify data was loaded
        assert store2.get_vector_count() == 2
        assert store2.dimension == 3
        assert store2.get_document("doc1") is not None
        assert store2.get_document("doc2") is not None
    
    def test_load_nonexistent_files(self):
        """Test loading when files don't exist."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Should not raise exception, just return False
        result = store.load()
        assert result is False
    
    def test_get_vector_count(self):
        """Test getting vector count."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Initially empty
        assert store.get_vector_count() == 0
        
        # Add some vectors
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2")
        ]
        store.add_vectors(vectors, documents)
        
        assert store.get_vector_count() == 2
    
    def test_health_check_healthy(self):
        """Test health check when store is healthy."""
        store = LocalVectorStore(str(self.storage_path))
        
        assert store.health_check() is True
    
    def test_health_check_with_data(self):
        """Test health check with data in store."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Add some data
        vectors = [[1.0, 2.0, 3.0]]
        documents = [Document(page_content="Document 1", doc_id="doc1")]
        store.add_vectors(vectors, documents)
        
        assert store.health_check() is True
    
    def test_rebuild_index(self):
        """Test rebuilding the index."""
        store = LocalVectorStore(str(self.storage_path))
        
        # Add vectors with embeddings
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1", embedding=[1.0, 2.0, 3.0]),
            Document(page_content="Document 2", doc_id="doc2", embedding=[4.0, 5.0, 6.0])
        ]
        store.add_vectors(vectors, documents)
        
        # Rebuild index
        result = store.rebuild_index()
        
        assert result is True
        assert store.get_vector_count() == 2
    
    def test_rebuild_index_empty_store(self):
        """Test rebuilding index on empty store."""
        store = LocalVectorStore(str(self.storage_path))
        
        result = store.rebuild_index()
        
        assert result is True
    
    def test_get_storage_info(self):
        """Test getting storage information."""
        store = LocalVectorStore(str(self.storage_path), dimension=384)
        
        info = store.get_storage_info()
        
        assert "storage_path" in info
        assert "dimension" in info
        assert "index_type" in info
        assert "metric" in info
        assert "vector_count" in info
        assert info["dimension"] == 384
        assert info["index_type"] == "flat"
        assert info["metric"] == "l2"
    
    def test_string_representations(self):
        """Test string representations of the store."""
        store = LocalVectorStore(str(self.storage_path), dimension=384)
        
        str_repr = str(store)
        assert "LocalVectorStore" in str_repr
        assert str(self.storage_path) in str_repr
        assert "384" in str_repr
        
        repr_str = repr(store)
        assert "LocalVectorStore" in repr_str
        assert "flat" in repr_str
        assert "l2" in repr_str


class TestLocalVectorStoreIndexTypes:
    """Test different FAISS index types."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_store"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_flat_index(self):
        """Test flat index type."""
        store = LocalVectorStore(str(self.storage_path), index_type="flat")
        
        vectors = [[1.0, 2.0, 3.0]]
        documents = [Document(page_content="Document 1", doc_id="doc1")]
        store.add_vectors(vectors, documents)
        
        assert store.index is not None
        assert store.get_vector_count() == 1
    
    def test_ivf_index_small_dataset(self):
        """Test IVF index with small dataset (should fall back to flat)."""
        store = LocalVectorStore(str(self.storage_path), index_type="ivf")
        
        # Add small number of vectors (less than training threshold)
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2")
        ]
        store.add_vectors(vectors, documents)
        
        assert store.get_vector_count() == 2
    
    def test_hnsw_index(self):
        """Test HNSW index type."""
        store = LocalVectorStore(str(self.storage_path), index_type="hnsw")
        
        vectors = [[1.0, 2.0, 3.0]]
        documents = [Document(page_content="Document 1", doc_id="doc1")]
        store.add_vectors(vectors, documents)
        
        assert store.index is not None
        assert store.get_vector_count() == 1


class TestLocalVectorStoreMetrics:
    """Test different distance metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_store"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_l2_metric(self):
        """Test L2 (Euclidean) distance metric."""
        store = LocalVectorStore(str(self.storage_path), metric="l2")
        
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2")
        ]
        store.add_vectors(vectors, documents)
        
        # Search should work
        results = store.search_vectors([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
    
    def test_ip_metric(self):
        """Test inner product metric."""
        store = LocalVectorStore(str(self.storage_path), metric="ip")
        
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2")
        ]
        store.add_vectors(vectors, documents)
        
        # Search should work
        results = store.search_vectors([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__])