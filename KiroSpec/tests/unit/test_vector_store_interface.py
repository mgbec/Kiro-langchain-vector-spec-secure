"""
Unit tests for VectorStoreInterface.
"""

import pytest
from abc import ABC
from typing import List, Tuple, Optional
from langchain_vector_db.storage.interface import VectorStoreInterface
from langchain_vector_db.models.document import Document


class ConcreteVectorStore(VectorStoreInterface):
    """Concrete implementation of VectorStoreInterface for testing."""
    
    def __init__(self):
        self.vectors = {}
        self.documents = {}
        self.next_id = 1
    
    def add_vectors(self, vectors: List[List[float]], documents: List[Document]) -> List[str]:
        doc_ids = []
        for vector, document in zip(vectors, documents):
            doc_id = document.doc_id or str(self.next_id)
            self.next_id += 1
            
            self.vectors[doc_id] = vector
            self.documents[doc_id] = document
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def search_vectors(self, query_vector: List[float], k: int = 4) -> List[Tuple[Document, float]]:
        # Simple similarity calculation for testing
        results = []
        for doc_id, vector in self.vectors.items():
            # Calculate dot product as similarity
            similarity = sum(a * b for a, b in zip(query_vector, vector))
            results.append((self.documents[doc_id], similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def update_vector(self, doc_id: str, vector: List[float], document: Document) -> bool:
        if doc_id in self.vectors:
            self.vectors[doc_id] = vector
            self.documents[doc_id] = document
            return True
        return False
    
    def delete_vectors(self, doc_ids: List[str]) -> bool:
        try:
            for doc_id in doc_ids:
                if doc_id in self.vectors:
                    del self.vectors[doc_id]
                    del self.documents[doc_id]
            return True
        except Exception:
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)
    
    def persist(self) -> bool:
        return True  # Mock implementation
    
    def load(self) -> bool:
        return True  # Mock implementation
    
    def get_vector_count(self) -> int:
        return len(self.vectors)
    
    def health_check(self) -> bool:
        return True  # Mock implementation


class TestVectorStoreInterface:
    """Test cases for VectorStoreInterface."""
    
    def test_interface_is_abstract(self):
        """Test that VectorStoreInterface is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            VectorStoreInterface()
    
    def test_interface_inheritance(self):
        """Test that VectorStoreInterface inherits from ABC."""
        assert issubclass(VectorStoreInterface, ABC)
    
    def test_concrete_implementation(self):
        """Test that concrete implementation can be instantiated."""
        store = ConcreteVectorStore()
        assert isinstance(store, VectorStoreInterface)
    
    def test_add_vectors(self):
        """Test adding vectors to the store."""
        store = ConcreteVectorStore()
        
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
    
    def test_search_vectors(self):
        """Test searching for similar vectors."""
        store = ConcreteVectorStore()
        
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
    
    def test_update_vector(self):
        """Test updating an existing vector."""
        store = ConcreteVectorStore()
        
        # Add a vector
        vectors = [[1.0, 2.0, 3.0]]
        documents = [Document(page_content="Original", doc_id="doc1")]
        store.add_vectors(vectors, documents)
        
        # Update the vector
        new_vector = [4.0, 5.0, 6.0]
        new_document = Document(page_content="Updated", doc_id="doc1")
        
        result = store.update_vector("doc1", new_vector, new_document)
        
        assert result is True
        
        # Verify the update
        retrieved_doc = store.get_document("doc1")
        assert retrieved_doc.page_content == "Updated"
    
    def test_update_nonexistent_vector(self):
        """Test updating a non-existent vector."""
        store = ConcreteVectorStore()
        
        new_vector = [1.0, 2.0, 3.0]
        new_document = Document(page_content="New", doc_id="nonexistent")
        
        result = store.update_vector("nonexistent", new_vector, new_document)
        
        assert result is False
    
    def test_delete_vectors(self):
        """Test deleting vectors."""
        store = ConcreteVectorStore()
        
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
        store = ConcreteVectorStore()
        
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
        assert store.get_document("doc1") is None
        assert store.get_document("doc2") is not None
        assert store.get_document("doc3") is None
    
    def test_get_document(self):
        """Test retrieving a document by ID."""
        store = ConcreteVectorStore()
        
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
        """Test retrieving a non-existent document."""
        store = ConcreteVectorStore()
        
        retrieved_doc = store.get_document("nonexistent")
        
        assert retrieved_doc is None
    
    def test_persist_and_load(self):
        """Test persist and load operations."""
        store = ConcreteVectorStore()
        
        # Test persist
        result = store.persist()
        assert isinstance(result, bool)
        
        # Test load
        result = store.load()
        assert isinstance(result, bool)
    
    def test_get_vector_count(self):
        """Test getting vector count."""
        store = ConcreteVectorStore()
        
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
    
    def test_health_check(self):
        """Test health check operation."""
        store = ConcreteVectorStore()
        
        result = store.health_check()
        assert isinstance(result, bool)
    
    def test_interface_method_signatures(self):
        """Test that all interface methods have correct signatures."""
        store = ConcreteVectorStore()
        
        # Test that all methods exist and are callable
        assert callable(store.add_vectors)
        assert callable(store.search_vectors)
        assert callable(store.update_vector)
        assert callable(store.delete_vectors)
        assert callable(store.get_document)
        assert callable(store.persist)
        assert callable(store.load)
        assert callable(store.get_vector_count)
        assert callable(store.health_check)


class IncompleteVectorStore(VectorStoreInterface):
    """Incomplete implementation to test abstract method enforcement."""
    
    def add_vectors(self, vectors: List[List[float]], documents: List[Document]) -> List[str]:
        return []
    
    # Missing other abstract methods


class TestAbstractMethodEnforcement:
    """Test that abstract methods are properly enforced."""
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementation cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            IncompleteVectorStore()
        
        # Should mention missing abstract methods
        error_message = str(exc_info.value)
        assert "abstract" in error_message.lower()


if __name__ == "__main__":
    pytest.main([__file__])