"""
Abstract base class for vector storage backends.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from ..models.document import Document


class VectorStoreInterface(ABC):
    """Abstract interface for vector storage backends."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def delete_vectors(self, doc_ids: List[str]) -> bool:
        """
        Delete vectors by document IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    def persist(self) -> bool:
        """
        Persist the vector store to storage.
        
        Returns:
            True if persistence was successful
        """
        pass
    
    @abstractmethod
    def load(self) -> bool:
        """
        Load the vector store from storage.
        
        Returns:
            True if loading was successful
        """
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the store.
        
        Returns:
            Number of vectors stored
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the vector store is healthy and accessible.
        
        Returns:
            True if the store is healthy
        """
        pass