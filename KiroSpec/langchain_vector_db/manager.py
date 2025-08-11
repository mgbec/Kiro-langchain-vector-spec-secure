"""
Main VectorDatabaseManager orchestrator that coordinates all components.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path

from .models.config import VectorDBConfig
from .models.document import Document
from .models.auth import AuthToken
from .models.observability import create_log_context
from .services.embedding import EmbeddingService
from .services.document_processor import DocumentProcessor
from .services.security import SecurityManager
from .services.observability import ObservabilityManager
from .storage.interface import VectorStoreInterface
from .persistence import PersistenceManager
from .factory import (
    VectorStoreFactory,
    EmbeddingServiceFactory,
    DocumentProcessorFactory,
    VectorDatabaseFactory
)
from .exceptions import (
    VectorDBException,
    ConfigurationException,
    StorageException,
    EmbeddingException
)


class VectorDatabaseManager:
    """
    Main orchestrator class that coordinates document processing, embedding generation,
    and vector storage operations.
    """
    
    def __init__(self, config: VectorDBConfig, auto_persist: bool = True):
        """
        Initialize the vector database manager.
        
        Args:
            config: Configuration for the vector database system
            auto_persist: Whether to enable automatic persistence
        """
        self.config = config
        
        # Initialize components
        self.embedding_service = self._create_embedding_service()
        self.document_processor = self._create_document_processor()
        self.vector_store = self._create_vector_store()
        
        # Initialize security manager if security is enabled
        self.security_manager = None
        if self.config.security and self.config.security.auth_enabled:
            self.security_manager = SecurityManager(self.config.security)
        
        # Initialize observability manager
        self.observability_manager = None
        if self.config.observability:
            self.observability_manager = ObservabilityManager(self.config.observability)
            
            # Register health checks
            self._register_health_checks()
        
        # Initialize persistence manager
        self.persistence_manager = PersistenceManager(
            vector_store=self.vector_store,
            config=self.config,
            auto_persist=auto_persist
        )
        
        # Cache for embedding dimension
        self._embedding_dimension: Optional[int] = None
    
    def _create_embedding_service(self) -> EmbeddingService:
        """Create the embedding service based on configuration."""
        return EmbeddingServiceFactory.create_embedding_service(self.config)
    
    def _create_document_processor(self) -> DocumentProcessor:
        """Create the document processor based on configuration."""
        return DocumentProcessorFactory.create_document_processor(self.config)
    
    def _create_vector_store(self) -> VectorStoreInterface:
        """Create the vector store based on configuration."""
        return VectorStoreFactory.create_vector_store(self.config)
    
    def _register_health_checks(self) -> None:
        """Register health checks for all components."""
        if not self.observability_manager:
            return
        
        # Vector store health check
        def vector_store_health():
            try:
                # Simple health check - try to get document count
                return True, "Vector store is healthy", {}
            except Exception as e:
                return False, f"Vector store error: {str(e)}", {"error": str(e)}
        
        # Embedding service health check
        def embedding_service_health():
            try:
                # Check if embedding service can generate embeddings
                test_embedding = self.embedding_service.embed_query("health check")
                return True, "Embedding service is healthy", {"dimension": len(test_embedding)}
            except Exception as e:
                return False, f"Embedding service error: {str(e)}", {"error": str(e)}
        
        # Security manager health check
        def security_health():
            if self.security_manager:
                try:
                    healthy = self.security_manager.health_check()
                    return healthy, "Security manager is healthy" if healthy else "Security manager is unhealthy", {}
                except Exception as e:
                    return False, f"Security manager error: {str(e)}", {"error": str(e)}
            return True, "Security not enabled", {}
        
        self.observability_manager.register_health_check("vector_store", vector_store_health)
        self.observability_manager.register_health_check("embedding_service", embedding_service_health)
        self.observability_manager.register_health_check("security_manager", security_health)
    
    def add_documents(
        self, 
        documents: List[Document],
        generate_embeddings: bool = True,
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> List[str]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of Document objects to add
            generate_embeddings: Whether to generate embeddings for documents
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
            
        Returns:
            List of document IDs that were added
            
        Raises:
            VectorDBException: If document addition fails
        """
        if not documents:
            return []
        
        # Set up observability context
        correlation_id = None
        if self.observability_manager:
            log_context = create_log_context(
                user_id=user_id or (auth_token.user_id if auth_token else "anonymous"),
                operation="add_documents",
                component="VectorDatabaseManager"
            )
            correlation_id = log_context.correlation_id
            self.observability_manager.set_log_context(log_context)
            self.observability_manager.log_event("INFO", f"Starting document addition for {len(documents)} documents")
        
        # Security checks
        if self.security_manager:
            # Check authentication and authorization
            if auth_token:
                if not self.security_manager.authorize_operation(
                    auth_token, "documents", "create"
                ):
                    if self.observability_manager:
                        self.observability_manager.log_event("ERROR", "Insufficient permissions to add documents")
                    raise VectorDBException("Insufficient permissions to add documents")
            
            # Check rate limits
            effective_user_id = user_id or (auth_token.user_id if auth_token else "anonymous")
            self.security_manager.check_rate_limit(
                user_id=effective_user_id,
                operation="add_documents",
                ip_address=ip_address
            )
        
        # Start distributed tracing and performance monitoring
        if self.observability_manager:
            with self.observability_manager.trace_distributed_operation(
                "add_documents",
                attributes={
                    "document_count": len(documents),
                    "generate_embeddings": generate_embeddings,
                    "operation.type": "document_ingestion"
                }
            ) as span:
                return self._execute_add_documents(
                    documents, generate_embeddings, embeddings_generated=0, span=span
                )
        else:
            return self._execute_add_documents(documents, generate_embeddings, embeddings_generated=0)
    
    def _execute_add_documents(
        self,
        documents: List[Document],
        generate_embeddings: bool,
        embeddings_generated: int = 0,
        span: Optional[Any] = None
    ) -> List[str]:
        """Execute the document addition logic."""
        try:
            processed_documents = []
            
            if self.observability_manager:
                self.observability_manager.log_event("DEBUG", "Processing documents for embedding generation")
            
            for document in documents:
                # Generate embedding if needed and requested
                if generate_embeddings and document.embedding is None:
                    if self.observability_manager:
                        self.observability_manager.log_event("DEBUG", f"Generating embedding for document: {document.doc_id}")
                    
                    # Time the embedding generation with distributed tracing
                    if self.observability_manager:
                        with self.observability_manager.trace_distributed_operation(
                            "embedding_generation",
                            attributes={
                                "document_id": document.doc_id,
                                "content_length": len(document.page_content),
                                "embedding_model": self.config.embedding_model
                            }
                        ) as embedding_span:
                            embedding = self.embedding_service.embed_query(document.page_content)
                            if embedding_span:
                                embedding_span.set_attribute("embedding_dimension", len(embedding))
                    else:
                        embedding = self.embedding_service.embed_query(document.page_content)
                    
                    document.embedding = embedding
                    embeddings_generated += 1
                    
                    # Record embedding generation metric
                    if self.observability_manager:
                        self.observability_manager.metrics.record_embedding_generated()
                
                processed_documents.append(document)
            
            if self.observability_manager:
                self.observability_manager.log_event("INFO", f"Generated {embeddings_generated} embeddings")
            
            # Extract embeddings for vector store
            embeddings = []
            docs_with_embeddings = []
            
            for doc in processed_documents:
                if doc.embedding is not None:
                    embeddings.append(doc.embedding)
                    docs_with_embeddings.append(doc)
            
            if not embeddings:
                error_msg = "No embeddings available for documents"
                if self.observability_manager:
                    self.observability_manager.log_event("ERROR", error_msg)
                    if trace_context:
                        trace_context.add_log(error_msg, "error")
                raise VectorDBException(error_msg)
            
            if self.observability_manager:
                self.observability_manager.log_event("DEBUG", f"Adding {len(embeddings)} vectors to store")
            
            # Add to vector store with distributed tracing
            if self.observability_manager:
                with self.observability_manager.trace_distributed_operation(
                    "vector_store_add",
                    attributes={
                        "vector_count": len(embeddings),
                        "storage_type": self.config.storage_type,
                        "embedding_dimension": len(embeddings[0]) if embeddings else 0
                    }
                ) as store_span:
                    doc_ids = self.vector_store.add_vectors(embeddings, docs_with_embeddings)
                    if store_span:
                        store_span.set_attribute("documents_stored", len(doc_ids))
            else:
                doc_ids = self.vector_store.add_vectors(embeddings, docs_with_embeddings)
            
            # Record document indexing metrics
            if self.observability_manager:
                for _ in doc_ids:
                    self.observability_manager.metrics.record_document_indexed()
                
                self.observability_manager.log_event("INFO", f"Successfully added {len(doc_ids)} documents")
                self.observability_manager.record_metric("documents_added", len(doc_ids))
                self.observability_manager.record_business_metric(
                    "documents_added_batch",
                    len(doc_ids),
                    "Documents added in this batch",
                    "count"
                )
                
                if span:
                    self.observability_manager.set_span_attribute("documents_added", len(doc_ids), span)
                    self.observability_manager.set_span_attribute("embeddings_generated", embeddings_generated, span)
                    self.observability_manager.add_span_event(
                        "documents_processed",
                        {
                            "total_documents": len(doc_ids),
                            "embeddings_generated": embeddings_generated,
                            "processing_complete": True
                        },
                        span
                    )
            
            # Mark changes as pending for persistence
            self.persistence_manager.mark_changes_pending()
            
            return doc_ids
            
        except Exception as e:
            # Log error and record in span
            if self.observability_manager:
                self.observability_manager.log_event("ERROR", f"Failed to add documents: {str(e)}", exception=e)
                if span:
                    self.observability_manager.record_span_exception(e, span)
            
            if isinstance(e, (VectorDBException, EmbeddingException, StorageException)):
                raise
            raise VectorDBException(f"Failed to add documents: {str(e)}")
        
        finally:
            # Clear observability context
            if self.observability_manager:
                self.observability_manager.clear_log_context()
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> List[str]:
        """
        Add text strings to the vector database.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries for each text
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
            
        Returns:
            List of document IDs that were added
            
        Raises:
            VectorDBException: If text addition fails
        """
        if not texts:
            return []
        
        # Security checks
        if self.security_manager:
            if auth_token:
                if not self.security_manager.authorize_operation(
                    auth_token, "documents", "create"
                ):
                    raise VectorDBException("Insufficient permissions to add texts")
            
            effective_user_id = user_id or (auth_token.user_id if auth_token else "anonymous")
            self.security_manager.check_rate_limit(
                user_id=effective_user_id,
                operation="add_texts",
                ip_address=ip_address
            )
        
        try:
            # Process texts into documents
            documents = self.document_processor.process_texts(texts, metadatas)
            
            # Add documents to the database with security context
            return self.add_documents(
                documents, 
                generate_embeddings=True,
                auth_token=auth_token,
                user_id=user_id,
                ip_address=ip_address
            )
            
        except Exception as e:
            if isinstance(e, VectorDBException):
                raise
            raise VectorDBException(f"Failed to add texts: {str(e)}")
    
    def add_documents_from_files(
        self,
        file_paths: List[Union[str, Path]],
        metadata_override: Optional[Dict[str, Any]] = None,
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> List[str]:
        """
        Add documents from files to the vector database.
        
        Args:
            file_paths: List of file paths to process
            metadata_override: Optional metadata to add to all documents
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
            
        Returns:
            List of document IDs that were added
            
        Raises:
            VectorDBException: If file processing fails
        """
        if not file_paths:
            return []
        
        # Security checks
        if self.security_manager:
            if auth_token:
                if not self.security_manager.authorize_operation(
                    auth_token, "documents", "create"
                ):
                    raise VectorDBException("Insufficient permissions to add documents from files")
            
            effective_user_id = user_id or (auth_token.user_id if auth_token else "anonymous")
            self.security_manager.check_rate_limit(
                user_id=effective_user_id,
                operation="add_documents_from_files",
                ip_address=ip_address
            )
        
        try:
            # Process files into documents
            documents = self.document_processor.process_documents(file_paths, metadata_override)
            
            # Add documents to the database with security context
            return self.add_documents(
                documents, 
                generate_embeddings=True,
                auth_token=auth_token,
                user_id=user_id,
                ip_address=ip_address
            )
            
        except Exception as e:
            if isinstance(e, VectorDBException):
                raise
            raise VectorDBException(f"Failed to add documents from files: {str(e)}")
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> List[Document]:
        """
        Perform similarity search for documents.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters (not implemented yet)
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
            
        Returns:
            List of similar documents
            
        Raises:
            VectorDBException: If search fails
        """
        # Set up observability context
        if self.observability_manager:
            log_context = create_log_context(
                user_id=user_id or (auth_token.user_id if auth_token else "anonymous"),
                operation="similarity_search",
                component="VectorDatabaseManager"
            )
            self.observability_manager.set_log_context(log_context)
            self.observability_manager.log_event("INFO", f"Starting similarity search with k={k}")
        
        # Security checks
        if self.security_manager:
            # Check authentication and authorization
            if auth_token:
                if not self.security_manager.authorize_operation(
                    auth_token, "search", "query"
                ):
                    if self.observability_manager:
                        self.observability_manager.log_event("ERROR", "Insufficient permissions to perform search")
                    raise VectorDBException("Insufficient permissions to perform search")
            
            # Check rate limits
            effective_user_id = user_id or (auth_token.user_id if auth_token else "anonymous")
            self.security_manager.check_rate_limit(
                user_id=effective_user_id,
                operation="similarity_search",
                ip_address=ip_address
            )
        
        # Start distributed tracing and performance monitoring
        if self.observability_manager:
            with self.observability_manager.trace_distributed_operation(
                "similarity_search",
                attributes={
                    "k": k,
                    "query_length": len(query),
                    "has_metadata_filter": filter_metadata is not None,
                    "operation.type": "vector_search"
                }
            ) as span:
                return self._execute_similarity_search(query, k, filter_metadata, span)
        else:
            return self._execute_similarity_search(query, k, filter_metadata)
    
    def _execute_similarity_search(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]],
        span: Optional[Any] = None
    ) -> List[Document]:
        """Execute the similarity search logic."""
        try:
            if self.observability_manager:
                self.observability_manager.log_event("DEBUG", "Generating query embedding")
            
            # Generate query embedding with distributed tracing
            if self.observability_manager:
                with self.observability_manager.trace_distributed_operation(
                    "query_embedding_generation",
                    attributes={
                        "query_length": len(query),
                        "embedding_model": self.config.embedding_model
                    }
                ) as embedding_span:
                    query_embedding = self.embedding_service.embed_query(query)
                    if embedding_span:
                        embedding_span.set_attribute("embedding_dimension", len(query_embedding))
            else:
                query_embedding = self.embedding_service.embed_query(query)
            
            if self.observability_manager:
                self.observability_manager.log_event("DEBUG", f"Searching vector store with embedding dimension: {len(query_embedding)}")
            
            # Search in vector store with distributed tracing
            if self.observability_manager:
                with self.observability_manager.trace_distributed_operation(
                    "vector_search",
                    attributes={
                        "k": k,
                        "embedding_dimension": len(query_embedding),
                        "storage_type": self.config.storage_type
                    }
                ) as search_span:
                    results = self.vector_store.search_vectors(query_embedding, k)
                    if search_span:
                        search_span.set_attribute("raw_results_count", len(results))
            else:
                results = self.vector_store.search_vectors(query_embedding, k)
            
            # Extract documents from results
            documents = [doc for doc, score in results]
            
            # Apply metadata filtering if specified
            if filter_metadata:
                if self.observability_manager:
                    self.observability_manager.log_event("DEBUG", "Applying metadata filtering")
                documents = self._filter_documents_by_metadata(documents, filter_metadata)
            
            # Record search metrics
            if self.observability_manager:
                self.observability_manager.metrics.record_search_performed()
                self.observability_manager.record_metric("search_results_count", len(documents))
                self.observability_manager.record_business_metric(
                    "search_results_returned",
                    len(documents),
                    "Number of results returned from search",
                    "count"
                )
                self.observability_manager.log_event("INFO", f"Search completed, returning {len(documents)} documents")
                
                if span:
                    self.observability_manager.set_span_attribute("results_count", len(documents), span)
                    self.observability_manager.add_span_event(
                        "search_completed",
                        {
                            "results_returned": len(documents),
                            "metadata_filtered": filter_metadata is not None,
                            "search_successful": True
                        },
                        span
                    )
            
            return documents
            
        except Exception as e:
            # Log error and record in span
            if self.observability_manager:
                self.observability_manager.log_event("ERROR", f"Failed to perform similarity search: {str(e)}", exception=e)
                if span:
                    self.observability_manager.record_span_exception(e, span)
            
            if isinstance(e, (EmbeddingException, StorageException)):
                raise
            raise VectorDBException(f"Failed to perform similarity search: {str(e)}")
        
        finally:
            # Clear observability context
            if self.observability_manager:
                self.observability_manager.clear_log_context()
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with similarity scores.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters (not implemented yet)
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
            
        Returns:
            List of (document, similarity_score) tuples
            
        Raises:
            VectorDBException: If search fails
        """
        # Security checks
        if self.security_manager:
            # Check authentication and authorization
            if auth_token:
                if not self.security_manager.authorize_operation(
                    auth_token, "search", "query"
                ):
                    raise VectorDBException("Insufficient permissions to perform search")
            
            # Check rate limits
            effective_user_id = user_id or (auth_token.user_id if auth_token else "anonymous")
            self.security_manager.check_rate_limit(
                user_id=effective_user_id,
                operation="similarity_search",
                ip_address=ip_address
            )
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            
            # Search in vector store
            results = self.vector_store.search_vectors(query_embedding, k)
            
            # Apply metadata filtering if specified
            if filter_metadata:
                filtered_results = []
                for doc, score in results:
                    if self._document_matches_metadata_filter(doc, filter_metadata):
                        filtered_results.append((doc, score))
                results = filtered_results
            
            return results
            
        except Exception as e:
            if isinstance(e, (EmbeddingException, StorageException)):
                raise
            raise VectorDBException(f"Failed to perform similarity search with scores: {str(e)}")
    
    def update_document(
        self,
        doc_id: str,
        document: Document,
        generate_embedding: bool = True,
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id: Document ID to update
            document: Updated document
            generate_embedding: Whether to generate new embedding
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
            
        Returns:
            True if update was successful
            
        Raises:
            VectorDBException: If update fails
        """
        # Security checks
        if self.security_manager:
            if auth_token:
                if not self.security_manager.authorize_operation(
                    auth_token, "documents", "update"
                ):
                    raise VectorDBException("Insufficient permissions to update document")
            
            effective_user_id = user_id or (auth_token.user_id if auth_token else "anonymous")
            self.security_manager.check_rate_limit(
                user_id=effective_user_id,
                operation="update_document",
                ip_address=ip_address
            )
        
        try:
            # Generate embedding if needed and requested
            if generate_embedding and document.embedding is None:
                embedding = self.embedding_service.embed_query(document.page_content)
                document.embedding = embedding
            
            # Update in vector store
            if document.embedding is not None:
                result = self.vector_store.update_vector(doc_id, document.embedding, document)
                if result:
                    self.persistence_manager.mark_changes_pending()
                return result
            else:
                # If no embedding, just update the document metadata
                existing_doc = self.vector_store.get_document(doc_id)
                if existing_doc and existing_doc.embedding:
                    document.embedding = existing_doc.embedding
                    result = self.vector_store.update_vector(doc_id, document.embedding, document)
                    if result:
                        self.persistence_manager.mark_changes_pending()
                    return result
                else:
                    raise VectorDBException(f"Cannot update document {doc_id}: no embedding available")
            
        except Exception as e:
            if isinstance(e, (VectorDBException, EmbeddingException, StorageException)):
                raise
            raise VectorDBException(f"Failed to update document: {str(e)}")
    
    def delete_documents(
        self, 
        doc_ids: List[str],
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
            
        Returns:
            True if deletion was successful
            
        Raises:
            VectorDBException: If deletion fails
        """
        # Security checks
        if self.security_manager:
            if auth_token:
                if not self.security_manager.authorize_operation(
                    auth_token, "documents", "delete"
                ):
                    raise VectorDBException("Insufficient permissions to delete documents")
            
            effective_user_id = user_id or (auth_token.user_id if auth_token else "anonymous")
            self.security_manager.check_rate_limit(
                user_id=effective_user_id,
                operation="delete_documents",
                ip_address=ip_address
            )
        
        try:
            result = self.vector_store.delete_vectors(doc_ids)
            if result:
                self.persistence_manager.mark_changes_pending()
            return result
        except Exception as e:
            if isinstance(e, StorageException):
                raise
            raise VectorDBException(f"Failed to delete documents: {str(e)}")
    
    def get_document(
        self, 
        doc_id: str,
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID to retrieve
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
            
        Returns:
            Document if found, None otherwise
            
        Raises:
            VectorDBException: If retrieval fails
        """
        # Security checks
        if self.security_manager:
            if auth_token:
                if not self.security_manager.authorize_operation(
                    auth_token, "documents", "read"
                ):
                    raise VectorDBException("Insufficient permissions to read document")
            
            effective_user_id = user_id or (auth_token.user_id if auth_token else "anonymous")
            self.security_manager.check_rate_limit(
                user_id=effective_user_id,
                operation="get_document",
                ip_address=ip_address
            )
        
        try:
            return self.vector_store.get_document(doc_id)
        except Exception as e:
            if isinstance(e, StorageException):
                raise
            raise VectorDBException(f"Failed to get document: {str(e)}")
    
    def get_document_metadata(
        self, 
        doc_id: str,
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a document.
        
        Args:
            doc_id: Document ID
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
            
        Returns:
            Document metadata if found, None otherwise
        """
        # Security checks are handled by get_document
        document = self.get_document(doc_id, auth_token, user_id, ip_address)
        if document:
            return document.metadata
        return None
    
    def persist(
        self, 
        force: bool = False,
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Persist the vector database to storage.
        
        Args:
            force: Force persistence even if no changes are pending
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
        
        Returns:
            True if persistence was successful
            
        Raises:
            VectorDBException: If persistence fails
        """
        # Security checks - admin operation
        if self.security_manager:
            if auth_token:
                if not self.security_manager.authorize_operation(
                    auth_token, "system", "manage"
                ):
                    raise VectorDBException("Insufficient permissions to persist database")
            
            effective_user_id = user_id or (auth_token.user_id if auth_token else "anonymous")
            self.security_manager.check_rate_limit(
                user_id=effective_user_id,
                operation="persist",
                ip_address=ip_address
            )
        
        try:
            return self.persistence_manager.persist(force=force)
        except Exception as e:
            if isinstance(e, StorageException):
                raise
            raise VectorDBException(f"Failed to persist database: {str(e)}")
    
    def load(
        self,
        auth_token: Optional[AuthToken] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Load the vector database from storage.
        
        Args:
            auth_token: Authentication token for security checks
            user_id: User ID for rate limiting
            ip_address: IP address for rate limiting
        
        Returns:
            True if loading was successful
            
        Raises:
            VectorDBException: If loading fails
        """
        # Security checks - admin operation
        if self.security_manager:
            if auth_token:
                if not self.security_manager.authorize_operation(
                    auth_token, "system", "manage"
                ):
                    raise VectorDBException("Insufficient permissions to load database")
            
            effective_user_id = user_id or (auth_token.user_id if auth_token else "anonymous")
            self.security_manager.check_rate_limit(
                user_id=effective_user_id,
                operation="load",
                ip_address=ip_address
            )
        
        try:
            return self.persistence_manager.load()
        except Exception as e:
            if isinstance(e, StorageException):
                raise
            raise VectorDBException(f"Failed to load database: {str(e)}")
    
    def force_persist(self) -> bool:
        """
        Force immediate persistence regardless of pending changes.
        
        Returns:
            True if persistence was successful
        """
        return self.persist(force=True)
    
    def get_persistence_info(self) -> Dict[str, Any]:
        """
        Get information about persistence state.
        
        Returns:
            Dictionary with persistence information
        """
        return self.persistence_manager.get_persistence_info()
    
    def get_persistence_stats(self) -> Dict[str, Any]:
        """
        Get persistence statistics.
        
        Returns:
            Dictionary with persistence statistics
        """
        return self.persistence_manager.get_persistence_stats()
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of persisted data.
        
        Returns:
            Dictionary with validation results
        """
        return self.persistence_manager.validate_integrity()
    
    def create_backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the current state.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to the created backup
        """
        return self.persistence_manager.create_backup(backup_path)
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_path: Path to the backup to restore from
            
        Returns:
            True if restore was successful
        """
        return self.persistence_manager.restore_from_backup(backup_path)
    
    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the database.
        
        Returns:
            Number of vectors stored
        """
        return self.vector_store.get_vector_count()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
            
        Raises:
            VectorDBException: If dimension cannot be determined
        """
        if self._embedding_dimension is None:
            try:
                self._embedding_dimension = self.embedding_service.get_embedding_dimension()
            except Exception as e:
                raise VectorDBException(f"Failed to get embedding dimension: {str(e)}")
        
        return self._embedding_dimension
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all components.
        
        Returns:
            Dictionary with health status of each component
        """
        health_status = {}
        
        try:
            health_status["embedding_service"] = self.embedding_service.health_check()
        except Exception:
            health_status["embedding_service"] = False
        
        try:
            health_status["document_processor"] = self.document_processor.health_check()
        except Exception:
            health_status["document_processor"] = False
        
        try:
            health_status["vector_store"] = self.vector_store.health_check()
        except Exception:
            health_status["vector_store"] = False
        
        # Overall health
        health_status["overall"] = all(health_status.values())
        
        return health_status
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dictionary containing system information
        """
        info = {
            "config": self.config.to_dict(),
            "vector_count": self.get_vector_count(),
            "embedding_dimension": None,
            "embedding_service": {},
            "document_processor": {},
            "vector_store": {},
            "persistence": self.get_persistence_info(),
            "health": self.health_check()
        }
        
        # Get embedding dimension safely
        try:
            info["embedding_dimension"] = self.get_embedding_dimension()
        except Exception:
            pass
        
        # Get component information safely
        try:
            info["embedding_service"] = self.embedding_service.get_model_info()
        except Exception:
            pass
        
        try:
            info["document_processor"] = self.document_processor.get_processor_info()
        except Exception:
            pass
        
        try:
            if hasattr(self.vector_store, 'get_storage_info'):
                info["vector_store"] = self.vector_store.get_storage_info()
            elif hasattr(self.vector_store, 'get_s3_info'):
                info["vector_store"] = self.vector_store.get_s3_info()
        except Exception:
            pass
        
        return info
    
    def _filter_documents_by_metadata(
        self,
        documents: List[Document],
        filter_metadata: Dict[str, Any]
    ) -> List[Document]:
        """Filter documents by metadata criteria."""
        filtered_docs = []
        
        for doc in documents:
            if self._document_matches_metadata_filter(doc, filter_metadata):
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _document_matches_metadata_filter(
        self,
        document: Document,
        filter_metadata: Dict[str, Any]
    ) -> bool:
        """Check if a document matches metadata filter criteria."""
        for key, value in filter_metadata.items():
            if key not in document.metadata:
                return False
            
            doc_value = document.metadata[key]
            
            # Handle different comparison types
            if isinstance(value, dict):
                # Support for operators like {"$gt": 10}, {"$in": [1, 2, 3]}
                for operator, operand in value.items():
                    if operator == "$gt" and not (doc_value > operand):
                        return False
                    elif operator == "$gte" and not (doc_value >= operand):
                        return False
                    elif operator == "$lt" and not (doc_value < operand):
                        return False
                    elif operator == "$lte" and not (doc_value <= operand):
                        return False
                    elif operator == "$in" and doc_value not in operand:
                        return False
                    elif operator == "$nin" and doc_value in operand:
                        return False
                    elif operator == "$ne" and doc_value == operand:
                        return False
            else:
                # Direct equality comparison
                if doc_value != value:
                    return False
        
        return True
    
    def validate_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, List[str]]:
        """
        Validate a list of file paths for processing.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dictionary with validation results
        """
        return self.document_processor.validate_files(file_paths)
    
    def get_supported_file_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return self.document_processor.get_supported_extensions()
    
    def __str__(self) -> str:
        """String representation of the vector database manager."""
        return (
            f"VectorDatabaseManager(storage={self.config.storage_type}, "
            f"embedding={self.config.embedding_model}, "
            f"vectors={self.get_vector_count()})"
        )
    
    def switch_backend(self, new_config: VectorDBConfig, migrate_data: bool = False) -> bool:
        """
        Switch to a different storage backend.
        
        Args:
            new_config: New configuration with different storage backend
            migrate_data: Whether to migrate existing data to new backend
            
        Returns:
            True if switch was successful
            
        Raises:
            VectorDBException: If backend switch fails
        """
        try:
            # Validate new configuration
            validation_result = VectorDatabaseFactory.validate_configuration(new_config)
            if not validation_result["overall"]:
                raise ConfigurationException("Invalid new configuration")
            
            # Store current data if migration is requested
            migrated_data = None
            if migrate_data and self.get_vector_count() > 0:
                # Get all documents with their embeddings
                migrated_data = self._export_all_data()
            
            # Create new components
            new_vector_store = VectorStoreFactory.create_vector_store(new_config)
            
            # Update configuration and components
            old_vector_store = self.vector_store
            self.config = new_config
            self.vector_store = new_vector_store
            
            # Migrate data if requested
            if migrate_data and migrated_data:
                self._import_all_data(migrated_data)
            
            # Clean up old vector store if possible
            if hasattr(old_vector_store, 'close'):
                try:
                    old_vector_store.close()
                except Exception:
                    pass
            
            return True
            
        except Exception as e:
            if isinstance(e, (VectorDBException, ConfigurationException)):
                raise
            raise VectorDBException(f"Failed to switch backend: {str(e)}")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration.
        
        Returns:
            Dictionary with validation results and suggestions
        """
        from .factory import ConfigurationValidator
        return ConfigurationValidator.validate_and_suggest(self.config)
    
    def get_available_backends(self) -> List[str]:
        """
        Get list of available storage backends.
        
        Returns:
            List of available backend names
        """
        return VectorStoreFactory.get_available_store_types()
    
    def _export_all_data(self) -> List[Document]:
        """Export all documents with embeddings for migration."""
        # This is a simplified implementation
        # In a production system, you might want to implement this more efficiently
        exported_docs = []
        
        # Note: This is a basic implementation that assumes we can iterate over all documents
        # In practice, you might need to implement this differently based on the storage backend
        try:
            # For now, we'll return an empty list as this would require backend-specific implementation
            # In a full implementation, each backend would need to support exporting all data
            return exported_docs
        except Exception:
            return []
    
    def _import_all_data(self, documents: List[Document]) -> None:
        """Import documents to the new backend."""
        if documents:
            # Extract embeddings and add to new backend
            embeddings = [doc.embedding for doc in documents if doc.embedding]
            docs_with_embeddings = [doc for doc in documents if doc.embedding]
            
            if embeddings and docs_with_embeddings:
                self.vector_store.add_vectors(embeddings, docs_with_embeddings)
    
    @classmethod
    def create_from_config(cls, config: VectorDBConfig) -> "VectorDatabaseManager":
        """
        Create a VectorDatabaseManager instance from configuration.
        
        Args:
            config: Vector database configuration
            
        Returns:
            VectorDatabaseManager instance
            
        Raises:
            ConfigurationException: If configuration is invalid
        """
        # Validate configuration before creating
        validation_result = VectorDatabaseFactory.validate_configuration(config)
        if not validation_result["overall"]:
            issues = []
            for key, is_valid in validation_result.items():
                if not is_valid and key != "overall":
                    issues.append(key)
            raise ConfigurationException(f"Invalid configuration: {issues}")
        
        return cls(config)
    
    @classmethod
    def create_with_validation(cls, config: VectorDBConfig) -> Tuple["VectorDatabaseManager", Dict[str, Any]]:
        """
        Create a VectorDatabaseManager with detailed validation results.
        
        Args:
            config: Vector database configuration
            
        Returns:
            Tuple of (VectorDatabaseManager instance, validation results)
            
        Raises:
            ConfigurationException: If configuration is invalid
        """
        from .factory import ConfigurationValidator
        
        validation_result = ConfigurationValidator.validate_and_suggest(config)
        
        if not validation_result["is_valid"]:
            raise ConfigurationException(
                f"Configuration validation failed: {validation_result['issues']}"
            )
        
        manager = cls(config)
        return manager, validation_result
    
    def __repr__(self) -> str:
        """Detailed string representation of the vector database manager."""
        return (
            f"VectorDatabaseManager(storage_type='{self.config.storage_type}', "
            f"embedding_model='{self.config.embedding_model}', "
            f"storage_path='{self.config.storage_path}', "
            f"vector_count={self.get_vector_count()})"
        )