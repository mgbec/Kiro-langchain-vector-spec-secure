"""
Document processing service for loading, splitting, and preprocessing documents.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredHTMLLoader
)
from langchain.schema import Document as LangChainDocument

from ..models.document import Document
from ..exceptions import ConfigurationException, StorageException


class DocumentProcessor:
    """Service for processing documents including loading, splitting, and metadata extraction."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        text_splitter_type: str = "recursive"
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
            text_splitter_type: Type of text splitter to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter_type = text_splitter_type
        
        # Validate parameters
        self._validate_parameters()
        
        # Initialize text splitter
        self.text_splitter = self._create_text_splitter()
        
        # Supported file extensions and their loaders
        self.supported_extensions = {
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
        }
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.chunk_size <= 0:
            raise ConfigurationException("chunk_size must be greater than 0")
        
        if self.chunk_overlap < 0:
            raise ConfigurationException("chunk_overlap cannot be negative")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationException("chunk_overlap must be less than chunk_size")
        
        valid_splitter_types = ["recursive", "character", "token"]
        if self.text_splitter_type not in valid_splitter_types:
            raise ConfigurationException(
                f"Invalid text_splitter_type '{self.text_splitter_type}'. "
                f"Must be one of: {valid_splitter_types}"
            )
    
    def _create_text_splitter(self):
        """Create the appropriate text splitter based on configuration."""
        common_kwargs = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        
        if self.text_splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(**common_kwargs)
        elif self.text_splitter_type == "character":
            return CharacterTextSplitter(**common_kwargs)
        elif self.text_splitter_type == "token":
            return TokenTextSplitter(**common_kwargs)
        else:
            raise ConfigurationException(f"Unsupported text splitter type: {self.text_splitter_type}")
    
    def process_documents(
        self,
        file_paths: List[Union[str, Path]],
        metadata_override: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process documents from file paths.
        
        Args:
            file_paths: List of file paths to process
            metadata_override: Optional metadata to add to all documents
            
        Returns:
            List of processed Document objects
            
        Raises:
            StorageException: If document processing fails
        """
        if not file_paths:
            return []
        
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = self._process_single_file(file_path, metadata_override)
                all_documents.extend(documents)
            except Exception as e:
                raise StorageException(
                    f"Failed to process file '{file_path}': {str(e)}"
                )
        
        return all_documents
    
    def _process_single_file(
        self,
        file_path: Union[str, Path],
        metadata_override: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process a single file and return Document objects."""
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise StorageException(f"File not found: {file_path}")
        
        # Check if file extension is supported
        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise StorageException(
                f"Unsupported file extension '{extension}'. "
                f"Supported extensions: {list(self.supported_extensions.keys())}"
            )
        
        # Load the document
        loader_class = self.supported_extensions[extension]
        loader = loader_class(str(file_path))
        
        try:
            langchain_docs = loader.load()
        except Exception as e:
            raise StorageException(f"Failed to load document: {str(e)}")
        
        # Process each loaded document
        processed_docs = []
        for i, langchain_doc in enumerate(langchain_docs):
            # Extract and enhance metadata
            metadata = self._extract_metadata(file_path, langchain_doc, i)
            if metadata_override:
                metadata.update(metadata_override)
            
            # Split the document into chunks
            chunks = self.split_text(langchain_doc.page_content)
            
            # Create Document objects for each chunk
            for j, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                })
                
                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                processed_docs.append(doc)
        
        return processed_docs
    
    def _extract_metadata(
        self,
        file_path: Path,
        langchain_doc: LangChainDocument,
        doc_index: int
    ) -> Dict[str, Any]:
        """Extract metadata from file and document."""
        # Get file statistics
        stat = file_path.stat()
        
        # Base metadata
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "file_modified_time": stat.st_mtime,
            "document_index": doc_index,
            "original_content_length": len(langchain_doc.page_content)
        }
        
        # Add any existing metadata from the document
        if hasattr(langchain_doc, 'metadata') and langchain_doc.metadata:
            metadata.update(langchain_doc.metadata)
        
        return metadata
    
    def process_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """
        Process raw text strings into Document objects.
        
        Args:
            texts: List of text strings to process
            metadatas: Optional list of metadata dictionaries for each text
            
        Returns:
            List of processed Document objects
        """
        if not texts:
            return []
        
        if metadatas and len(metadatas) != len(texts):
            raise ConfigurationException(
                f"Number of metadatas ({len(metadatas)}) must match number of texts ({len(texts)})"
            )
        
        all_documents = []
        
        for i, text in enumerate(texts):
            # Get metadata for this text
            metadata = metadatas[i] if metadatas else {}
            metadata.update({
                "text_index": i,
                "original_content_length": len(text)
            })
            
            # Split the text into chunks
            chunks = self.split_text(text)
            
            # Create Document objects for each chunk
            for j, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                })
                
                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                all_documents.append(doc)
        
        return all_documents
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using the configured text splitter.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        try:
            chunks = self.text_splitter.split_text(text)
            return [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks
        except Exception as e:
            raise StorageException(f"Failed to split text: {str(e)}")
    
    def process_documents_batch(
        self,
        file_paths: List[Union[str, Path]],
        batch_size: int = 10,
        metadata_override: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process documents in batches for memory efficiency.
        
        Args:
            file_paths: List of file paths to process
            batch_size: Number of files to process in each batch
            metadata_override: Optional metadata to add to all documents
            
        Returns:
            List of processed Document objects
        """
        if batch_size <= 0:
            raise ConfigurationException("batch_size must be greater than 0")
        
        all_documents = []
        
        # Process files in batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            batch_documents = self.process_documents(batch, metadata_override)
            all_documents.extend(batch_documents)
        
        return all_documents
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return list(self.supported_extensions.keys())
    
    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file is supported based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is supported, False otherwise
        """
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_extensions
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise StorageException(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        return {
            "path": str(file_path),
            "name": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_time": stat.st_mtime,
            "is_supported": self.is_supported_file(file_path),
            "estimated_chunks": self._estimate_chunks(file_path)
        }
    
    def _estimate_chunks(self, file_path: Path) -> Optional[int]:
        """Estimate the number of chunks a file will produce."""
        try:
            if file_path.suffix.lower() == '.txt':
                # For text files, we can estimate based on file size
                stat = file_path.stat()
                estimated_chars = stat.st_size  # Rough estimate
                return max(1, estimated_chars // self.chunk_size)
            else:
                # For other file types, we can't easily estimate without loading
                return None
        except Exception:
            return None
    
    def validate_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, List[str]]:
        """
        Validate a list of file paths.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dictionary with 'valid', 'missing', and 'unsupported' file lists
        """
        valid = []
        missing = []
        unsupported = []
        
        for file_path in file_paths:
            path = Path(file_path)
            
            if not path.exists():
                missing.append(str(file_path))
            elif not self.is_supported_file(path):
                unsupported.append(str(file_path))
            else:
                valid.append(str(file_path))
        
        return {
            "valid": valid,
            "missing": missing,
            "unsupported": unsupported
        }
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about the document processor configuration.
        
        Returns:
            Dictionary containing processor information
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "text_splitter_type": self.text_splitter_type,
            "supported_extensions": self.get_supported_extensions(),
            "text_splitter_class": self.text_splitter.__class__.__name__
        }
    
    def health_check(self) -> bool:
        """
        Check if the document processor is healthy.
        
        Returns:
            True if the processor is healthy, False otherwise
        """
        try:
            # Test text splitting functionality
            test_text = "This is a test document for health check purposes."
            chunks = self.split_text(test_text)
            return len(chunks) > 0
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of the document processor."""
        return (
            f"DocumentProcessor(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"splitter={self.text_splitter_type})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the document processor."""
        return (
            f"DocumentProcessor(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"text_splitter_type='{self.text_splitter_type}', "
            f"supported_extensions={len(self.supported_extensions)})"
        )