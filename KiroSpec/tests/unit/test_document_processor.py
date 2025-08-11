"""
Unit tests for DocumentProcessor.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document as LangChainDocument

from langchain_vector_db.services.document_processor import DocumentProcessor
from langchain_vector_db.models.document import Document
from langchain_vector_db.exceptions import ConfigurationException, StorageException


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    def test_initialization_with_defaults(self):
        """Test processor initialization with default parameters."""
        processor = DocumentProcessor()
        
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
        assert processor.text_splitter_type == "recursive"
        assert processor.text_splitter is not None
    
    def test_initialization_with_custom_parameters(self):
        """Test processor initialization with custom parameters."""
        processor = DocumentProcessor(
            chunk_size=500,
            chunk_overlap=50,
            text_splitter_type="character"
        )
        
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
        assert processor.text_splitter_type == "character"
    
    def test_invalid_chunk_size(self):
        """Test initialization with invalid chunk size."""
        with pytest.raises(ConfigurationException) as exc_info:
            DocumentProcessor(chunk_size=0)
        
        assert "chunk_size must be greater than 0" in str(exc_info.value)
    
    def test_invalid_chunk_overlap(self):
        """Test initialization with invalid chunk overlap."""
        with pytest.raises(ConfigurationException) as exc_info:
            DocumentProcessor(chunk_overlap=-1)
        
        assert "chunk_overlap cannot be negative" in str(exc_info.value)
    
    def test_chunk_overlap_greater_than_size(self):
        """Test initialization with chunk overlap greater than size."""
        with pytest.raises(ConfigurationException) as exc_info:
            DocumentProcessor(chunk_size=100, chunk_overlap=100)
        
        assert "chunk_overlap must be less than chunk_size" in str(exc_info.value)
    
    def test_invalid_text_splitter_type(self):
        """Test initialization with invalid text splitter type."""
        with pytest.raises(ConfigurationException) as exc_info:
            DocumentProcessor(text_splitter_type="invalid")
        
        assert "Invalid text_splitter_type" in str(exc_info.value)
    
    def test_split_text_basic(self):
        """Test basic text splitting functionality."""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        
        text = "This is a long text that should be split into multiple chunks for testing purposes."
        chunks = processor.split_text(text)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= 60 for chunk in chunks)  # Allow some flexibility
    
    def test_split_text_empty(self):
        """Test splitting empty text."""
        processor = DocumentProcessor()
        
        chunks = processor.split_text("")
        assert chunks == []
    
    def test_split_text_short(self):
        """Test splitting text shorter than chunk size."""
        processor = DocumentProcessor(chunk_size=1000)
        
        text = "Short text"
        chunks = processor.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_process_texts_basic(self):
        """Test processing raw text strings."""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        
        texts = [
            "This is the first document that needs to be processed.",
            "This is the second document for testing purposes."
        ]
        
        documents = processor.process_texts(texts)
        
        assert len(documents) >= 2  # At least one chunk per text
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(doc.page_content for doc in documents)
        
        # Check metadata
        for doc in documents:
            assert "text_index" in doc.metadata
            assert "chunk_index" in doc.metadata
            assert "total_chunks" in doc.metadata
    
    def test_process_texts_with_metadata(self):
        """Test processing texts with custom metadata."""
        processor = DocumentProcessor()
        
        texts = ["Test document"]
        metadatas = [{"source": "test", "category": "example"}]
        
        documents = processor.process_texts(texts, metadatas)
        
        assert len(documents) >= 1
        assert documents[0].metadata["source"] == "test"
        assert documents[0].metadata["category"] == "example"
    
    def test_process_texts_metadata_mismatch(self):
        """Test processing texts with mismatched metadata count."""
        processor = DocumentProcessor()
        
        texts = ["Text 1", "Text 2"]
        metadatas = [{"source": "test"}]  # Only one metadata for two texts
        
        with pytest.raises(ConfigurationException) as exc_info:
            processor.process_texts(texts, metadatas)
        
        assert "Number of metadatas" in str(exc_info.value)
    
    def test_process_texts_empty_list(self):
        """Test processing empty text list."""
        processor = DocumentProcessor()
        
        documents = processor.process_texts([])
        assert documents == []
    
    def test_get_supported_extensions(self):
        """Test getting supported file extensions."""
        processor = DocumentProcessor()
        
        extensions = processor.get_supported_extensions()
        
        assert isinstance(extensions, list)
        assert '.txt' in extensions
        assert '.pdf' in extensions
        assert '.md' in extensions
    
    def test_is_supported_file(self):
        """Test checking if file is supported."""
        processor = DocumentProcessor()
        
        assert processor.is_supported_file("test.txt") is True
        assert processor.is_supported_file("test.pdf") is True
        assert processor.is_supported_file("test.xyz") is False
        assert processor.is_supported_file("test") is False
    
    def test_validate_files(self):
        """Test file validation functionality."""
        processor = DocumentProcessor()
        
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a valid file
            valid_file = temp_path / "test.txt"
            valid_file.write_text("Test content")
            
            file_paths = [
                str(valid_file),  # Valid file
                str(temp_path / "missing.txt"),  # Missing file
                str(temp_path / "unsupported.xyz")  # Unsupported extension
            ]
            
            result = processor.validate_files(file_paths)
            
            assert len(result["valid"]) == 1
            assert len(result["missing"]) == 1
            assert len(result["unsupported"]) == 1
            assert str(valid_file) in result["valid"]
    
    def test_get_file_info(self):
        """Test getting file information."""
        processor = DocumentProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Test content for file info")
            temp_file_path = temp_file.name
        
        try:
            info = processor.get_file_info(temp_file_path)
            
            assert "path" in info
            assert "name" in info
            assert "extension" in info
            assert "size_bytes" in info
            assert "is_supported" in info
            assert info["extension"] == ".txt"
            assert info["is_supported"] is True
            
        finally:
            os.unlink(temp_file_path)
    
    def test_get_file_info_missing_file(self):
        """Test getting info for missing file."""
        processor = DocumentProcessor()
        
        with pytest.raises(StorageException) as exc_info:
            processor.get_file_info("nonexistent.txt")
        
        assert "File not found" in str(exc_info.value)
    
    def test_process_documents_batch(self):
        """Test batch processing of documents."""
        processor = DocumentProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple test files
            file_paths = []
            for i in range(5):
                file_path = temp_path / f"test_{i}.txt"
                file_path.write_text(f"Test content for document {i}")
                file_paths.append(str(file_path))
            
            # Process in batches
            documents = processor.process_documents_batch(file_paths, batch_size=2)
            
            assert len(documents) == 5  # One document per file
            assert all(isinstance(doc, Document) for doc in documents)
    
    def test_process_documents_batch_invalid_batch_size(self):
        """Test batch processing with invalid batch size."""
        processor = DocumentProcessor()
        
        with pytest.raises(ConfigurationException) as exc_info:
            processor.process_documents_batch(["test.txt"], batch_size=0)
        
        assert "batch_size must be greater than 0" in str(exc_info.value)
    
    @patch('langchain_vector_db.services.document_processor.TextLoader')
    def test_process_documents_with_mock_loader(self, mock_loader_class):
        """Test document processing with mocked loader."""
        # Setup mock
        mock_loader = Mock()
        mock_langchain_doc = LangChainDocument(
            page_content="Test document content",
            metadata={"source": "test.txt"}
        )
        mock_loader.load.return_value = [mock_langchain_doc]
        mock_loader_class.return_value = mock_loader
        
        processor = DocumentProcessor(chunk_size=50)
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            documents = processor.process_documents([temp_file_path])
            
            assert len(documents) >= 1
            assert all(isinstance(doc, Document) for doc in documents)
            
            # Check that metadata was properly extracted
            for doc in documents:
                assert "source" in doc.metadata
                assert "filename" in doc.metadata
                assert "file_extension" in doc.metadata
                
        finally:
            os.unlink(temp_file_path)
    
    def test_process_documents_unsupported_extension(self):
        """Test processing document with unsupported extension."""
        processor = DocumentProcessor()
        
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            with pytest.raises(StorageException) as exc_info:
                processor.process_documents([temp_file_path])
            
            assert "Unsupported file extension" in str(exc_info.value)
            
        finally:
            os.unlink(temp_file_path)
    
    def test_process_documents_missing_file(self):
        """Test processing missing document."""
        processor = DocumentProcessor()
        
        with pytest.raises(StorageException) as exc_info:
            processor.process_documents(["nonexistent.txt"])
        
        assert "File not found" in str(exc_info.value)
    
    def test_get_processor_info(self):
        """Test getting processor information."""
        processor = DocumentProcessor(
            chunk_size=500,
            chunk_overlap=50,
            text_splitter_type="character"
        )
        
        info = processor.get_processor_info()
        
        assert info["chunk_size"] == 500
        assert info["chunk_overlap"] == 50
        assert info["text_splitter_type"] == "character"
        assert "supported_extensions" in info
        assert "text_splitter_class" in info
    
    def test_health_check_healthy(self):
        """Test health check when processor is healthy."""
        processor = DocumentProcessor()
        
        assert processor.health_check() is True
    
    def test_health_check_unhealthy(self):
        """Test health check when processor is unhealthy."""
        processor = DocumentProcessor()
        
        # Mock the text splitter to raise an exception
        with patch.object(processor, 'split_text', side_effect=Exception("Splitter error")):
            assert processor.health_check() is False
    
    def test_string_representations(self):
        """Test string representations of document processor."""
        processor = DocumentProcessor(
            chunk_size=500,
            chunk_overlap=50,
            text_splitter_type="character"
        )
        
        str_repr = str(processor)
        assert "DocumentProcessor" in str_repr
        assert "500" in str_repr
        assert "50" in str_repr
        assert "character" in str_repr
        
        repr_str = repr(processor)
        assert "DocumentProcessor" in repr_str
        assert "500" in repr_str
        assert "character" in repr_str


class TestDocumentProcessorIntegration:
    """Integration tests for DocumentProcessor with real files."""
    
    def test_process_real_text_file(self):
        """Test processing a real text file."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            content = "This is a test document. " * 20  # Create longer content
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            documents = processor.process_documents([temp_file_path])
            
            assert len(documents) > 1  # Should be split into multiple chunks
            assert all(isinstance(doc, Document) for doc in documents)
            
            # Verify metadata
            for doc in documents:
                assert doc.metadata["filename"] == Path(temp_file_path).name
                assert doc.metadata["file_extension"] == ".txt"
                assert "chunk_index" in doc.metadata
                assert "total_chunks" in doc.metadata
                
        finally:
            os.unlink(temp_file_path)
    
    def test_different_text_splitter_types(self):
        """Test different text splitter types."""
        text = "This is a test. Another sentence. And another one."
        
        for splitter_type in ["recursive", "character", "token"]:
            processor = DocumentProcessor(
                chunk_size=30,
                chunk_overlap=5,
                text_splitter_type=splitter_type
            )
            
            chunks = processor.split_text(text)
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__])