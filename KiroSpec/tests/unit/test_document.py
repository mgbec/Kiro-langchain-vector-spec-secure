"""
Unit tests for Document model.
"""

import json
import pytest
from datetime import datetime
from langchain.schema import Document as LangChainDocument

from langchain_vector_db.models.document import Document


class TestDocument:
    """Test cases for Document model."""
    
    def test_document_creation_with_defaults(self):
        """Test document creation with default values."""
        doc = Document(page_content="Test content")
        
        assert doc.page_content == "Test content"
        assert doc.metadata == {}
        assert doc.doc_id is not None
        assert doc.embedding is None
        assert doc.created_at is not None
        assert doc.updated_at is not None
        assert doc.access_level == "public"
        assert doc.owner is None
        assert doc.pii_detected is False
        assert doc.encrypted is False
    
    def test_document_creation_with_custom_values(self):
        """Test document creation with custom values."""
        custom_metadata = {"source": "test.txt", "category": "test"}
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        
        doc = Document(
            page_content="Custom content",
            metadata=custom_metadata,
            doc_id="custom-id",
            access_level="confidential",
            owner="user123",
            created_at=custom_time
        )
        
        assert doc.page_content == "Custom content"
        assert doc.metadata == custom_metadata
        assert doc.doc_id == "custom-id"
        assert doc.access_level == "confidential"
        assert doc.owner == "user123"
        assert doc.created_at == custom_time
    
    def test_update_content(self):
        """Test content update functionality."""
        doc = Document(page_content="Original content")
        original_updated_at = doc.updated_at
        
        # Wait a bit to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        doc.update_content("New content", user_id="user123")
        
        assert doc.page_content == "New content"
        assert doc.updated_at > original_updated_at
        assert doc.metadata["last_modified_by"] == "user123"
        assert doc.embedding is None  # Should be cleared
    
    def test_add_metadata(self):
        """Test adding metadata."""
        doc = Document(page_content="Test content")
        original_updated_at = doc.updated_at
        
        import time
        time.sleep(0.01)
        
        doc.add_metadata("key1", "value1")
        doc.add_metadata("key2", 42)
        
        assert doc.metadata["key1"] == "value1"
        assert doc.metadata["key2"] == 42
        assert doc.updated_at > original_updated_at
    
    def test_remove_metadata(self):
        """Test removing metadata."""
        doc = Document(
            page_content="Test content",
            metadata={"key1": "value1", "key2": "value2"}
        )
        
        # Remove existing key
        result = doc.remove_metadata("key1")
        assert result is True
        assert "key1" not in doc.metadata
        assert "key2" in doc.metadata
        
        # Try to remove non-existing key
        result = doc.remove_metadata("nonexistent")
        assert result is False
    
    def test_set_access_level(self):
        """Test setting access level."""
        doc = Document(page_content="Test content")
        
        doc.set_access_level("confidential", user_id="admin")
        
        assert doc.access_level == "confidential"
        assert doc.metadata["access_level_changed_by"] == "admin"
        assert "access_level_changed_at" in doc.metadata
    
    def test_set_invalid_access_level(self):
        """Test setting invalid access level."""
        doc = Document(page_content="Test content")
        
        with pytest.raises(ValueError) as exc_info:
            doc.set_access_level("invalid")
        
        assert "Invalid access_level" in str(exc_info.value)
    
    def test_mark_pii_detected(self):
        """Test marking PII as detected."""
        doc = Document(page_content="Test content")
        
        pii_types = ["email", "phone"]
        doc.mark_pii_detected(pii_types)
        
        assert doc.pii_detected is True
        assert doc.metadata["pii_types"] == pii_types
        assert "pii_detected_at" in doc.metadata
    
    def test_mark_encrypted(self):
        """Test marking document as encrypted."""
        doc = Document(page_content="Test content")
        
        doc.mark_encrypted("AES-256-GCM")
        
        assert doc.encrypted is True
        assert doc.metadata["encryption_method"] == "AES-256-GCM"
        assert "encrypted_at" in doc.metadata
    
    def test_to_langchain_document(self):
        """Test conversion to LangChain document."""
        doc = Document(
            page_content="Test content",
            metadata={"source": "test.txt"},
            access_level="internal",
            owner="user123"
        )
        
        langchain_doc = doc.to_langchain_document()
        
        assert isinstance(langchain_doc, LangChainDocument)
        assert langchain_doc.page_content == "Test content"
        assert langchain_doc.metadata["source"] == "test.txt"
        assert langchain_doc.metadata["doc_id"] == doc.doc_id
        assert langchain_doc.metadata["access_level"] == "internal"
        assert langchain_doc.metadata["owner"] == "user123"
    
    def test_from_langchain_document(self):
        """Test creation from LangChain document."""
        langchain_doc = LangChainDocument(
            page_content="Test content",
            metadata={
                "source": "test.txt",
                "doc_id": "test-id",
                "access_level": "confidential",
                "pii_detected": True
            }
        )
        
        doc = Document.from_langchain_document(langchain_doc)
        
        assert doc.page_content == "Test content"
        assert doc.metadata["source"] == "test.txt"
        assert doc.doc_id == "test-id"
        assert doc.access_level == "confidential"
        assert doc.pii_detected is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        doc = Document(
            page_content="Test content",
            metadata={"source": "test.txt"},
            access_level="internal"
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["page_content"] == "Test content"
        assert doc_dict["metadata"]["source"] == "test.txt"
        assert doc_dict["doc_id"] == doc.doc_id
        assert doc_dict["access_level"] == "internal"
        assert isinstance(doc_dict["created_at"], str)  # Should be ISO format
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        doc_dict = {
            "page_content": "Test content",
            "metadata": {"source": "test.txt"},
            "doc_id": "test-id",
            "access_level": "confidential",
            "owner": "user123",
            "pii_detected": True,
            "encrypted": False,
            "created_at": "2023-01-01T12:00:00",
            "updated_at": "2023-01-01T12:30:00"
        }
        
        doc = Document.from_dict(doc_dict)
        
        assert doc.page_content == "Test content"
        assert doc.metadata["source"] == "test.txt"
        assert doc.doc_id == "test-id"
        assert doc.access_level == "confidential"
        assert doc.owner == "user123"
        assert doc.pii_detected is True
        assert doc.encrypted is False
        assert doc.created_at.year == 2023
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        original_doc = Document(
            page_content="Test content",
            metadata={"source": "test.txt"},
            access_level="internal"
        )
        
        # Serialize to JSON
        json_str = original_doc.to_json()
        assert isinstance(json_str, str)
        
        # Deserialize from JSON
        restored_doc = Document.from_json(json_str)
        
        assert restored_doc.page_content == original_doc.page_content
        assert restored_doc.metadata == original_doc.metadata
        assert restored_doc.doc_id == original_doc.doc_id
        assert restored_doc.access_level == original_doc.access_level
    
    def test_clone_document(self):
        """Test document cloning."""
        original_doc = Document(
            page_content="Test content",
            metadata={"source": "test.txt"},
            embedding=[1.0, 2.0, 3.0],
            access_level="internal"
        )
        
        # Clone with new ID
        cloned_doc = original_doc.clone(new_doc_id=True)
        
        assert cloned_doc.page_content == original_doc.page_content
        assert cloned_doc.metadata == original_doc.metadata
        assert cloned_doc.doc_id != original_doc.doc_id
        assert cloned_doc.embedding == original_doc.embedding
        assert cloned_doc.access_level == original_doc.access_level
        
        # Clone with same ID
        cloned_same_id = original_doc.clone(new_doc_id=False)
        assert cloned_same_id.doc_id == original_doc.doc_id
    
    def test_get_content_preview(self):
        """Test content preview functionality."""
        short_content = "Short content"
        long_content = "This is a very long content that should be truncated when getting a preview"
        
        short_doc = Document(page_content=short_content)
        long_doc = Document(page_content=long_content)
        
        # Short content should not be truncated
        assert short_doc.get_content_preview(100) == short_content
        
        # Long content should be truncated
        preview = long_doc.get_content_preview(20)
        assert len(preview) == 23  # 20 chars + "..."
        assert preview.endswith("...")
    
    def test_string_representations(self):
        """Test string representations of document."""
        doc = Document(
            page_content="Test content",
            access_level="confidential"
        )
        
        str_repr = str(doc)
        assert "Document(" in str_repr
        assert doc.doc_id in str_repr
        assert "confidential" in str_repr
        
        repr_str = repr(doc)
        assert "Document(" in repr_str
        assert doc.doc_id in repr_str
        assert "confidential" in repr_str
        assert "pii_detected=False" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])