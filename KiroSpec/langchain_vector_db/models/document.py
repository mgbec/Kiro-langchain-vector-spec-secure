"""
Document model with enhanced metadata support.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain.schema import Document as LangChainDocument


@dataclass
class Document:
    """
    Extended LangChain Document class with additional metadata and security features.
    """
    
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Security metadata
    access_level: str = "public"  # "public", "internal", "confidential"
    owner: Optional[str] = None
    pii_detected: bool = False
    encrypted: bool = False
    
    def __post_init__(self):
        """Initialize document with auto-generated fields."""
        if self.doc_id is None:
            self.doc_id = self._generate_doc_id()
        
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        if self.updated_at is None:
            self.updated_at = self.created_at
    
    def _generate_doc_id(self) -> str:
        """Generate a unique document ID."""
        return str(uuid.uuid4())
    
    def update_content(self, new_content: str, user_id: Optional[str] = None) -> None:
        """
        Update document content and metadata.
        
        Args:
            new_content: New page content
            user_id: ID of user making the update
        """
        self.page_content = new_content
        self.updated_at = datetime.utcnow()
        
        if user_id:
            self.metadata["last_modified_by"] = user_id
        
        # Clear embedding since content changed
        self.embedding = None
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add or update metadata field.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()
    
    def remove_metadata(self, key: str) -> bool:
        """
        Remove metadata field.
        
        Args:
            key: Metadata key to remove
            
        Returns:
            True if key was removed, False if key didn't exist
        """
        if key in self.metadata:
            del self.metadata[key]
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def set_access_level(self, access_level: str, user_id: Optional[str] = None) -> None:
        """
        Set document access level.
        
        Args:
            access_level: New access level ("public", "internal", "confidential")
            user_id: ID of user making the change
        """
        valid_levels = ["public", "internal", "confidential"]
        if access_level not in valid_levels:
            raise ValueError(f"Invalid access_level. Must be one of: {valid_levels}")
        
        self.access_level = access_level
        self.updated_at = datetime.utcnow()
        
        if user_id:
            self.metadata["access_level_changed_by"] = user_id
            self.metadata["access_level_changed_at"] = self.updated_at.isoformat()
    
    def mark_pii_detected(self, pii_types: List[str]) -> None:
        """
        Mark document as containing PII.
        
        Args:
            pii_types: List of detected PII types
        """
        self.pii_detected = True
        self.metadata["pii_types"] = pii_types
        self.metadata["pii_detected_at"] = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow()
    
    def mark_encrypted(self, encryption_method: str) -> None:
        """
        Mark document as encrypted.
        
        Args:
            encryption_method: Encryption method used
        """
        self.encrypted = True
        self.metadata["encryption_method"] = encryption_method
        self.metadata["encrypted_at"] = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow()
    
    def to_langchain_document(self) -> LangChainDocument:
        """
        Convert to standard LangChain Document.
        
        Returns:
            LangChain Document instance
        """
        # Include our additional metadata in the LangChain document
        enhanced_metadata = self.metadata.copy()
        enhanced_metadata.update({
            "doc_id": self.doc_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "access_level": self.access_level,
            "owner": self.owner,
            "pii_detected": self.pii_detected,
            "encrypted": self.encrypted,
        })
        
        return LangChainDocument(
            page_content=self.page_content,
            metadata=enhanced_metadata
        )
    
    @classmethod
    def from_langchain_document(
        cls, 
        langchain_doc: LangChainDocument,
        doc_id: Optional[str] = None
    ) -> "Document":
        """
        Create Document from LangChain Document.
        
        Args:
            langchain_doc: LangChain Document instance
            doc_id: Optional document ID override
            
        Returns:
            Document instance
        """
        metadata = langchain_doc.metadata.copy()
        
        # Extract our special fields from metadata
        extracted_doc_id = doc_id or metadata.pop("doc_id", None)
        created_at_str = metadata.pop("created_at", None)
        updated_at_str = metadata.pop("updated_at", None)
        access_level = metadata.pop("access_level", "public")
        owner = metadata.pop("owner", None)
        pii_detected = metadata.pop("pii_detected", False)
        encrypted = metadata.pop("encrypted", False)
        
        # Parse datetime strings
        created_at = None
        updated_at = None
        
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        if updated_at_str:
            try:
                updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        return cls(
            page_content=langchain_doc.page_content,
            metadata=metadata,
            doc_id=extracted_doc_id,
            created_at=created_at,
            updated_at=updated_at,
            access_level=access_level,
            owner=owner,
            pii_detected=pii_detected,
            encrypted=encrypted
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary for serialization.
        
        Returns:
            Dictionary representation of the document
        """
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "doc_id": self.doc_id,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "access_level": self.access_level,
            "owner": self.owner,
            "pii_detected": self.pii_detected,
            "encrypted": self.encrypted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Create document from dictionary.
        
        Args:
            data: Dictionary containing document data
            
        Returns:
            Document instance
        """
        # Parse datetime strings
        created_at = None
        updated_at = None
        
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        if data.get("updated_at"):
            try:
                updated_at = datetime.fromisoformat(data["updated_at"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        return cls(
            page_content=data["page_content"],
            metadata=data.get("metadata", {}),
            doc_id=data.get("doc_id"),
            embedding=data.get("embedding"),
            created_at=created_at,
            updated_at=updated_at,
            access_level=data.get("access_level", "public"),
            owner=data.get("owner"),
            pii_detected=data.get("pii_detected", False),
            encrypted=data.get("encrypted", False),
        )
    
    def to_json(self) -> str:
        """
        Convert document to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), default=str, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Document":
        """
        Create document from JSON string.
        
        Args:
            json_str: JSON string containing document data
            
        Returns:
            Document instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def clone(self, new_doc_id: bool = True) -> "Document":
        """
        Create a copy of the document.
        
        Args:
            new_doc_id: Whether to generate a new document ID
            
        Returns:
            Cloned document instance
        """
        cloned = Document(
            page_content=self.page_content,
            metadata=self.metadata.copy(),
            doc_id=None if new_doc_id else self.doc_id,
            embedding=self.embedding.copy() if self.embedding else None,
            created_at=self.created_at,
            updated_at=self.updated_at,
            access_level=self.access_level,
            owner=self.owner,
            pii_detected=self.pii_detected,
            encrypted=self.encrypted,
        )
        
        if new_doc_id:
            cloned.created_at = datetime.utcnow()
            cloned.updated_at = cloned.created_at
        
        return cloned
    
    def get_content_preview(self, max_length: int = 100) -> str:
        """
        Get a preview of the document content.
        
        Args:
            max_length: Maximum length of preview
            
        Returns:
            Truncated content preview
        """
        if len(self.page_content) <= max_length:
            return self.page_content
        
        return self.page_content[:max_length] + "..."
    
    def __str__(self) -> str:
        """String representation of the document."""
        return f"Document(id={self.doc_id}, content_length={len(self.page_content)}, access_level={self.access_level})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the document."""
        return (
            f"Document(doc_id='{self.doc_id}', "
            f"content_length={len(self.page_content)}, "
            f"access_level='{self.access_level}', "
            f"pii_detected={self.pii_detected}, "
            f"encrypted={self.encrypted}, "
            f"created_at={self.created_at})"
        )