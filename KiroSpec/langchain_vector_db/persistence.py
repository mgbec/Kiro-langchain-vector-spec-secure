"""
Persistence management layer for vector database operations.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from .models.config import VectorDBConfig
from .storage.interface import VectorStoreInterface
from .exceptions import StorageException


@dataclass
class PersistenceMetadata:
    """Metadata about persistence operations."""
    
    last_persist_time: datetime
    last_load_time: Optional[datetime]
    persist_count: int
    load_count: int
    storage_type: str
    storage_path: str
    vector_count: int
    embedding_dimension: Optional[int]
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data["last_persist_time"] = self.last_persist_time.isoformat()
        if self.last_load_time:
            data["last_load_time"] = self.last_load_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersistenceMetadata":
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        data["last_persist_time"] = datetime.fromisoformat(data["last_persist_time"])
        if data.get("last_load_time"):
            data["last_load_time"] = datetime.fromisoformat(data["last_load_time"])
        return cls(**data)


class PersistenceManager:
    """Manages persistence operations for vector stores."""
    
    def __init__(
        self,
        vector_store: VectorStoreInterface,
        config: VectorDBConfig,
        auto_persist: bool = True,
        persist_interval: int = 300  # 5 minutes
    ):
        """
        Initialize persistence manager.
        
        Args:
            vector_store: Vector store to manage
            config: Vector database configuration
            auto_persist: Whether to enable automatic persistence
            persist_interval: Interval between auto-persists (seconds)
        """
        self.vector_store = vector_store
        self.config = config
        self.auto_persist = auto_persist
        self.persist_interval = persist_interval
        
        # Persistence state
        self.metadata: Optional[PersistenceMetadata] = None
        self.last_auto_persist = time.time()
        self.pending_changes = False
        
        # Metadata file path
        self.metadata_path = self._get_metadata_path()
        
        # Load existing metadata
        self._load_metadata()
    
    def _get_metadata_path(self) -> Path:
        """Get path for persistence metadata file."""
        if self.config.storage_type == "local":
            base_path = Path(self.config.storage_path)
            return base_path / ".persistence_metadata.json"
        else:
            # For S3 and other cloud storage, use local temp directory
            temp_dir = Path.home() / ".langchain_vector_db" / "metadata"
            temp_dir.mkdir(parents=True, exist_ok=True)
            # Create unique filename based on storage path
            safe_name = self.config.storage_path.replace("/", "_").replace(":", "_")
            return temp_dir / f"{safe_name}_persistence_metadata.json"
    
    def _load_metadata(self) -> None:
        """Load persistence metadata from file."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                self.metadata = PersistenceMetadata.from_dict(data)
        except Exception:
            # If metadata loading fails, start fresh
            self.metadata = None
    
    def _save_metadata(self) -> None:
        """Save persistence metadata to file."""
        try:
            if self.metadata:
                self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.metadata.to_dict(), f, indent=2)
        except Exception:
            # Don't fail the operation if metadata saving fails
            pass
    
    def _update_metadata(self, operation: str) -> None:
        """Update persistence metadata after an operation."""
        now = datetime.utcnow()
        vector_count = self.vector_store.get_vector_count()
        
        if self.metadata is None:
            # Create new metadata
            self.metadata = PersistenceMetadata(
                last_persist_time=now if operation == "persist" else datetime.min,
                last_load_time=now if operation == "load" else None,
                persist_count=1 if operation == "persist" else 0,
                load_count=1 if operation == "load" else 0,
                storage_type=self.config.storage_type,
                storage_path=self.config.storage_path,
                vector_count=vector_count,
                embedding_dimension=None
            )
        else:
            # Update existing metadata
            if operation == "persist":
                self.metadata.last_persist_time = now
                self.metadata.persist_count += 1
            elif operation == "load":
                self.metadata.last_load_time = now
                self.metadata.load_count += 1
            
            self.metadata.vector_count = vector_count
        
        self._save_metadata()
    
    def persist(self, force: bool = False) -> bool:
        """
        Persist the vector store with metadata tracking.
        
        Args:
            force: Force persistence even if no changes are pending
            
        Returns:
            True if persistence was successful
            
        Raises:
            StorageException: If persistence fails
        """
        try:
            # Check if persistence is needed
            if not force and not self.pending_changes:
                return True
            
            # Perform the actual persistence
            success = self.vector_store.persist()
            
            if success:
                self._update_metadata("persist")
                self.pending_changes = False
                self.last_auto_persist = time.time()
            
            return success
            
        except Exception as e:
            raise StorageException(f"Persistence failed: {str(e)}")
    
    def load(self) -> bool:
        """
        Load the vector store with metadata tracking.
        
        Returns:
            True if loading was successful
            
        Raises:
            StorageException: If loading fails
        """
        try:
            success = self.vector_store.load()
            
            if success:
                self._update_metadata("load")
                self.pending_changes = False
            
            return success
            
        except Exception as e:
            raise StorageException(f"Loading failed: {str(e)}")
    
    def mark_changes_pending(self) -> None:
        """Mark that there are pending changes that need persistence."""
        self.pending_changes = True
        
        # Auto-persist if enabled and interval has passed
        if self.auto_persist and self._should_auto_persist():
            try:
                self.persist()
            except Exception:
                # Don't fail the operation if auto-persist fails
                pass
    
    def _should_auto_persist(self) -> bool:
        """Check if auto-persistence should be triggered."""
        return (time.time() - self.last_auto_persist) >= self.persist_interval
    
    def force_persist(self) -> bool:
        """Force immediate persistence regardless of pending changes."""
        return self.persist(force=True)
    
    def get_persistence_info(self) -> Dict[str, Any]:
        """
        Get information about persistence state.
        
        Returns:
            Dictionary with persistence information
        """
        info = {
            "auto_persist_enabled": self.auto_persist,
            "persist_interval": self.persist_interval,
            "pending_changes": self.pending_changes,
            "last_auto_persist": self.last_auto_persist,
            "metadata_path": str(self.metadata_path),
            "metadata": None
        }
        
        if self.metadata:
            info["metadata"] = self.metadata.to_dict()
        
        return info
    
    def get_persistence_stats(self) -> Dict[str, Any]:
        """
        Get persistence statistics.
        
        Returns:
            Dictionary with persistence statistics
        """
        if not self.metadata:
            return {
                "total_persists": 0,
                "total_loads": 0,
                "last_persist": None,
                "last_load": None,
                "current_vector_count": self.vector_store.get_vector_count()
            }
        
        return {
            "total_persists": self.metadata.persist_count,
            "total_loads": self.metadata.load_count,
            "last_persist": self.metadata.last_persist_time.isoformat(),
            "last_load": self.metadata.last_load_time.isoformat() if self.metadata.last_load_time else None,
            "current_vector_count": self.vector_store.get_vector_count(),
            "metadata_vector_count": self.metadata.vector_count,
            "storage_type": self.metadata.storage_type,
            "storage_path": self.metadata.storage_path
        }
    
    def validate_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of persisted data.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "current_vector_count": self.vector_store.get_vector_count(),
            "metadata_vector_count": None,
            "last_persist_age_seconds": None
        }
        
        if not self.metadata:
            validation_result["warnings"].append("No persistence metadata found")
            return validation_result
        
        validation_result["metadata_vector_count"] = self.metadata.vector_count
        
        # Check vector count consistency
        current_count = self.vector_store.get_vector_count()
        if current_count != self.metadata.vector_count:
            validation_result["warnings"].append(
                f"Vector count mismatch: current={current_count}, metadata={self.metadata.vector_count}"
            )
        
        # Check last persist age
        if self.metadata.last_persist_time:
            age_seconds = (datetime.utcnow() - self.metadata.last_persist_time).total_seconds()
            validation_result["last_persist_age_seconds"] = age_seconds
            
            # Warn if data hasn't been persisted recently and there are pending changes
            if age_seconds > 3600 and self.pending_changes:  # 1 hour
                validation_result["warnings"].append(
                    f"Data hasn't been persisted for {age_seconds/3600:.1f} hours with pending changes"
                )
        
        # Check storage consistency
        if self.metadata.storage_type != self.config.storage_type:
            validation_result["issues"].append(
                f"Storage type mismatch: metadata={self.metadata.storage_type}, config={self.config.storage_type}"
            )
            validation_result["is_valid"] = False
        
        if self.metadata.storage_path != self.config.storage_path:
            validation_result["issues"].append(
                f"Storage path mismatch: metadata={self.metadata.storage_path}, config={self.config.storage_path}"
            )
            validation_result["is_valid"] = False
        
        return validation_result
    
    def cleanup_metadata(self) -> bool:
        """
        Clean up persistence metadata files.
        
        Returns:
            True if cleanup was successful
        """
        try:
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            self.metadata = None
            return True
        except Exception:
            return False
    
    def create_backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the current state.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to the created backup
            
        Raises:
            StorageException: If backup creation fails
        """
        try:
            if backup_path is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                if self.config.storage_type == "local":
                    base_path = Path(self.config.storage_path)
                    backup_path = str(base_path.parent / f"{base_path.name}_backup_{timestamp}")
                else:
                    backup_path = f"{self.config.storage_path}_backup_{timestamp}"
            
            # For now, this is a placeholder implementation
            # In a full implementation, you would copy all the vector store data
            # to the backup location
            
            # Force a persist to ensure current state is saved
            self.persist(force=True)
            
            return backup_path
            
        except Exception as e:
            raise StorageException(f"Backup creation failed: {str(e)}")
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_path: Path to the backup to restore from
            
        Returns:
            True if restore was successful
            
        Raises:
            StorageException: If restore fails
        """
        try:
            # This is a placeholder implementation
            # In a full implementation, you would restore all vector store data
            # from the backup location
            
            # After restore, reload the data
            success = self.load()
            
            if success:
                self._update_metadata("load")
            
            return success
            
        except Exception as e:
            raise StorageException(f"Restore from backup failed: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure data is persisted."""
        if self.pending_changes:
            try:
                self.persist()
            except Exception:
                # Don't raise exceptions in __exit__
                pass