"""
Unit tests for PersistenceManager.
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_vector_db.persistence import PersistenceManager, PersistenceMetadata
from langchain_vector_db.models.config import VectorDBConfig
from langchain_vector_db.exceptions import StorageException


class TestPersistenceMetadata:
    """Test cases for PersistenceMetadata."""
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        now = datetime.utcnow()
        metadata = PersistenceMetadata(
            last_persist_time=now,
            last_load_time=now,
            persist_count=5,
            load_count=3,
            storage_type="local",
            storage_path="./test_db",
            vector_count=100,
            embedding_dimension=384
        )
        
        # Convert to dict
        data = metadata.to_dict()
        
        assert data["last_persist_time"] == now.isoformat()
        assert data["last_load_time"] == now.isoformat()
        assert data["persist_count"] == 5
        assert data["load_count"] == 3
        
        # Convert back from dict
        restored_metadata = PersistenceMetadata.from_dict(data)
        
        assert restored_metadata.last_persist_time == now
        assert restored_metadata.last_load_time == now
        assert restored_metadata.persist_count == 5
        assert restored_metadata.load_count == 3
        assert restored_metadata.storage_type == "local"
        assert restored_metadata.vector_count == 100
    
    def test_from_dict_with_none_load_time(self):
        """Test deserialization with None load time."""
        now = datetime.utcnow()
        data = {
            "last_persist_time": now.isoformat(),
            "last_load_time": None,
            "persist_count": 1,
            "load_count": 0,
            "storage_type": "s3",
            "storage_path": "test-bucket",
            "vector_count": 50,
            "embedding_dimension": 512
        }
        
        metadata = PersistenceMetadata.from_dict(data)
        
        assert metadata.last_persist_time == now
        assert metadata.last_load_time is None
        assert metadata.persist_count == 1
        assert metadata.load_count == 0


class TestPersistenceManager:
    """Test cases for PersistenceManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_vector_store = Mock()
        self.mock_vector_store.persist.return_value = True
        self.mock_vector_store.load.return_value = True
        self.mock_vector_store.get_vector_count.return_value = 10
        
        self.config = VectorDBConfig(
            storage_type="local",
            embedding_model="huggingface",
            storage_path="./test_db"
        )
        
        self.temp_dir = tempfile.mkdtemp()
        self.config.storage_path = self.temp_dir
    
    def test_initialization(self):
        """Test persistence manager initialization."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config,
            auto_persist=True,
            persist_interval=300
        )
        
        assert manager.vector_store == self.mock_vector_store
        assert manager.config == self.config
        assert manager.auto_persist is True
        assert manager.persist_interval == 300
        assert manager.pending_changes is False
    
    def test_get_metadata_path_local(self):
        """Test metadata path generation for local storage."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        expected_path = Path(self.temp_dir) / ".persistence_metadata.json"
        assert manager.metadata_path == expected_path
    
    def test_get_metadata_path_s3(self):
        """Test metadata path generation for S3 storage."""
        s3_config = VectorDBConfig(
            storage_type="s3",
            embedding_model="openai",
            storage_path="test-bucket"
        )
        
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=s3_config
        )
        
        # Should create path in home directory
        assert "test-bucket" in str(manager.metadata_path)
        assert ".langchain_vector_db" in str(manager.metadata_path)
    
    def test_persist_success(self):
        """Test successful persistence."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        manager.pending_changes = True
        result = manager.persist()
        
        assert result is True
        assert manager.pending_changes is False
        self.mock_vector_store.persist.assert_called_once()
        
        # Check that metadata was created
        assert manager.metadata is not None
        assert manager.metadata.persist_count == 1
        assert manager.metadata.storage_type == "local"
    
    def test_persist_no_changes(self):
        """Test persistence when no changes are pending."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        manager.pending_changes = False
        result = manager.persist()
        
        assert result is True
        # Should not call vector store persist
        self.mock_vector_store.persist.assert_not_called()
    
    def test_persist_force(self):
        """Test forced persistence."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        manager.pending_changes = False
        result = manager.persist(force=True)
        
        assert result is True
        # Should call vector store persist even without pending changes
        self.mock_vector_store.persist.assert_called_once()
    
    def test_persist_failure(self):
        """Test persistence failure."""
        self.mock_vector_store.persist.return_value = False
        
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        manager.pending_changes = True
        result = manager.persist()
        
        assert result is False
        # Pending changes should remain true on failure
        assert manager.pending_changes is True
    
    def test_load_success(self):
        """Test successful loading."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        result = manager.load()
        
        assert result is True
        assert manager.pending_changes is False
        self.mock_vector_store.load.assert_called_once()
        
        # Check that metadata was created
        assert manager.metadata is not None
        assert manager.metadata.load_count == 1
    
    def test_load_failure(self):
        """Test loading failure."""
        self.mock_vector_store.load.return_value = False
        
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        result = manager.load()
        
        assert result is False
    
    def test_mark_changes_pending(self):
        """Test marking changes as pending."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config,
            auto_persist=False  # Disable auto-persist for this test
        )
        
        assert manager.pending_changes is False
        
        manager.mark_changes_pending()
        
        assert manager.pending_changes is True
    
    @patch('time.time')
    def test_auto_persist_triggered(self, mock_time):
        """Test that auto-persist is triggered when interval passes."""
        # Mock time to simulate interval passing
        mock_time.side_effect = [0, 0, 400]  # Start, last_auto_persist, current time
        
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config,
            auto_persist=True,
            persist_interval=300  # 5 minutes
        )
        
        manager.mark_changes_pending()
        
        # Should have triggered auto-persist
        self.mock_vector_store.persist.assert_called_once()
    
    def test_get_persistence_info(self):
        """Test getting persistence information."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config,
            auto_persist=True,
            persist_interval=300
        )
        
        info = manager.get_persistence_info()
        
        assert "auto_persist_enabled" in info
        assert "persist_interval" in info
        assert "pending_changes" in info
        assert "metadata_path" in info
        assert info["auto_persist_enabled"] is True
        assert info["persist_interval"] == 300
    
    def test_get_persistence_stats_no_metadata(self):
        """Test getting persistence stats when no metadata exists."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        stats = manager.get_persistence_stats()
        
        assert stats["total_persists"] == 0
        assert stats["total_loads"] == 0
        assert stats["last_persist"] is None
        assert stats["last_load"] is None
        assert stats["current_vector_count"] == 10
    
    def test_get_persistence_stats_with_metadata(self):
        """Test getting persistence stats with existing metadata."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        # Create some metadata
        manager.persist()
        manager.load()
        
        stats = manager.get_persistence_stats()
        
        assert stats["total_persists"] == 1
        assert stats["total_loads"] == 1
        assert stats["last_persist"] is not None
        assert stats["last_load"] is not None
        assert stats["storage_type"] == "local"
    
    def test_validate_integrity_no_metadata(self):
        """Test integrity validation without metadata."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        result = manager.validate_integrity()
        
        assert result["is_valid"] is True
        assert "No persistence metadata found" in result["warnings"]
        assert result["current_vector_count"] == 10
    
    def test_validate_integrity_with_metadata(self):
        """Test integrity validation with metadata."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        # Create metadata
        manager.persist()
        
        result = manager.validate_integrity()
        
        assert result["is_valid"] is True
        assert result["current_vector_count"] == 10
        assert result["metadata_vector_count"] == 10
    
    def test_validate_integrity_vector_count_mismatch(self):
        """Test integrity validation with vector count mismatch."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        # Create metadata with different vector count
        manager.persist()
        
        # Change vector count
        self.mock_vector_store.get_vector_count.return_value = 20
        
        result = manager.validate_integrity()
        
        assert result["is_valid"] is True  # Still valid, just a warning
        assert any("Vector count mismatch" in warning for warning in result["warnings"])
    
    def test_validate_integrity_storage_type_mismatch(self):
        """Test integrity validation with storage type mismatch."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        # Create metadata
        manager.persist()
        
        # Change config storage type
        manager.config.storage_type = "s3"
        
        result = manager.validate_integrity()
        
        assert result["is_valid"] is False
        assert any("Storage type mismatch" in issue for issue in result["issues"])
    
    def test_cleanup_metadata(self):
        """Test metadata cleanup."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        # Create metadata
        manager.persist()
        assert manager.metadata is not None
        
        # Cleanup
        result = manager.cleanup_metadata()
        
        assert result is True
        assert manager.metadata is None
    
    def test_context_manager(self):
        """Test using persistence manager as context manager."""
        manager = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        with manager as pm:
            pm.mark_changes_pending()
            assert pm.pending_changes is True
        
        # Should have persisted on exit
        self.mock_vector_store.persist.assert_called_once()
    
    def test_metadata_persistence_across_instances(self):
        """Test that metadata persists across manager instances."""
        # Create first manager and persist
        manager1 = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        manager1.persist()
        
        # Create second manager - should load existing metadata
        manager2 = PersistenceManager(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        assert manager2.metadata is not None
        assert manager2.metadata.persist_count == 1


if __name__ == "__main__":
    pytest.main([__file__])