"""
Unit tests for S3VectorStore.
"""

import pytest
import json
import pickle
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError

from langchain_vector_db.storage.s3 import S3VectorStore
from langchain_vector_db.models.document import Document
from langchain_vector_db.exceptions import StorageException, ConfigurationException, S3Exception


class TestS3VectorStore:
    """Test cases for S3VectorStore."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.bucket_name = "test-bucket"
        self.s3_prefix = "test-vectors/"
        
        # Mock S3 client
        self.mock_s3_client = Mock()
        self.mock_s3_client.head_bucket.return_value = {}
        
        # Mock FAISS index
        self.mock_index = Mock()
        self.mock_index.is_trained = True
        self.mock_index.add = Mock()
        self.mock_index.search = Mock(return_value=([0.5, 0.3], [0, 1]))
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_initialization_with_defaults(self, mock_faiss, mock_session):
        """Test store initialization with default parameters."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        assert store.bucket_name == self.bucket_name
        assert store.s3_prefix == "vectors/"
        assert store.aws_region == "us-east-1"
        assert store.dimension is None
        assert store.index_type == "flat"
        assert store.metric == "l2"
        assert store.max_retries == 3
        assert store.retry_delay == 1.0
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_initialization_with_custom_parameters(self, mock_faiss, mock_session):
        """Test store initialization with custom parameters."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(
            bucket_name=self.bucket_name,
            s3_prefix="custom/",
            aws_region="us-west-2",
            dimension=384,
            index_type="ivf",
            metric="ip",
            max_retries=5,
            retry_delay=2.0
        )
        
        assert store.s3_prefix == "custom/"
        assert store.aws_region == "us-west-2"
        assert store.dimension == 384
        assert store.index_type == "ivf"
        assert store.metric == "ip"
        assert store.max_retries == 5
        assert store.retry_delay == 2.0
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_s3_prefix_normalization(self, mock_faiss, mock_session):
        """Test S3 prefix normalization (adds trailing slash)."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name, s3_prefix="vectors")
        
        assert store.s3_prefix == "vectors/"
    
    def test_empty_bucket_name(self):
        """Test initialization with empty bucket name."""
        with pytest.raises(ConfigurationException) as exc_info:
            S3VectorStore("")
        
        assert "bucket_name cannot be empty" in str(exc_info.value)
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_invalid_index_type(self, mock_faiss, mock_session):
        """Test initialization with invalid index type."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        with pytest.raises(ConfigurationException) as exc_info:
            S3VectorStore(self.bucket_name, index_type="invalid")
        
        assert "Invalid index_type" in str(exc_info.value)
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_invalid_metric(self, mock_faiss, mock_session):
        """Test initialization with invalid metric."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        with pytest.raises(ConfigurationException) as exc_info:
            S3VectorStore(self.bucket_name, metric="invalid")
        
        assert "Invalid metric" in str(exc_info.value)
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    def test_s3_connection_validation_bucket_not_found(self, mock_session):
        """Test S3 connection validation with non-existent bucket."""
        mock_s3_client = Mock()
        mock_s3_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404'}}, 'HeadBucket'
        )
        mock_session.return_value.client.return_value = mock_s3_client
        
        with pytest.raises(S3Exception) as exc_info:
            S3VectorStore(self.bucket_name)
        
        assert "does not exist" in str(exc_info.value)
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    def test_s3_connection_validation_access_denied(self, mock_session):
        """Test S3 connection validation with access denied."""
        mock_s3_client = Mock()
        mock_s3_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '403'}}, 'HeadBucket'
        )
        mock_session.return_value.client.return_value = mock_s3_client
        
        with pytest.raises(S3Exception) as exc_info:
            S3VectorStore(self.bucket_name)
        
        assert "Access denied" in str(exc_info.value)
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    def test_s3_connection_validation_no_credentials(self, mock_session):
        """Test S3 connection validation with no credentials."""
        mock_s3_client = Mock()
        mock_s3_client.head_bucket.side_effect = NoCredentialsError()
        mock_session.return_value.client.return_value = mock_s3_client
        
        with pytest.raises(S3Exception) as exc_info:
            S3VectorStore(self.bucket_name)
        
        assert "credentials not found" in str(exc_info.value)
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_add_vectors_basic(self, mock_faiss, mock_session):
        """Test basic vector addition."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        mock_faiss.IndexFlatL2.return_value = self.mock_index
        
        store = S3VectorStore(self.bucket_name)
        
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        documents = [
            Document(page_content="Document 1", doc_id="doc1"),
            Document(page_content="Document 2", doc_id="doc2")
        ]
        
        # Mock persist operation
        store.persist = Mock(return_value=True)
        
        doc_ids = store.add_vectors(vectors, documents)
        
        assert len(doc_ids) == 2
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert store.get_vector_count() == 2
        assert store.dimension == 3
        
        # Verify FAISS operations
        self.mock_index.add.assert_called_once()
        store.persist.assert_called_once()
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_add_vectors_empty_lists(self, mock_faiss, mock_session):
        """Test adding empty vector lists."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        doc_ids = store.add_vectors([], [])
        
        assert doc_ids == []
        assert store.get_vector_count() == 0
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_add_vectors_mismatched_lengths(self, mock_faiss, mock_session):
        """Test adding vectors with mismatched document count."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        documents = [Document(page_content="Document 1", doc_id="doc1")]
        
        with pytest.raises(StorageException) as exc_info:
            store.add_vectors(vectors, documents)
        
        assert "must match number of documents" in str(exc_info.value)
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_search_vectors_basic(self, mock_faiss, mock_session):
        """Test basic vector search."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        mock_faiss.IndexFlatL2.return_value = self.mock_index
        
        store = S3VectorStore(self.bucket_name)
        store.dimension = 3
        store.index = self.mock_index
        
        # Add some test documents
        store.documents = {
            "doc1": Document(page_content="Document 1", doc_id="doc1"),
            "doc2": Document(page_content="Document 2", doc_id="doc2")
        }
        store.index_to_id = {0: "doc1", 1: "doc2"}
        
        # Mock search results
        self.mock_index.search.return_value = ([[0.5, 0.3]], [[0, 1]])
        
        query_vector = [1.0, 0.0, 0.0]
        results = store.search_vectors(query_vector, k=2)
        
        assert len(results) == 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        assert all(isinstance(result[0], Document) for result in results)
        assert all(isinstance(result[1], (int, float)) for result in results)
        
        # Verify FAISS search was called
        self.mock_index.search.assert_called_once()
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_search_vectors_empty_store(self, mock_faiss, mock_session):
        """Test searching in empty store."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        query_vector = [1.0, 2.0, 3.0]
        results = store.search_vectors(query_vector, k=5)
        
        assert results == []
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_search_vectors_dimension_mismatch(self, mock_faiss, mock_session):
        """Test searching with wrong dimension."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        store.dimension = 3
        store.index = self.mock_index
        store.documents = {"doc1": Document(page_content="Document 1", doc_id="doc1")}
        
        # Search with wrong dimension
        query_vector = [1.0, 2.0]  # Wrong dimension
        
        with pytest.raises(StorageException) as exc_info:
            store.search_vectors(query_vector, k=1)
        
        assert "doesn't match index dimension" in str(exc_info.value)
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_get_document(self, mock_faiss, mock_session):
        """Test retrieving documents by ID."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        # Add a test document
        test_doc = Document(page_content="Test document", doc_id="doc1")
        store.documents["doc1"] = test_doc
        
        # Retrieve the document
        retrieved_doc = store.get_document("doc1")
        
        assert retrieved_doc is not None
        assert retrieved_doc.page_content == "Test document"
        assert retrieved_doc.doc_id == "doc1"
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_get_nonexistent_document(self, mock_faiss, mock_session):
        """Test retrieving non-existent document."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        retrieved_doc = store.get_document("nonexistent")
        
        assert retrieved_doc is None
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_update_vector(self, mock_faiss, mock_session):
        """Test updating an existing vector."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        # Add a test document
        original_doc = Document(page_content="Original", doc_id="doc1")
        store.documents["doc1"] = original_doc
        
        # Mock persist operation
        store.persist = Mock(return_value=True)
        
        # Update the vector
        new_vector = [4.0, 5.0, 6.0]
        new_document = Document(page_content="Updated", doc_id="doc1")
        
        result = store.update_vector("doc1", new_vector, new_document)
        
        assert result is True
        
        # Verify the document was updated
        retrieved_doc = store.get_document("doc1")
        assert retrieved_doc.page_content == "Updated"
        
        # Verify persist was called
        store.persist.assert_called_once()
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_update_nonexistent_vector(self, mock_faiss, mock_session):
        """Test updating non-existent vector."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        new_vector = [1.0, 2.0, 3.0]
        new_document = Document(page_content="New", doc_id="nonexistent")
        
        result = store.update_vector("nonexistent", new_vector, new_document)
        
        assert result is False
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_delete_vectors(self, mock_faiss, mock_session):
        """Test deleting vectors."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        # Add some test documents
        store.documents = {
            "doc1": Document(page_content="Document 1", doc_id="doc1"),
            "doc2": Document(page_content="Document 2", doc_id="doc2")
        }
        store.id_to_index = {"doc1": 0, "doc2": 1}
        store.index_to_id = {0: "doc1", 1: "doc2"}
        
        # Mock persist operation
        store.persist = Mock(return_value=True)
        
        # Delete one vector
        result = store.delete_vectors(["doc1"])
        
        assert result is True
        assert store.get_vector_count() == 1
        assert store.get_document("doc1") is None
        assert store.get_document("doc2") is not None
        
        # Verify persist was called
        store.persist.assert_called_once()
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_persist_operation(self, mock_faiss, mock_session):
        """Test persisting data to S3."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        mock_faiss.write_index = Mock()
        
        store = S3VectorStore(self.bucket_name)
        store.index = self.mock_index
        store.dimension = 3
        store.documents = {"doc1": Document(page_content="Test", doc_id="doc1")}
        
        result = store.persist()
        
        assert result is True
        
        # Verify S3 uploads were called
        assert self.mock_s3_client.put_object.call_count == 3  # index, metadata, documents
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_load_operation(self, mock_faiss, mock_session):
        """Test loading data from S3."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        mock_faiss.read_index = Mock(return_value=self.mock_index)
        
        store = S3VectorStore(self.bucket_name)
        
        # Mock S3 key existence checks
        store._key_exists_in_s3 = Mock(return_value=True)
        
        # Mock S3 downloads
        metadata = {
            "dimension": 3,
            "index_type": "flat",
            "metric": "l2",
            "next_index": 1,
            "id_to_index": {"doc1": 0},
            "index_to_id": {"0": "doc1"}
        }
        documents = {"doc1": Document(page_content="Test", doc_id="doc1")}
        
        store._download_from_s3 = Mock(side_effect=[
            b"fake_index_data",  # index data
            json.dumps(metadata).encode('utf-8'),  # metadata
            pickle.dumps(documents)  # documents
        ])
        
        result = store.load()
        
        assert result is True
        assert store.dimension == 3
        assert store.get_vector_count() == 1
        assert "doc1" in store.documents
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_load_missing_files(self, mock_faiss, mock_session):
        """Test loading when S3 files don't exist."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        # Mock S3 key existence checks to return False
        store._key_exists_in_s3 = Mock(return_value=False)
        
        result = store.load()
        
        assert result is True  # Should succeed even with missing files
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_health_check_healthy(self, mock_faiss, mock_session):
        """Test health check when store is healthy."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        # Mock successful S3 operations
        self.mock_s3_client.delete_object.return_value = {}
        
        assert store.health_check() is True
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_health_check_unhealthy(self, mock_faiss, mock_session):
        """Test health check when store is unhealthy."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        # Mock S3 operation failure
        self.mock_s3_client.head_bucket.side_effect = Exception("S3 error")
        
        assert store.health_check() is False
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_get_s3_info(self, mock_faiss, mock_session):
        """Test getting S3 storage information."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name, dimension=384)
        
        info = store.get_s3_info()
        
        assert "bucket_name" in info
        assert "s3_prefix" in info
        assert "aws_region" in info
        assert "dimension" in info
        assert "index_type" in info
        assert "metric" in info
        assert "vector_count" in info
        assert info["bucket_name"] == self.bucket_name
        assert info["dimension"] == 384
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_list_s3_objects(self, mock_faiss, mock_session):
        """Test listing S3 objects."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name)
        
        # Mock S3 list response
        self.mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {
                    'Key': 'vectors/faiss_index.bin',
                    'Size': 1024,
                    'LastModified': '2023-01-01T00:00:00Z',
                    'ETag': '"abc123"'
                }
            ]
        }
        
        objects = store.list_s3_objects()
        
        assert len(objects) == 1
        assert objects[0]['key'] == 'vectors/faiss_index.bin'
        assert objects[0]['size'] == 1024
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_retry_mechanism(self, mock_faiss, mock_session):
        """Test S3 operation retry mechanism."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name, max_retries=2, retry_delay=0.1)
        
        # Mock S3 operation that fails twice then succeeds
        self.mock_s3_client.put_object.side_effect = [
            ClientError({'Error': {'Code': '500'}}, 'PutObject'),
            ClientError({'Error': {'Code': '500'}}, 'PutObject'),
            {}  # Success on third try
        ]
        
        # Should succeed after retries
        store._upload_to_s3("test-key", b"test-data")
        
        # Verify it was called 3 times (initial + 2 retries)
        assert self.mock_s3_client.put_object.call_count == 3
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_retry_mechanism_exhausted(self, mock_faiss, mock_session):
        """Test S3 operation retry mechanism when retries are exhausted."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name, max_retries=1, retry_delay=0.1)
        
        # Mock S3 operation that always fails
        self.mock_s3_client.put_object.side_effect = ClientError(
            {'Error': {'Code': '500'}}, 'PutObject'
        )
        
        # Should raise S3Exception after exhausting retries
        with pytest.raises(S3Exception) as exc_info:
            store._upload_to_s3("test-key", b"test-data")
        
        assert "failed after" in str(exc_info.value)
        
        # Verify it was called max_retries + 1 times
        assert self.mock_s3_client.put_object.call_count == 2
    
    @patch('langchain_vector_db.storage.s3.boto3.Session')
    @patch('langchain_vector_db.storage.s3.faiss')
    def test_string_representations(self, mock_faiss, mock_session):
        """Test string representations of the store."""
        mock_session.return_value.client.return_value = self.mock_s3_client
        
        store = S3VectorStore(self.bucket_name, dimension=384)
        
        str_repr = str(store)
        assert "S3VectorStore" in str_repr
        assert self.bucket_name in str_repr
        assert "384" in str_repr
        
        repr_str = repr(store)
        assert "S3VectorStore" in repr_str
        assert "flat" in repr_str
        assert "l2" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])