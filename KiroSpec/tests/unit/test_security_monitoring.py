"""
Unit tests for SecurityMonitoringService.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch

from langchain_vector_db.services.security_monitoring import (
    SecurityMonitoringService, SecurityAlert, ThreatIndicator
)
from langchain_vector_db.models.auth import AuditEvent


class TestSecurityMonitoringService:
    """Test cases for SecurityMonitoringService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = SecurityMonitoringService(
            enable_monitoring=True,
            alert_retention_hours=24,
            threat_detection_window_minutes=60
        )
    
    def test_initialization(self):
        """Test security monitoring service initialization."""
        service = SecurityMonitoringService()
        
        assert service.enable_monitoring is True
        assert service.alert_retention_hours == 168  # 7 days default
        assert service.threat_detection_window_minutes == 60
        assert len(service._alerts) == 0
        assert len(service._threat_indicators) == 0
    
    def test_process_failed_authentication_brute_force(self):
        """Test detection of brute force attacks."""
        user_id = "test_user"
        
        # Generate multiple failed authentication events
        for i in range(12):  # Above threshold of 10
            event = AuditEvent(
                timestamp=datetime.utcnow(),
                user_id=user_id,
                operation="authentication",
                resource="api_key",
                status="failed",
                details={"reason": "invalid_api_key"},
                correlation_id=f"test_{i}"
            )
            
            alerts = self.service.process_audit_event(event)
        
        # Should generate brute force alert
        brute_force_alerts = [
            alert for alert in self.service._alerts 
            if alert.alert_type == "brute_force_attack"
        ]
        
        assert len(brute_force_alerts) >= 1
        alert = brute_force_alerts[0]
        assert alert.severity == "high"
        assert alert.user_id == user_id
        assert "brute force" in alert.description.lower()
    
    def test_process_failed_authentication_rapid_attempts(self):
        """Test detection of rapid failed authentication attempts."""
        user_id = "rapid_user"
        
        # Generate rapid failed attempts (within 1 minute)
        for i in range(6):  # Above threshold of 5
            event = AuditEvent(
                timestamp=datetime.utcnow(),
                user_id=user_id,
                operation="authentication",
                resource="jwt",
                status="failed",
                details={"reason": "invalid_token"},
                correlation_id=f"rapid_{i}"
            )
            
            alerts = self.service.process_audit_event(event)
        
        # Should generate rapid attempt alert
        rapid_alerts = [
            alert for alert in self.service._alerts 
            if alert.alert_type == "rapid_failed_auth"
        ]
        
        assert len(rapid_alerts) >= 1
        alert = rapid_alerts[0]
        assert alert.severity == "medium"
        assert alert.user_id == user_id
    
    def test_process_privilege_escalation_attempt(self):
        """Test detection of privilege escalation attempts."""
        user_id = "escalation_user"
        
        # Generate multiple failed authorization events
        for i in range(16):  # Above threshold of 15
            event = AuditEvent(
                timestamp=datetime.utcnow(),
                user_id=user_id,
                operation="authorization",
                resource="system.manage",
                status="failed",
                details={"reason": "insufficient_permissions", "required_permission": "system.manage"},
                correlation_id=f"escalation_{i}"
            )
            
            alerts = self.service.process_audit_event(event)
        
        # Should generate privilege escalation alert
        escalation_alerts = [
            alert for alert in self.service._alerts 
            if alert.alert_type == "privilege_escalation_attempt"
        ]
        
        assert len(escalation_alerts) >= 1
        alert = escalation_alerts[0]
        assert alert.severity == "high"
        assert alert.user_id == user_id
    
    def test_process_high_risk_pii_detection(self):
        """Test detection of high-risk PII."""
        user_id = "pii_user"
        
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            operation="pii_detection",
            resource="document",
            status="success",
            details={
                "matches_found": 3,
                "pii_types": ["email", "ssn", "credit_card"]  # Contains high-risk types
            },
            correlation_id="pii_test"
        )
        
        alerts = self.service.process_audit_event(event)
        
        # Should generate high-risk PII alert
        pii_alerts = [
            alert for alert in self.service._alerts 
            if alert.alert_type == "high_risk_pii_detected"
        ]
        
        assert len(pii_alerts) >= 1
        alert = pii_alerts[0]
        assert alert.severity == "medium"
        assert alert.user_id == user_id
        assert "ssn" in alert.details["high_risk_pii_types"]
        assert "credit_card" in alert.details["high_risk_pii_types"]
    
    def test_process_rate_limit_violations(self):
        """Test detection of rate limit abuse."""
        user_id = "rate_limit_user"
        
        # Generate multiple rate limit violations
        for i in range(22):  # Above threshold of 20
            event = AuditEvent(
                timestamp=datetime.utcnow(),
                user_id=user_id,
                operation="authorization",
                resource="documents.read",
                status="failed",
                details={"reason": "rate_limit_exceeded"},
                correlation_id=f"rate_limit_{i}"
            )
            
            alerts = self.service.process_audit_event(event)
        
        # Should generate rate limit abuse alert
        rate_limit_alerts = [
            alert for alert in self.service._alerts 
            if alert.alert_type == "rate_limit_abuse"
        ]
        
        assert len(rate_limit_alerts) >= 1
        alert = rate_limit_alerts[0]
        assert alert.severity == "medium"
        assert alert.user_id == user_id
    
    def test_threat_indicator_creation(self):
        """Test creation and updating of threat indicators."""
        # Create event with IP address
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            user_id="test_user",
            operation="authentication",
            resource="api_key",
            status="failed",
            details={"ip_address": "192.168.1.100", "user_agent": "TestAgent/1.0"},
            correlation_id="threat_test"
        )
        
        self.service.process_audit_event(event)
        
        # Should create threat indicators
        indicators = self.service.get_threat_indicators()
        
        ip_indicators = [ind for ind in indicators if ind.indicator_type == "ip_address"]
        assert len(ip_indicators) >= 1
        
        ip_indicator = ip_indicators[0]
        assert ip_indicator.value == "192.168.1.100"
        assert ip_indicator.severity == "medium"
        assert ip_indicator.count == 1
    
    def test_threat_indicator_escalation(self):
        """Test threat indicator severity escalation."""
        # Create multiple events from same IP
        for i in range(15):
            event = AuditEvent(
                timestamp=datetime.utcnow(),
                user_id=f"user_{i}",
                operation="authentication",
                resource="api_key",
                status="failed",
                details={"ip_address": "192.168.1.200"},
                correlation_id=f"escalation_{i}"
            )
            
            self.service.process_audit_event(event)
        
        # Check if severity escalated
        indicators = self.service.get_threat_indicators()
        ip_indicators = [ind for ind in indicators if ind.value == "192.168.1.200"]
        
        assert len(ip_indicators) == 1
        indicator = ip_indicators[0]
        assert indicator.count >= 15
        # Severity should escalate based on count
        assert indicator.severity in ["medium", "high"]
    
    def test_get_active_alerts_filtering(self):
        """Test filtering of active alerts."""
        # Create alerts with different severities and types
        alert1 = SecurityAlert(
            alert_id="alert1",
            alert_type="brute_force_attack",
            severity="high",
            timestamp=datetime.utcnow(),
            user_id="user1",
            description="Test alert 1"
        )
        
        alert2 = SecurityAlert(
            alert_id="alert2",
            alert_type="rate_limit_abuse",
            severity="medium",
            timestamp=datetime.utcnow(),
            user_id="user2",
            description="Test alert 2"
        )
        
        self.service._alerts.extend([alert1, alert2])
        
        # Test severity filtering
        high_alerts = self.service.get_active_alerts(severity="high")
        assert len(high_alerts) == 1
        assert high_alerts[0].alert_id == "alert1"
        
        # Test type filtering
        brute_force_alerts = self.service.get_active_alerts(alert_type="brute_force_attack")
        assert len(brute_force_alerts) == 1
        assert brute_force_alerts[0].alert_id == "alert1"
    
    def test_resolve_alert(self):
        """Test resolving security alerts."""
        alert = SecurityAlert(
            alert_id="test_alert",
            alert_type="test_type",
            severity="medium",
            timestamp=datetime.utcnow(),
            user_id="test_user",
            description="Test alert"
        )
        
        self.service._alerts.append(alert)
        
        # Resolve the alert
        result = self.service.resolve_alert("test_alert", "admin")
        
        assert result is True
        assert alert.resolved is True
        assert alert.resolved_at is not None
        assert alert.details["resolved_by"] == "admin"
        
        # Try to resolve non-existent alert
        result = self.service.resolve_alert("non_existent", "admin")
        assert result is False
    
    def test_ip_blocking(self):
        """Test IP address blocking functionality."""
        ip_address = "192.168.1.50"
        
        # Initially not blocked
        assert self.service.is_ip_blocked(ip_address) is False
        
        # Block IP
        self.service.block_ip(ip_address, "Security violation")
        
        # Should be blocked now
        assert self.service.is_ip_blocked(ip_address) is True
        
        # Should create alert for blocking
        block_alerts = [
            alert for alert in self.service._alerts 
            if alert.alert_type == "ip_blocked"
        ]
        assert len(block_alerts) == 1
        
        # Unblock IP
        result = self.service.unblock_ip(ip_address)
        assert result is True
        assert self.service.is_ip_blocked(ip_address) is False
        
        # Try to unblock non-blocked IP
        result = self.service.unblock_ip("192.168.1.99")
        assert result is False
    
    def test_get_security_metrics(self):
        """Test getting security metrics."""
        # Add some test data
        self.service._failed_auth_attempts["user1"] = [datetime.utcnow()]
        self.service._suspicious_activities["user2"] = [datetime.utcnow()]
        self.service._blocked_ips.add("192.168.1.1")
        
        metrics = self.service.get_security_metrics()
        
        assert "monitoring_enabled" in metrics
        assert "active_alerts" in metrics
        assert "blocked_ips" in metrics
        assert "threat_indicators" in metrics
        assert "recent_failed_auth" in metrics
        assert "thresholds" in metrics
        
        assert metrics["monitoring_enabled"] is True
        assert metrics["blocked_ips"] == 1
    
    def test_update_thresholds(self):
        """Test updating security thresholds."""
        original_threshold = self.service.thresholds["failed_auth_per_hour"]
        
        new_thresholds = {"failed_auth_per_hour": 20}
        self.service.update_thresholds(new_thresholds)
        
        assert self.service.thresholds["failed_auth_per_hour"] == 20
        assert self.service.thresholds["failed_auth_per_hour"] != original_threshold
        
        # Test invalid threshold (should be ignored)
        self.service.update_thresholds({"invalid_threshold": 10})
        assert "invalid_threshold" not in self.service.thresholds
    
    def test_get_user_risk_score(self):
        """Test user risk score calculation."""
        user_id = "risky_user"
        
        # Add some risky activities
        now = datetime.utcnow()
        self.service._failed_auth_attempts[user_id] = [now, now - timedelta(minutes=30)]
        self.service._suspicious_activities[user_id] = [now]
        self.service._rate_limit_violations[user_id] = [now, now - timedelta(minutes=15)]
        
        risk_assessment = self.service.get_user_risk_score(user_id)
        
        assert "user_id" in risk_assessment
        assert "risk_score" in risk_assessment
        assert "risk_level" in risk_assessment
        assert "risk_factors" in risk_assessment
        
        assert risk_assessment["user_id"] == user_id
        assert risk_assessment["risk_score"] > 0
        assert risk_assessment["risk_level"] in ["low", "medium", "high", "critical"]
        assert len(risk_assessment["risk_factors"]) > 0
    
    def test_cleanup_old_data(self):
        """Test cleanup of old monitoring data."""
        # Add old data
        old_time = datetime.utcnow() - timedelta(hours=25)  # Older than retention
        recent_time = datetime.utcnow()
        
        # Add old alert
        old_alert = SecurityAlert(
            alert_id="old_alert",
            alert_type="test",
            severity="low",
            timestamp=old_time,
            user_id="test_user",
            description="Old alert"
        )
        
        # Add recent alert
        recent_alert = SecurityAlert(
            alert_id="recent_alert",
            alert_type="test",
            severity="low",
            timestamp=recent_time,
            user_id="test_user",
            description="Recent alert"
        )
        
        self.service._alerts.extend([old_alert, recent_alert])
        
        # Add old tracking data
        self.service._failed_auth_attempts["user1"] = [old_time, recent_time]
        
        # Trigger cleanup
        self.service._cleanup_old_data()
        
        # Old alert should be removed
        alert_ids = [alert.alert_id for alert in self.service._alerts]
        assert "old_alert" not in alert_ids
        assert "recent_alert" in alert_ids
        
        # Old tracking data should be removed
        assert len(self.service._failed_auth_attempts["user1"]) == 1
        assert self.service._failed_auth_attempts["user1"][0] == recent_time
    
    def test_monitoring_disabled(self):
        """Test behavior when monitoring is disabled."""
        service = SecurityMonitoringService(enable_monitoring=False)
        
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            user_id="test_user",
            operation="authentication",
            resource="api_key",
            status="failed",
            details={},
            correlation_id="test"
        )
        
        alerts = service.process_audit_event(event)
        
        # Should not generate any alerts
        assert len(alerts) == 0
        assert len(service._alerts) == 0
    
    def test_health_check(self):
        """Test security monitoring health check."""
        result = self.service.health_check()
        assert result is True
    
    def test_health_check_failure(self):
        """Test health check failure scenario."""
        # Break the service by setting invalid state
        original_lock = self.service._lock
        self.service._lock = None
        
        result = self.service.health_check()
        
        # Restore lock
        self.service._lock = original_lock
        
        assert result is False
    
    def test_string_representations(self):
        """Test string representations of security monitoring service."""
        str_repr = str(self.service)
        assert "SecurityMonitoringService" in str_repr
        assert "enabled=True" in str_repr
        
        repr_str = repr(self.service)
        assert "SecurityMonitoringService" in repr_str
        assert "retention_hours=24" in repr_str


class TestSecurityAlert:
    """Test cases for SecurityAlert class."""
    
    def test_security_alert_creation(self):
        """Test creating a security alert."""
        alert = SecurityAlert(
            alert_id="test_alert",
            alert_type="brute_force_attack",
            severity="high",
            timestamp=datetime.utcnow(),
            user_id="test_user",
            description="Test security alert"
        )
        
        assert alert.alert_id == "test_alert"
        assert alert.alert_type == "brute_force_attack"
        assert alert.severity == "high"
        assert alert.user_id == "test_user"
        assert alert.resolved is False
        assert alert.resolved_at is None


class TestThreatIndicator:
    """Test cases for ThreatIndicator class."""
    
    def test_threat_indicator_creation(self):
        """Test creating a threat indicator."""
        now = datetime.utcnow()
        
        indicator = ThreatIndicator(
            indicator_type="ip_address",
            value="192.168.1.1",
            severity="medium",
            description="Suspicious IP address",
            first_seen=now,
            last_seen=now
        )
        
        assert indicator.indicator_type == "ip_address"
        assert indicator.value == "192.168.1.1"
        assert indicator.severity == "medium"
        assert indicator.count == 1
        assert indicator.first_seen == now
        assert indicator.last_seen == now


if __name__ == "__main__":
    pytest.main([__file__])