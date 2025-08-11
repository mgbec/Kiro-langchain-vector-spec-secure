"""
Security monitoring and intrusion detection service.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

from ..models.auth import AuditEvent
from ..exceptions import SecurityException


@dataclass
class SecurityAlert:
    """Represents a security alert."""
    
    alert_id: str
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime
    user_id: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ThreatIndicator:
    """Represents a threat indicator."""
    
    indicator_type: str
    value: str
    severity: str
    description: str
    first_seen: datetime
    last_seen: datetime
    count: int = 1


class SecurityMonitoringService:
    """Service for security monitoring and threat detection."""
    
    def __init__(
        self,
        enable_monitoring: bool = True,
        alert_retention_hours: int = 168,  # 7 days
        threat_detection_window_minutes: int = 60
    ):
        """
        Initialize security monitoring service.
        
        Args:
            enable_monitoring: Whether to enable security monitoring
            alert_retention_hours: How long to retain alerts
            threat_detection_window_minutes: Time window for threat detection
        """
        self.enable_monitoring = enable_monitoring
        self.alert_retention_hours = alert_retention_hours
        self.threat_detection_window_minutes = threat_detection_window_minutes
        
        # Storage for alerts and indicators
        self._alerts: List[SecurityAlert] = []
        self._threat_indicators: Dict[str, ThreatIndicator] = {}
        
        # Tracking for various security metrics
        self._failed_auth_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self._suspicious_activities: Dict[str, List[datetime]] = defaultdict(list)
        self._rate_limit_violations: Dict[str, List[datetime]] = defaultdict(list)
        self._blocked_ips: Set[str] = set()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Thresholds for threat detection
        self.thresholds = {
            "failed_auth_per_hour": 10,
            "failed_auth_per_minute": 5,
            "rate_limit_violations_per_hour": 20,
            "suspicious_activities_per_hour": 15,
            "unique_failed_users_per_hour": 5
        }
    
    def process_audit_event(self, event: AuditEvent) -> List[SecurityAlert]:
        """
        Process an audit event and generate security alerts if needed.
        
        Args:
            event: Audit event to process
            
        Returns:
            List of generated security alerts
        """
        if not self.enable_monitoring:
            return []
        
        alerts = []
        
        with self._lock:
            # Process different types of events
            if event.operation == "authentication" and event.status == "failed":
                alerts.extend(self._process_failed_authentication(event))
            
            elif event.operation == "authorization" and event.status == "failed":
                alerts.extend(self._process_failed_authorization(event))
            
            elif event.operation in ["pii_detection", "data_masking"] and event.status == "success":
                alerts.extend(self._process_pii_detection(event))
            
            elif "rate_limit" in event.details.get("reason", ""):
                alerts.extend(self._process_rate_limit_violation(event))
            
            # Update threat indicators
            self._update_threat_indicators(event)
            
            # Clean up old data
            self._cleanup_old_data()
        
        return alerts
    
    def _process_failed_authentication(self, event: AuditEvent) -> List[SecurityAlert]:
        """Process failed authentication events."""
        alerts = []
        user_id = event.user_id
        now = datetime.utcnow()
        
        # Track failed attempts
        self._failed_auth_attempts[user_id].append(now)
        
        # Check for brute force attacks
        recent_failures = self._get_recent_events(
            self._failed_auth_attempts[user_id], 
            minutes=60
        )
        
        if len(recent_failures) >= self.thresholds["failed_auth_per_hour"]:
            alert = SecurityAlert(
                alert_id=f"auth_bruteforce_{user_id}_{int(time.time())}",
                alert_type="brute_force_attack",
                severity="high",
                timestamp=now,
                user_id=user_id,
                description=f"Potential brute force attack detected for user {user_id}",
                details={
                    "failed_attempts_last_hour": len(recent_failures),
                    "threshold": self.thresholds["failed_auth_per_hour"],
                    "event_details": event.details
                }
            )
            alerts.append(alert)
            self._alerts.append(alert)
        
        # Check for rapid failed attempts (within minutes)
        recent_rapid_failures = self._get_recent_events(
            self._failed_auth_attempts[user_id], 
            minutes=1
        )
        
        if len(recent_rapid_failures) >= self.thresholds["failed_auth_per_minute"]:
            alert = SecurityAlert(
                alert_id=f"auth_rapid_{user_id}_{int(time.time())}",
                alert_type="rapid_failed_auth",
                severity="medium",
                timestamp=now,
                user_id=user_id,
                description=f"Rapid failed authentication attempts for user {user_id}",
                details={
                    "failed_attempts_last_minute": len(recent_rapid_failures),
                    "threshold": self.thresholds["failed_auth_per_minute"]
                }
            )
            alerts.append(alert)
            self._alerts.append(alert)
        
        return alerts
    
    def _process_failed_authorization(self, event: AuditEvent) -> List[SecurityAlert]:
        """Process failed authorization events."""
        alerts = []
        user_id = event.user_id
        now = datetime.utcnow()
        
        # Track suspicious activities
        self._suspicious_activities[user_id].append(now)
        
        # Check for privilege escalation attempts
        if "insufficient_permissions" in event.details.get("reason", ""):
            recent_attempts = self._get_recent_events(
                self._suspicious_activities[user_id], 
                minutes=60
            )
            
            if len(recent_attempts) >= self.thresholds["suspicious_activities_per_hour"]:
                alert = SecurityAlert(
                    alert_id=f"privilege_escalation_{user_id}_{int(time.time())}",
                    alert_type="privilege_escalation_attempt",
                    severity="high",
                    timestamp=now,
                    user_id=user_id,
                    description=f"Potential privilege escalation attempt by user {user_id}",
                    details={
                        "failed_authorization_attempts": len(recent_attempts),
                        "threshold": self.thresholds["suspicious_activities_per_hour"],
                        "requested_operation": event.details.get("required_permission")
                    }
                )
                alerts.append(alert)
                self._alerts.append(alert)
        
        return alerts
    
    def _process_pii_detection(self, event: AuditEvent) -> List[SecurityAlert]:
        """Process PII detection events."""
        alerts = []
        
        # Check for high-risk PII types
        pii_types = event.details.get("pii_types", [])
        high_risk_types = {"ssn", "credit_card", "medical_record", "passport_us"}
        
        detected_high_risk = set(pii_types).intersection(high_risk_types)
        
        if detected_high_risk:
            alert = SecurityAlert(
                alert_id=f"pii_high_risk_{event.user_id}_{int(time.time())}",
                alert_type="high_risk_pii_detected",
                severity="medium",
                timestamp=event.timestamp,
                user_id=event.user_id,
                description=f"High-risk PII detected in user data",
                details={
                    "high_risk_pii_types": list(detected_high_risk),
                    "total_matches": event.details.get("matches_found", 0)
                }
            )
            alerts.append(alert)
            self._alerts.append(alert)
        
        return alerts
    
    def _process_rate_limit_violation(self, event: AuditEvent) -> List[SecurityAlert]:
        """Process rate limit violation events."""
        alerts = []
        user_id = event.user_id
        now = datetime.utcnow()
        
        # Track rate limit violations
        self._rate_limit_violations[user_id].append(now)
        
        recent_violations = self._get_recent_events(
            self._rate_limit_violations[user_id], 
            minutes=60
        )
        
        if len(recent_violations) >= self.thresholds["rate_limit_violations_per_hour"]:
            alert = SecurityAlert(
                alert_id=f"rate_limit_abuse_{user_id}_{int(time.time())}",
                alert_type="rate_limit_abuse",
                severity="medium",
                timestamp=now,
                user_id=user_id,
                description=f"Excessive rate limit violations by user {user_id}",
                details={
                    "violations_last_hour": len(recent_violations),
                    "threshold": self.thresholds["rate_limit_violations_per_hour"]
                }
            )
            alerts.append(alert)
            self._alerts.append(alert)
        
        return alerts
    
    def _update_threat_indicators(self, event: AuditEvent) -> None:
        """Update threat indicators based on events."""
        # Extract potential threat indicators
        indicators = []
        
        # IP addresses from failed authentications
        if event.operation == "authentication" and event.status == "failed":
            ip_address = event.details.get("ip_address")
            if ip_address:
                indicators.append({
                    "type": "ip_address",
                    "value": ip_address,
                    "severity": "medium",
                    "description": "IP with failed authentication attempts"
                })
        
        # User agents from suspicious activities
        user_agent = event.details.get("user_agent")
        if user_agent and event.status == "failed":
            indicators.append({
                "type": "user_agent",
                "value": user_agent,
                "severity": "low",
                "description": "User agent associated with failed operations"
            })
        
        # Update or create threat indicators
        now = datetime.utcnow()
        for indicator_data in indicators:
            key = f"{indicator_data['type']}:{indicator_data['value']}"
            
            if key in self._threat_indicators:
                # Update existing indicator
                indicator = self._threat_indicators[key]
                indicator.last_seen = now
                indicator.count += 1
                
                # Escalate severity if count is high
                if indicator.count >= 10 and indicator.severity == "low":
                    indicator.severity = "medium"
                elif indicator.count >= 50 and indicator.severity == "medium":
                    indicator.severity = "high"
            else:
                # Create new indicator
                self._threat_indicators[key] = ThreatIndicator(
                    indicator_type=indicator_data["type"],
                    value=indicator_data["value"],
                    severity=indicator_data["severity"],
                    description=indicator_data["description"],
                    first_seen=now,
                    last_seen=now
                )
    
    def _get_recent_events(
        self, 
        events: List[datetime], 
        minutes: int
    ) -> List[datetime]:
        """Get events within the specified time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [event for event in events if event > cutoff]
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.alert_retention_hours)
        
        # Clean up old alerts
        self._alerts = [alert for alert in self._alerts if alert.timestamp > cutoff_time]
        
        # Clean up old tracking data
        for user_events in self._failed_auth_attempts.values():
            user_events[:] = [event for event in user_events if event > cutoff_time]
        
        for user_events in self._suspicious_activities.values():
            user_events[:] = [event for event in user_events if event > cutoff_time]
        
        for user_events in self._rate_limit_violations.values():
            user_events[:] = [event for event in user_events if event > cutoff_time]
        
        # Clean up old threat indicators
        old_indicators = []
        for key, indicator in self._threat_indicators.items():
            if indicator.last_seen < cutoff_time:
                old_indicators.append(key)
        
        for key in old_indicators:
            del self._threat_indicators[key]
    
    def get_active_alerts(
        self, 
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        limit: int = 100
    ) -> List[SecurityAlert]:
        """
        Get active security alerts.
        
        Args:
            severity: Filter by severity level
            alert_type: Filter by alert type
            limit: Maximum number of alerts to return
            
        Returns:
            List of security alerts
        """
        with self._lock:
            alerts = [alert for alert in self._alerts if not alert.resolved]
            
            # Apply filters
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            
            if alert_type:
                alerts = [alert for alert in alerts if alert.alert_type == alert_type]
            
            # Sort by timestamp (most recent first) and limit
            alerts.sort(key=lambda a: a.timestamp, reverse=True)
            return alerts[:limit]
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """
        Resolve a security alert.
        
        Args:
            alert_id: ID of the alert to resolve
            resolved_by: Who resolved the alert
            
        Returns:
            True if alert was resolved
        """
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    alert.details["resolved_by"] = resolved_by
                    return True
        
        return False
    
    def get_threat_indicators(
        self, 
        indicator_type: Optional[str] = None,
        min_severity: str = "low"
    ) -> List[ThreatIndicator]:
        """
        Get threat indicators.
        
        Args:
            indicator_type: Filter by indicator type
            min_severity: Minimum severity level
            
        Returns:
            List of threat indicators
        """
        severity_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = severity_levels.get(min_severity, 0)
        
        with self._lock:
            indicators = list(self._threat_indicators.values())
            
            # Apply filters
            if indicator_type:
                indicators = [ind for ind in indicators if ind.indicator_type == indicator_type]
            
            # Filter by severity
            indicators = [
                ind for ind in indicators 
                if severity_levels.get(ind.severity, 0) >= min_level
            ]
            
            # Sort by last seen (most recent first)
            indicators.sort(key=lambda i: i.last_seen, reverse=True)
            
            return indicators
    
    def block_ip(self, ip_address: str, reason: str = "Security violation") -> None:
        """
        Block an IP address.
        
        Args:
            ip_address: IP address to block
            reason: Reason for blocking
        """
        with self._lock:
            self._blocked_ips.add(ip_address)
            
            # Create alert for IP blocking
            alert = SecurityAlert(
                alert_id=f"ip_blocked_{ip_address}_{int(time.time())}",
                alert_type="ip_blocked",
                severity="high",
                timestamp=datetime.utcnow(),
                user_id="system",
                description=f"IP address {ip_address} has been blocked",
                details={
                    "ip_address": ip_address,
                    "reason": reason
                }
            )
            self._alerts.append(alert)
    
    def unblock_ip(self, ip_address: str) -> bool:
        """
        Unblock an IP address.
        
        Args:
            ip_address: IP address to unblock
            
        Returns:
            True if IP was unblocked
        """
        with self._lock:
            if ip_address in self._blocked_ips:
                self._blocked_ips.remove(ip_address)
                return True
        
        return False
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """
        Check if an IP address is blocked.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            True if IP is blocked
        """
        with self._lock:
            return ip_address in self._blocked_ips
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """
        Get security monitoring metrics.
        
        Returns:
            Dictionary with security metrics
        """
        with self._lock:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)
            
            # Count recent events
            recent_failed_auth = sum(
                len([event for event in events if event > hour_ago])
                for events in self._failed_auth_attempts.values()
            )
            
            recent_suspicious = sum(
                len([event for event in events if event > hour_ago])
                for events in self._suspicious_activities.values()
            )
            
            recent_rate_limits = sum(
                len([event for event in events if event > hour_ago])
                for events in self._rate_limit_violations.values()
            )
            
            # Count alerts
            active_alerts = len([alert for alert in self._alerts if not alert.resolved])
            critical_alerts = len([
                alert for alert in self._alerts 
                if not alert.resolved and alert.severity == "critical"
            ])
            
            return {
                "monitoring_enabled": self.enable_monitoring,
                "active_alerts": active_alerts,
                "critical_alerts": critical_alerts,
                "blocked_ips": len(self._blocked_ips),
                "threat_indicators": len(self._threat_indicators),
                "recent_failed_auth": recent_failed_auth,
                "recent_suspicious_activities": recent_suspicious,
                "recent_rate_limit_violations": recent_rate_limits,
                "total_users_tracked": len(self._failed_auth_attempts),
                "alert_retention_hours": self.alert_retention_hours,
                "thresholds": self.thresholds
            }
    
    def update_thresholds(self, new_thresholds: Dict[str, int]) -> None:
        """
        Update threat detection thresholds.
        
        Args:
            new_thresholds: Dictionary of new threshold values
        """
        with self._lock:
            for key, value in new_thresholds.items():
                if key in self.thresholds and isinstance(value, int) and value > 0:
                    self.thresholds[key] = value
    
    def get_user_risk_score(self, user_id: str) -> Dict[str, Any]:
        """
        Calculate risk score for a user.
        
        Args:
            user_id: User ID to analyze
            
        Returns:
            Dictionary with risk assessment
        """
        with self._lock:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)
            
            # Count recent activities
            failed_auth_hour = len([
                event for event in self._failed_auth_attempts.get(user_id, [])
                if event > hour_ago
            ])
            
            failed_auth_day = len([
                event for event in self._failed_auth_attempts.get(user_id, [])
                if event > day_ago
            ])
            
            suspicious_hour = len([
                event for event in self._suspicious_activities.get(user_id, [])
                if event > hour_ago
            ])
            
            rate_limit_hour = len([
                event for event in self._rate_limit_violations.get(user_id, [])
                if event > hour_ago
            ])
            
            # Calculate risk score (0-100)
            risk_score = 0
            risk_factors = []
            
            if failed_auth_hour > 0:
                risk_score += min(failed_auth_hour * 10, 40)
                risk_factors.append(f"Failed auth attempts: {failed_auth_hour}")
            
            if suspicious_hour > 0:
                risk_score += min(suspicious_hour * 15, 30)
                risk_factors.append(f"Suspicious activities: {suspicious_hour}")
            
            if rate_limit_hour > 0:
                risk_score += min(rate_limit_hour * 5, 20)
                risk_factors.append(f"Rate limit violations: {rate_limit_hour}")
            
            # Cap at 100
            risk_score = min(risk_score, 100)
            
            # Determine risk level
            if risk_score >= 80:
                risk_level = "critical"
            elif risk_score >= 60:
                risk_level = "high"
            elif risk_score >= 30:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "user_id": user_id,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "failed_auth_last_hour": failed_auth_hour,
                "failed_auth_last_day": failed_auth_day,
                "suspicious_activities_last_hour": suspicious_hour,
                "rate_limit_violations_last_hour": rate_limit_hour
            }
    
    def health_check(self) -> bool:
        """
        Perform health check on security monitoring service.
        
        Returns:
            True if service is healthy
        """
        try:
            with self._lock:
                # Check if basic functionality works
                test_event = AuditEvent(
                    timestamp=datetime.utcnow(),
                    user_id="health_check",
                    operation="test",
                    resource="health",
                    status="success",
                    details={},
                    correlation_id="health_check"
                )
                
                # Process test event
                self.process_audit_event(test_event)
                
                # Check if we can get metrics
                metrics = self.get_security_metrics()
                
                return isinstance(metrics, dict) and "monitoring_enabled" in metrics
                
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of security monitoring service."""
        return (
            f"SecurityMonitoringService(enabled={self.enable_monitoring}, "
            f"alerts={len(self._alerts)}, "
            f"indicators={len(self._threat_indicators)})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of security monitoring service."""
        return (
            f"SecurityMonitoringService(enabled={self.enable_monitoring}, "
            f"alerts={len(self._alerts)}, "
            f"threat_indicators={len(self._threat_indicators)}, "
            f"blocked_ips={len(self._blocked_ips)}, "
            f"retention_hours={self.alert_retention_hours})"
        )