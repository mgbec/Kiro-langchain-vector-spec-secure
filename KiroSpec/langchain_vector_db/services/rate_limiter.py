"""
Rate limiting service for API endpoints and operations.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from enum import Enum

from ..exceptions import SecurityException


class RateLimitType(Enum):
    """Types of rate limits."""
    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_OPERATION = "per_operation"
    GLOBAL = "global"


@dataclass
class RateLimitRule:
    """Represents a rate limiting rule."""
    
    name: str
    limit_type: RateLimitType
    max_requests: int
    window_seconds: int
    operation: Optional[str] = None
    burst_allowance: int = 0  # Additional requests allowed in burst
    description: str = ""


@dataclass
class RateLimitViolation:
    """Represents a rate limit violation."""
    
    rule_name: str
    identifier: str  # user_id, ip_address, etc.
    timestamp: datetime
    current_count: int
    limit: int
    window_seconds: int
    operation: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float, burst_allowance: int = 0):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
            burst_allowance: Additional tokens for burst traffic
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.burst_allowance = burst_allowance
        self.max_tokens = capacity + burst_allowance
        
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed successfully
        """
        with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_tokens,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_tokens(self) -> float:
        """Get current number of tokens."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            return min(
                self.max_tokens,
                self.tokens + (elapsed * self.refill_rate)
            )


class SlidingWindowCounter:
    """Sliding window counter for rate limiting."""
    
    def __init__(self, window_seconds: int, max_requests: int):
        """
        Initialize sliding window counter.
        
        Args:
            window_seconds: Time window in seconds
            max_requests: Maximum requests in window
        """
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.requests: deque = deque()
        self._lock = threading.Lock()
    
    def is_allowed(self) -> Tuple[bool, int]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (is_allowed, current_count)
        """
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            # Remove old requests
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            current_count = len(self.requests)
            
            if current_count < self.max_requests:
                self.requests.append(now)
                return True, current_count + 1
            
            return False, current_count
    
    def get_current_count(self) -> int:
        """Get current request count in window."""
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            # Remove old requests
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            return len(self.requests)


class RateLimitingService:
    """Service for rate limiting API requests and operations."""
    
    def __init__(
        self,
        enable_rate_limiting: bool = True,
        default_rules: Optional[List[RateLimitRule]] = None,
        violation_retention_hours: int = 24
    ):
        """
        Initialize rate limiting service.
        
        Args:
            enable_rate_limiting: Whether to enable rate limiting
            default_rules: Default rate limiting rules
            violation_retention_hours: How long to retain violation records
        """
        self.enable_rate_limiting = enable_rate_limiting
        self.violation_retention_hours = violation_retention_hours
        
        # Storage for rules and counters
        self.rules: Dict[str, RateLimitRule] = {}
        self.counters: Dict[str, SlidingWindowCounter] = {}
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.violations: List[RateLimitViolation] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Add default rules
        if default_rules:
            for rule in default_rules:
                self.add_rule(rule)
        else:
            self._add_default_rules()
    
    def _add_default_rules(self) -> None:
        """Add default rate limiting rules."""
        default_rules = [
            RateLimitRule(
                name="global_requests",
                limit_type=RateLimitType.GLOBAL,
                max_requests=1000,
                window_seconds=60,
                description="Global request limit per minute"
            ),
            RateLimitRule(
                name="user_requests",
                limit_type=RateLimitType.PER_USER,
                max_requests=100,
                window_seconds=60,
                burst_allowance=20,
                description="Per-user request limit per minute"
            ),
            RateLimitRule(
                name="user_auth_attempts",
                limit_type=RateLimitType.PER_USER,
                max_requests=5,
                window_seconds=300,  # 5 minutes
                operation="authentication",
                description="Authentication attempts per user per 5 minutes"
            ),
            RateLimitRule(
                name="ip_requests",
                limit_type=RateLimitType.PER_IP,
                max_requests=200,
                window_seconds=60,
                burst_allowance=50,
                description="Per-IP request limit per minute"
            ),
            RateLimitRule(
                name="search_operations",
                limit_type=RateLimitType.PER_USER,
                max_requests=50,
                window_seconds=60,
                operation="similarity_search",
                description="Search operations per user per minute"
            ),
            RateLimitRule(
                name="document_ingestion",
                limit_type=RateLimitType.PER_USER,
                max_requests=20,
                window_seconds=60,
                operation="add_documents",
                description="Document ingestion per user per minute"
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: RateLimitRule) -> None:
        """
        Add a rate limiting rule.
        
        Args:
            rule: Rate limiting rule to add
        """
        with self._lock:
            self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a rate limiting rule.
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was removed
        """
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                
                # Clean up associated counters
                keys_to_remove = [
                    key for key in self.counters.keys()
                    if key.startswith(f"{rule_name}:")
                ]
                for key in keys_to_remove:
                    del self.counters[key]
                
                keys_to_remove = [
                    key for key in self.token_buckets.keys()
                    if key.startswith(f"{rule_name}:")
                ]
                for key in keys_to_remove:
                    del self.token_buckets[key]
                
                return True
        
        return False
    
    def check_rate_limit(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        operation: Optional[str] = None,
        tokens: int = 1
    ) -> Tuple[bool, List[RateLimitViolation]]:
        """
        Check if request is within rate limits.
        
        Args:
            user_id: User identifier
            ip_address: IP address
            operation: Operation being performed
            tokens: Number of tokens to consume
            
        Returns:
            Tuple of (is_allowed, violations)
        """
        if not self.enable_rate_limiting:
            return True, []
        
        violations = []
        
        with self._lock:
            # Check applicable rules
            for rule in self.rules.values():
                # Skip if operation doesn't match
                if rule.operation and rule.operation != operation:
                    continue
                
                # Determine identifier based on rule type
                identifier = None
                if rule.limit_type == RateLimitType.PER_USER and user_id:
                    identifier = user_id
                elif rule.limit_type == RateLimitType.PER_IP and ip_address:
                    identifier = ip_address
                elif rule.limit_type == RateLimitType.GLOBAL:
                    identifier = "global"
                elif rule.limit_type == RateLimitType.PER_OPERATION and operation:
                    identifier = operation
                
                if not identifier:
                    continue
                
                # Check rate limit
                allowed = self._check_rule(rule, identifier, tokens)
                
                if not allowed:
                    violation = RateLimitViolation(
                        rule_name=rule.name,
                        identifier=identifier,
                        timestamp=datetime.utcnow(),
                        current_count=self._get_current_count(rule, identifier),
                        limit=rule.max_requests,
                        window_seconds=rule.window_seconds,
                        operation=operation,
                        details={
                            "user_id": user_id,
                            "ip_address": ip_address,
                            "tokens_requested": tokens
                        }
                    )
                    violations.append(violation)
                    self.violations.append(violation)
            
            # Clean up old violations
            self._cleanup_old_violations()
            
            # Return result
            is_allowed = len(violations) == 0
            return is_allowed, violations
    
    def _check_rule(
        self,
        rule: RateLimitRule,
        identifier: str,
        tokens: int
    ) -> bool:
        """
        Check a specific rule against an identifier.
        
        Args:
            rule: Rate limiting rule
            identifier: Identifier to check
            tokens: Number of tokens to consume
            
        Returns:
            True if request is allowed
        """
        counter_key = f"{rule.name}:{identifier}"
        
        # Use token bucket if burst allowance is configured
        if rule.burst_allowance > 0:
            if counter_key not in self.token_buckets:
                refill_rate = rule.max_requests / rule.window_seconds
                self.token_buckets[counter_key] = TokenBucket(
                    capacity=rule.max_requests,
                    refill_rate=refill_rate,
                    burst_allowance=rule.burst_allowance
                )
            
            return self.token_buckets[counter_key].consume(tokens)
        
        # Use sliding window counter
        else:
            if counter_key not in self.counters:
                self.counters[counter_key] = SlidingWindowCounter(
                    window_seconds=rule.window_seconds,
                    max_requests=rule.max_requests
                )
            
            allowed, _ = self.counters[counter_key].is_allowed()
            return allowed
    
    def _get_current_count(self, rule: RateLimitRule, identifier: str) -> int:
        """Get current count for a rule and identifier."""
        counter_key = f"{rule.name}:{identifier}"
        
        if rule.burst_allowance > 0 and counter_key in self.token_buckets:
            bucket = self.token_buckets[counter_key]
            return int(rule.max_requests + rule.burst_allowance - bucket.get_tokens())
        
        elif counter_key in self.counters:
            return self.counters[counter_key].get_current_count()
        
        return 0
    
    def get_rate_limit_status(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current rate limit status.
        
        Args:
            user_id: User identifier
            ip_address: IP address
            operation: Operation to check
            
        Returns:
            Dictionary with rate limit status
        """
        status = {}
        
        with self._lock:
            for rule in self.rules.values():
                # Skip if operation doesn't match
                if rule.operation and rule.operation != operation:
                    continue
                
                # Determine identifier
                identifier = None
                if rule.limit_type == RateLimitType.PER_USER and user_id:
                    identifier = user_id
                elif rule.limit_type == RateLimitType.PER_IP and ip_address:
                    identifier = ip_address
                elif rule.limit_type == RateLimitType.GLOBAL:
                    identifier = "global"
                elif rule.limit_type == RateLimitType.PER_OPERATION and operation:
                    identifier = operation
                
                if not identifier:
                    continue
                
                current_count = self._get_current_count(rule, identifier)
                remaining = max(0, rule.max_requests - current_count)
                
                status[rule.name] = {
                    "limit": rule.max_requests,
                    "remaining": remaining,
                    "current_count": current_count,
                    "window_seconds": rule.window_seconds,
                    "reset_time": datetime.utcnow() + timedelta(seconds=rule.window_seconds),
                    "burst_allowance": rule.burst_allowance
                }
        
        return status
    
    def reset_rate_limit(
        self,
        rule_name: str,
        identifier: str
    ) -> bool:
        """
        Reset rate limit for a specific rule and identifier.
        
        Args:
            rule_name: Name of the rule
            identifier: Identifier to reset
            
        Returns:
            True if reset was successful
        """
        with self._lock:
            counter_key = f"{rule_name}:{identifier}"
            
            # Reset token bucket
            if counter_key in self.token_buckets:
                rule = self.rules.get(rule_name)
                if rule:
                    refill_rate = rule.max_requests / rule.window_seconds
                    self.token_buckets[counter_key] = TokenBucket(
                        capacity=rule.max_requests,
                        refill_rate=refill_rate,
                        burst_allowance=rule.burst_allowance
                    )
                    return True
            
            # Reset sliding window counter
            if counter_key in self.counters:
                rule = self.rules.get(rule_name)
                if rule:
                    self.counters[counter_key] = SlidingWindowCounter(
                        window_seconds=rule.window_seconds,
                        max_requests=rule.max_requests
                    )
                    return True
        
        return False
    
    def get_violations(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        rule_name: Optional[str] = None,
        hours: int = 24
    ) -> List[RateLimitViolation]:
        """
        Get rate limit violations.
        
        Args:
            user_id: Filter by user ID
            ip_address: Filter by IP address
            rule_name: Filter by rule name
            hours: Hours to look back
            
        Returns:
            List of rate limit violations
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            violations = [v for v in self.violations if v.timestamp > cutoff]
            
            # Apply filters
            if user_id:
                violations = [
                    v for v in violations
                    if v.details.get("user_id") == user_id
                ]
            
            if ip_address:
                violations = [
                    v for v in violations
                    if v.details.get("ip_address") == ip_address
                ]
            
            if rule_name:
                violations = [v for v in violations if v.rule_name == rule_name]
            
            # Sort by timestamp (most recent first)
            violations.sort(key=lambda v: v.timestamp, reverse=True)
            
            return violations
    
    def _cleanup_old_violations(self) -> None:
        """Clean up old violation records."""
        cutoff = datetime.utcnow() - timedelta(hours=self.violation_retention_hours)
        self.violations = [v for v in self.violations if v.timestamp > cutoff]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get rate limiting metrics.
        
        Returns:
            Dictionary with metrics
        """
        with self._lock:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            
            # Count recent violations
            recent_violations = len([
                v for v in self.violations
                if v.timestamp > hour_ago
            ])
            
            # Count violations by rule
            violations_by_rule = defaultdict(int)
            for violation in self.violations:
                if violation.timestamp > hour_ago:
                    violations_by_rule[violation.rule_name] += 1
            
            return {
                "rate_limiting_enabled": self.enable_rate_limiting,
                "total_rules": len(self.rules),
                "active_counters": len(self.counters),
                "active_token_buckets": len(self.token_buckets),
                "total_violations": len(self.violations),
                "recent_violations": recent_violations,
                "violations_by_rule": dict(violations_by_rule),
                "violation_retention_hours": self.violation_retention_hours
            }
    
    def update_rule(self, rule_name: str, **kwargs) -> bool:
        """
        Update an existing rate limiting rule.
        
        Args:
            rule_name: Name of rule to update
            **kwargs: Fields to update
            
        Returns:
            True if rule was updated
        """
        with self._lock:
            if rule_name not in self.rules:
                return False
            
            rule = self.rules[rule_name]
            
            # Update allowed fields
            if "max_requests" in kwargs:
                rule.max_requests = kwargs["max_requests"]
            if "window_seconds" in kwargs:
                rule.window_seconds = kwargs["window_seconds"]
            if "burst_allowance" in kwargs:
                rule.burst_allowance = kwargs["burst_allowance"]
            if "description" in kwargs:
                rule.description = kwargs["description"]
            
            # Reset counters for this rule
            keys_to_remove = [
                key for key in self.counters.keys()
                if key.startswith(f"{rule_name}:")
            ]
            for key in keys_to_remove:
                del self.counters[key]
            
            keys_to_remove = [
                key for key in self.token_buckets.keys()
                if key.startswith(f"{rule_name}:")
            ]
            for key in keys_to_remove:
                del self.token_buckets[key]
            
            return True
    
    def health_check(self) -> bool:
        """
        Perform health check on rate limiting service.
        
        Returns:
            True if service is healthy
        """
        try:
            with self._lock:
                # Test basic functionality
                allowed, violations = self.check_rate_limit(
                    user_id="health_check",
                    operation="test"
                )
                
                # Check if we can get metrics
                metrics = self.get_metrics()
                
                return isinstance(metrics, dict) and "rate_limiting_enabled" in metrics
                
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of rate limiting service."""
        return (
            f"RateLimitingService(enabled={self.enable_rate_limiting}, "
            f"rules={len(self.rules)}, "
            f"violations={len(self.violations)})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of rate limiting service."""
        return (
            f"RateLimitingService(enabled={self.enable_rate_limiting}, "
            f"rules={len(self.rules)}, "
            f"counters={len(self.counters)}, "
            f"token_buckets={len(self.token_buckets)}, "
            f"violations={len(self.violations)})"
        )