"""
Unit tests for RateLimitingService.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch

from langchain_vector_db.services.rate_limiter import (
    RateLimitingService,
    RateLimitRule,
    RateLimitType,
    RateLimitViolation,
    TokenBucket,
    SlidingWindowCounter
)


class TestTokenBucket:
    """Test cases for TokenBucket."""
    
    def test_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0, burst_allowance=5)
        
        assert bucket.capacity == 10
        assert bucket.refill_rate == 1.0
        assert bucket.burst_allowance == 5
        assert bucket.max_tokens == 15  # capacity + burst_allowance
        assert bucket.tokens == 10.0  # starts at capacity
    
    def test_consume_tokens_success(self):
        """Test successful token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Should be able to consume tokens
        assert bucket.consume(5) is True
        assert bucket.get_tokens() == 5.0
        
        # Should be able to consume remaining tokens
        assert bucket.consume(5) is True
        assert bucket.get_tokens() == 0.0
    
    def test_consume_tokens_failure(self):
        """Test token consumption failure when insufficient tokens."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Try to consume more tokens than available
        assert bucket.consume(10) is False
        assert bucket.get_tokens() == 5.0  # Should remain unchanged
    
    def test_token_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)  # 2 tokens per second
        
        # Consume all tokens
        bucket.consume(10)
        assert bucket.get_tokens() == 0.0
        
        # Mock time passage
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 2.5]  # 2.5 seconds later
            bucket.last_refill = 0
            
            # Should have refilled 5 tokens (2.5 * 2.0)
            assert bucket.get_tokens() == 5.0
    
    def test_burst_allowance(self):
        """Test burst allowance functionality."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0, burst_allowance=5)
        
        # Should be able to consume up to max_tokens (15)
        assert bucket.consume(15) is True
        assert bucket.get_tokens() == 0.0
        
        # Should not be able to consume more
        assert bucket.consume(1) is False


class TestSlidingWindowCounter:
    """Test cases for SlidingWindowCounter."""
    
    def test_initialization(self):
        """Test sliding window counter initialization."""
        counter = SlidingWindowCounter(window_seconds=60, max_requests=10)
        
        assert counter.window_seconds == 60
        assert counter.max_requests == 10
        assert len(counter.requests) == 0
    
    def test_requests_within_limit(self):
        """Test requests within limit."""
        counter = SlidingWindowCounter(window_seconds=60, max_requests=5)
        
        # Should allow requests within limit
        for i in range(5):
            allowed, count = counter.is_allowed()
            assert allowed is True
            assert count == i + 1
        
        # Should reject request over limit
        allowed, count = counter.is_allowed()
        assert allowed is False
        assert count == 5
    
    def test_sliding_window_cleanup(self):
        """Test sliding window cleanup of old requests."""
        counter = SlidingWindowCounter(window_seconds=60, max_requests=5)
        
        with patch('time.time') as mock_time:
            # Add requests at different times
            mock_time.return_value = 0
            counter.is_allowed()  # Request at time 0
            
            mock_time.return_value = 30
            counter.is_allowed()  # Request at time 30
            
            mock_time.return_value = 70  # 70 seconds later
            # First request should be cleaned up (older than 60 seconds)
            allowed, count = counter.is_allowed()
            assert allowed is True
            assert count == 2  # Only requests at 30 and 70 remain
    
    def test_get_current_count(self):
        """Test getting current count."""
        counter = SlidingWindowCounter(window_seconds=60, max_requests=10)
        
        assert counter.get_current_count() == 0
        
        counter.is_allowed()
        counter.is_allowed()
        assert counter.get_current_count() == 2


class TestRateLimitingService:
    """Test cases for RateLimitingService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = RateLimitingService(
            enable_rate_limiting=True,
            default_rules=None,  # Don't add default rules for testing
            violation_retention_hours=24
        )
    
    def test_initialization(self):
        """Test rate limiting service initialization."""
        service = RateLimitingService()
        
        assert service.enable_rate_limiting is True
        assert service.violation_retention_hours == 24
        assert len(service.rules) > 0  # Should have default rules
        assert len(service.counters) == 0
        assert len(service.violations) == 0
    
    def test_add_rule(self):
        """Test adding rate limiting rules."""
        rule = RateLimitRule(
            name="test_rule",
            limit_type=RateLimitType.PER_USER,
            max_requests=10,
            window_seconds=60,
            description="Test rule"
        )
        
        self.service.add_rule(rule)
        assert "test_rule" in self.service.rules
        assert self.service.rules["test_rule"] == rule
    
    def test_remove_rule(self):
        """Test removing rate limiting rules."""
        rule = RateLimitRule(
            name="test_rule",
            limit_type=RateLimitType.PER_USER,
            max_requests=10,
            window_seconds=60
        )
        
        self.service.add_rule(rule)
        assert "test_rule" in self.service.rules
        
        # Remove rule
        result = self.service.remove_rule("test_rule")
        assert result is True
        assert "test_rule" not in self.service.rules
        
        # Try to remove non-existent rule
        result = self.service.remove_rule("non_existent")
        assert result is False
    
    def test_check_rate_limit_disabled(self):
        """Test rate limit check when disabled."""
        service = RateLimitingService(enable_rate_limiting=False)
        
        allowed, violations = service.check_rate_limit(user_id="test_user")
        assert allowed is True
        assert len(violations) == 0
    
    def test_check_rate_limit_per_user(self):
        """Test per-user rate limiting."""
        rule = RateLimitRule(
            name="user_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=3,
            window_seconds=60
        )
        self.service.add_rule(rule)
        
        user_id = "test_user"
        
        # First 3 requests should be allowed
        for i in range(3):
            allowed, violations = self.service.check_rate_limit(user_id=user_id)
            assert allowed is True
            assert len(violations) == 0
        
        # 4th request should be denied
        allowed, violations = self.service.check_rate_limit(user_id=user_id)
        assert allowed is False
        assert len(violations) == 1
        assert violations[0].rule_name == "user_limit"
        assert violations[0].identifier == user_id
    
    def test_check_rate_limit_per_ip(self):
        """Test per-IP rate limiting."""
        rule = RateLimitRule(
            name="ip_limit",
            limit_type=RateLimitType.PER_IP,
            max_requests=2,
            window_seconds=60
        )
        self.service.add_rule(rule)
        
        ip_address = "192.168.1.1"
        
        # First 2 requests should be allowed
        for i in range(2):
            allowed, violations = self.service.check_rate_limit(ip_address=ip_address)
            assert allowed is True
            assert len(violations) == 0
        
        # 3rd request should be denied
        allowed, violations = self.service.check_rate_limit(ip_address=ip_address)
        assert allowed is False
        assert len(violations) == 1
        assert violations[0].rule_name == "ip_limit"
        assert violations[0].identifier == ip_address
    
    def test_check_rate_limit_global(self):
        """Test global rate limiting."""
        rule = RateLimitRule(
            name="global_limit",
            limit_type=RateLimitType.GLOBAL,
            max_requests=2,
            window_seconds=60
        )
        self.service.add_rule(rule)
        
        # First 2 requests from different users should be allowed
        allowed, violations = self.service.check_rate_limit(user_id="user1")
        assert allowed is True
        
        allowed, violations = self.service.check_rate_limit(user_id="user2")
        assert allowed is True
        
        # 3rd request should be denied regardless of user
        allowed, violations = self.service.check_rate_limit(user_id="user3")
        assert allowed is False
        assert len(violations) == 1
        assert violations[0].rule_name == "global_limit"
        assert violations[0].identifier == "global"
    
    def test_check_rate_limit_operation_specific(self):
        """Test operation-specific rate limiting."""
        rule = RateLimitRule(
            name="search_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=2,
            window_seconds=60,
            operation="search"
        )
        self.service.add_rule(rule)
        
        user_id = "test_user"
        
        # Search operations should be limited
        allowed, violations = self.service.check_rate_limit(
            user_id=user_id, operation="search"
        )
        assert allowed is True
        
        allowed, violations = self.service.check_rate_limit(
            user_id=user_id, operation="search"
        )
        assert allowed is True
        
        allowed, violations = self.service.check_rate_limit(
            user_id=user_id, operation="search"
        )
        assert allowed is False
        
        # Other operations should not be affected
        allowed, violations = self.service.check_rate_limit(
            user_id=user_id, operation="create"
        )
        assert allowed is True
    
    def test_burst_allowance(self):
        """Test burst allowance functionality."""
        rule = RateLimitRule(
            name="burst_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=5,
            window_seconds=60,
            burst_allowance=3
        )
        self.service.add_rule(rule)
        
        user_id = "test_user"
        
        # Should allow up to max_requests + burst_allowance (8 total)
        for i in range(8):
            allowed, violations = self.service.check_rate_limit(user_id=user_id)
            assert allowed is True, f"Request {i+1} should be allowed"
        
        # 9th request should be denied
        allowed, violations = self.service.check_rate_limit(user_id=user_id)
        assert allowed is False
    
    def test_get_rate_limit_status(self):
        """Test getting rate limit status."""
        rule = RateLimitRule(
            name="test_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=5,
            window_seconds=60
        )
        self.service.add_rule(rule)
        
        user_id = "test_user"
        
        # Make some requests
        self.service.check_rate_limit(user_id=user_id)
        self.service.check_rate_limit(user_id=user_id)
        
        status = self.service.get_rate_limit_status(user_id=user_id)
        
        assert "test_limit" in status
        limit_status = status["test_limit"]
        assert limit_status["limit"] == 5
        assert limit_status["remaining"] == 3
        assert limit_status["current_count"] == 2
        assert limit_status["window_seconds"] == 60
    
    def test_reset_rate_limit(self):
        """Test resetting rate limits."""
        rule = RateLimitRule(
            name="test_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=2,
            window_seconds=60
        )
        self.service.add_rule(rule)
        
        user_id = "test_user"
        
        # Exhaust rate limit
        self.service.check_rate_limit(user_id=user_id)
        self.service.check_rate_limit(user_id=user_id)
        
        allowed, violations = self.service.check_rate_limit(user_id=user_id)
        assert allowed is False
        
        # Reset rate limit
        result = self.service.reset_rate_limit("test_limit", user_id)
        assert result is True
        
        # Should be able to make requests again
        allowed, violations = self.service.check_rate_limit(user_id=user_id)
        assert allowed is True
    
    def test_get_violations(self):
        """Test getting rate limit violations."""
        rule = RateLimitRule(
            name="test_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=1,
            window_seconds=60
        )
        self.service.add_rule(rule)
        
        user_id = "test_user"
        ip_address = "192.168.1.1"
        
        # Generate violations
        self.service.check_rate_limit(user_id=user_id, ip_address=ip_address)
        self.service.check_rate_limit(user_id=user_id, ip_address=ip_address)  # This should violate
        
        violations = self.service.get_violations()
        assert len(violations) == 1
        assert violations[0].rule_name == "test_limit"
        assert violations[0].identifier == user_id
        
        # Test filtering by user_id
        violations = self.service.get_violations(user_id=user_id)
        assert len(violations) == 1
        
        violations = self.service.get_violations(user_id="other_user")
        assert len(violations) == 0
    
    def test_update_rule(self):
        """Test updating rate limiting rules."""
        rule = RateLimitRule(
            name="test_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=5,
            window_seconds=60
        )
        self.service.add_rule(rule)
        
        # Update rule
        result = self.service.update_rule(
            "test_limit",
            max_requests=10,
            window_seconds=120
        )
        assert result is True
        
        updated_rule = self.service.rules["test_limit"]
        assert updated_rule.max_requests == 10
        assert updated_rule.window_seconds == 120
        
        # Try to update non-existent rule
        result = self.service.update_rule("non_existent", max_requests=5)
        assert result is False
    
    def test_get_metrics(self):
        """Test getting rate limiting metrics."""
        rule = RateLimitRule(
            name="test_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=1,
            window_seconds=60
        )
        self.service.add_rule(rule)
        
        # Generate some violations
        self.service.check_rate_limit(user_id="user1")
        self.service.check_rate_limit(user_id="user1")  # Violation
        
        metrics = self.service.get_metrics()
        
        assert metrics["rate_limiting_enabled"] is True
        assert metrics["total_rules"] == 1
        assert metrics["total_violations"] == 1
        assert metrics["recent_violations"] == 1
        assert "test_limit" in metrics["violations_by_rule"]
    
    def test_health_check(self):
        """Test health check functionality."""
        result = self.service.health_check()
        assert result is True
        
        # Test with disabled service
        service = RateLimitingService(enable_rate_limiting=False)
        result = service.health_check()
        assert result is True
    
    def test_violation_cleanup(self):
        """Test cleanup of old violations."""
        rule = RateLimitRule(
            name="test_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=1,
            window_seconds=60
        )
        self.service.add_rule(rule)
        
        # Generate violation
        self.service.check_rate_limit(user_id="test_user")
        self.service.check_rate_limit(user_id="test_user")  # Violation
        
        assert len(self.service.violations) == 1
        
        # Mock old violation
        old_violation = self.service.violations[0]
        old_violation.timestamp = datetime.utcnow() - timedelta(hours=25)
        
        # Trigger cleanup by making another request
        self.service.check_rate_limit(user_id="test_user2")
        
        # Old violation should be cleaned up
        assert len(self.service.violations) == 0
    
    def test_multiple_rules_same_user(self):
        """Test multiple rules applying to the same user."""
        rule1 = RateLimitRule(
            name="general_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=5,
            window_seconds=60
        )
        
        rule2 = RateLimitRule(
            name="search_limit",
            limit_type=RateLimitType.PER_USER,
            max_requests=2,
            window_seconds=60,
            operation="search"
        )
        
        self.service.add_rule(rule1)
        self.service.add_rule(rule2)
        
        user_id = "test_user"
        
        # First search should be allowed (both rules satisfied)
        allowed, violations = self.service.check_rate_limit(
            user_id=user_id, operation="search"
        )
        assert allowed is True
        
        # Second search should be allowed
        allowed, violations = self.service.check_rate_limit(
            user_id=user_id, operation="search"
        )
        assert allowed is True
        
        # Third search should be denied (violates search_limit)
        allowed, violations = self.service.check_rate_limit(
            user_id=user_id, operation="search"
        )
        assert allowed is False
        assert len(violations) == 1
        assert violations[0].rule_name == "search_limit"
    
    def test_string_representations(self):
        """Test string representations of the service."""
        str_repr = str(self.service)
        assert "RateLimitingService" in str_repr
        assert "enabled=True" in str_repr
        
        repr_str = repr(self.service)
        assert "RateLimitingService" in repr_str
        assert "enabled=True" in repr_str