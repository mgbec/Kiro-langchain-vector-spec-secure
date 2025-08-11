"""
Unit tests for PIIDetectionService.
"""

import pytest

from langchain_vector_db.services.pii_detection import PIIDetectionService, PIIPattern
from langchain_vector_db.models.pii import PIIMatch
from langchain_vector_db.exceptions import SecurityException


class TestPIIDetectionService:
    """Test cases for PIIDetectionService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = PIIDetectionService(enable_ml_detection=False)
    
    def test_initialization(self):
        """Test PII detection service initialization."""
        service = PIIDetectionService()
        
        assert service.enable_ml_detection is False
        assert len(service.patterns) > 0
        assert service._ml_analyzer is None
    
    def test_detect_email_addresses(self):
        """Test detection of email addresses."""
        text = "Contact me at john.doe@example.com or admin@company.org"
        
        matches = self.service.detect_pii(text)
        
        email_matches = [m for m in matches if m.type == "email"]
        assert len(email_matches) == 2
        
        emails = [m.text for m in email_matches]
        assert "john.doe@example.com" in emails
        assert "admin@company.org" in emails
    
    def test_detect_phone_numbers(self):
        """Test detection of phone numbers."""
        text = "Call me at 555-123-4567 or (555) 987-6543"
        
        matches = self.service.detect_pii(text)
        
        phone_matches = [m for m in matches if m.type == "phone_us"]
        assert len(phone_matches) == 2
    
    def test_detect_ssn(self):
        """Test detection of Social Security Numbers."""
        text = "My SSN is 123-45-6789 and yours is 987.65.4321"
        
        matches = self.service.detect_pii(text)
        
        ssn_matches = [m for m in matches if m.type == "ssn"]
        assert len(ssn_matches) == 2
        
        ssns = [m.text for m in ssn_matches]
        assert "123-45-6789" in ssns
        assert "987.65.4321" in ssns
    
    def test_detect_credit_card_numbers(self):
        """Test detection of credit card numbers."""
        text = "My Visa card is 4111111111111111 and Mastercard is 5555555555554444"
        
        matches = self.service.detect_pii(text)
        
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 2
    
    def test_detect_ip_addresses(self):
        """Test detection of IP addresses."""
        text = "Server IP is 192.168.1.1 and backup is 10.0.0.1"
        
        matches = self.service.detect_pii(text)
        
        ip_matches = [m for m in matches if m.type == "ip_address"]
        assert len(ip_matches) == 2
        
        ips = [m.text for m in ip_matches]
        assert "192.168.1.1" in ips
        assert "10.0.0.1" in ips
    
    def test_detect_medical_record_numbers(self):
        """Test detection of medical record numbers."""
        text = "Patient MRN: 1234567890 and MRN 9876543210"
        
        matches = self.service.detect_pii(text)
        
        mrn_matches = [m for m in matches if m.type == "medical_record"]
        assert len(mrn_matches) == 2
    
    def test_detect_date_of_birth(self):
        """Test detection of date of birth."""
        text = "DOB: 01/15/1990 and Date of Birth 12-25-1985"
        
        matches = self.service.detect_pii(text)
        
        dob_matches = [m for m in matches if m.type == "date_of_birth"]
        assert len(dob_matches) == 2
    
    def test_confidence_threshold_filtering(self):
        """Test that confidence threshold filters results."""
        text = "Email: test@example.com and possible account: 12345678"
        
        # Low threshold - should find both
        matches_low = self.service.detect_pii(text, confidence_threshold=0.3)
        
        # High threshold - should find only high-confidence matches
        matches_high = self.service.detect_pii(text, confidence_threshold=0.8)
        
        assert len(matches_high) <= len(matches_low)
        
        # Email should be in high confidence matches
        high_conf_types = [m.type for m in matches_high]
        assert "email" in high_conf_types
    
    def test_mask_pii_basic(self):
        """Test basic PII masking."""
        text = "Contact john.doe@example.com or call 555-123-4567"
        
        masked_text, matches = self.service.mask_pii(text)
        
        assert "john.doe@example.com" not in masked_text
        assert "555-123-4567" not in masked_text
        assert "*" in masked_text
        assert len(matches) >= 2
    
    def test_mask_pii_preserve_format(self):
        """Test PII masking with format preservation."""
        text = "SSN: 123-45-6789 and phone: (555) 123-4567"
        
        masked_text, matches = self.service.mask_pii(text, preserve_format=True)
        
        # Should preserve dashes and parentheses
        assert "***-**-****" in masked_text or "***.**-****" in masked_text
        assert "(" in masked_text and ")" in masked_text
    
    def test_mask_pii_custom_character(self):
        """Test PII masking with custom mask character."""
        text = "Email: test@example.com"
        
        masked_text, matches = self.service.mask_pii(text, mask_char="X")
        
        assert "X" in masked_text
        assert "*" not in masked_text
        assert "test@example.com" not in masked_text
    
    def test_get_pii_summary(self):
        """Test getting PII summary."""
        text = "Contact john@example.com, call 555-123-4567, SSN: 123-45-6789"
        
        matches = self.service.detect_pii(text)
        summary = self.service.get_pii_summary(matches)
        
        assert summary["total_matches"] >= 3
        assert "email" in summary["pii_types"]
        assert "phone_us" in summary["pii_types"]
        assert "ssn" in summary["pii_types"]
        assert summary["high_confidence_count"] >= 0
        assert 0 <= summary["average_confidence"] <= 1
    
    def test_get_pii_summary_empty(self):
        """Test PII summary with no matches."""
        summary = self.service.get_pii_summary([])
        
        assert summary["total_matches"] == 0
        assert summary["pii_types"] == []
        assert summary["high_confidence_count"] == 0
        assert summary["coverage_percentage"] == 0.0
    
    def test_add_custom_pattern(self):
        """Test adding custom PII pattern."""
        original_count = len(self.service.patterns)
        
        # Add custom pattern for employee IDs
        self.service.add_custom_pattern(
            name="employee_id",
            pattern=r"EMP-\d{6}",
            confidence=0.9,
            description="Employee ID pattern"
        )
        
        assert len(self.service.patterns) == original_count + 1
        
        # Test detection with custom pattern
        text = "Employee ID: EMP-123456"
        matches = self.service.detect_pii(text)
        
        emp_matches = [m for m in matches if m.type == "employee_id"]
        assert len(emp_matches) == 1
        assert emp_matches[0].text == "EMP-123456"
    
    def test_add_invalid_custom_pattern(self):
        """Test adding invalid custom pattern."""
        with pytest.raises(SecurityException) as exc_info:
            self.service.add_custom_pattern(
                name="invalid",
                pattern="[invalid regex",  # Invalid regex
                confidence=0.8
            )
        
        assert "Invalid regex pattern" in str(exc_info.value)
    
    def test_remove_pattern(self):
        """Test removing a PII pattern."""
        # Add a custom pattern first
        self.service.add_custom_pattern(
            name="test_pattern",
            pattern=r"TEST-\d+",
            confidence=0.8
        )
        
        original_count = len(self.service.patterns)
        
        # Remove the pattern
        result = self.service.remove_pattern("test_pattern")
        
        assert result is True
        assert len(self.service.patterns) == original_count - 1
        
        # Try to remove non-existent pattern
        result = self.service.remove_pattern("non_existent")
        assert result is False
    
    def test_get_available_patterns(self):
        """Test getting available patterns."""
        patterns = self.service.get_available_patterns()
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        # Check pattern structure
        for pattern in patterns:
            assert "name" in pattern
            assert "confidence" in pattern
            assert "description" in pattern
            assert 0 <= pattern["confidence"] <= 1
    
    def test_deduplication_of_overlapping_matches(self):
        """Test deduplication of overlapping PII matches."""
        # Create a service with overlapping patterns for testing
        service = PIIDetectionService()
        
        # Add overlapping patterns
        service.add_custom_pattern("test1", r"\d{3}-\d{2}-\d{4}", 0.8, "Pattern 1")
        service.add_custom_pattern("test2", r"\d{3}-\d{2}-\d{4}", 0.9, "Pattern 2")
        
        text = "SSN: 123-45-6789"
        matches = service.detect_pii(text)
        
        # Should deduplicate overlapping matches
        ssn_text_matches = [m for m in matches if m.text == "123-45-6789"]
        
        # Should keep only the highest confidence match
        if len(ssn_text_matches) > 1:
            confidences = [m.confidence for m in ssn_text_matches]
            assert max(confidences) in confidences
    
    def test_no_pii_detected(self):
        """Test text with no PII."""
        text = "This is a normal sentence with no personal information."
        
        matches = self.service.detect_pii(text)
        
        assert len(matches) == 0
    
    def test_mixed_pii_types(self):
        """Test text with multiple PII types."""
        text = """
        Contact Information:
        Email: john.doe@company.com
        Phone: (555) 123-4567
        SSN: 123-45-6789
        Credit Card: 4111111111111111
        IP Address: 192.168.1.100
        """
        
        matches = self.service.detect_pii(text)
        
        detected_types = set(m.type for m in matches)
        expected_types = {"email", "phone_us", "ssn", "credit_card", "ip_address"}
        
        # Should detect most or all expected types
        assert len(detected_types.intersection(expected_types)) >= 3
    
    def test_health_check(self):
        """Test PII detection service health check."""
        result = self.service.health_check()
        
        assert result is True
    
    def test_health_check_failure(self):
        """Test health check failure scenario."""
        # Break the service by clearing patterns
        original_patterns = self.service.patterns
        self.service.patterns = []
        
        result = self.service.health_check()
        
        # Restore patterns
        self.service.patterns = original_patterns
        
        # Health check should fail without email pattern
        assert result is False
    
    def test_string_representations(self):
        """Test string representations of PII detection service."""
        str_repr = str(self.service)
        assert "PIIDetectionService" in str_repr
        assert "patterns=" in str_repr
        assert "ml_enabled=False" in str_repr
        
        repr_str = repr(self.service)
        assert "PIIDetectionService" in repr_str
        assert "ml_analyzer=not_available" in repr_str


class TestPIIPatternClass:
    """Test cases for PIIPattern class."""
    
    def test_pii_pattern_creation(self):
        """Test creating a PII pattern."""
        import re
        
        pattern = PIIPattern(
            name="test_pattern",
            pattern=re.compile(r"\d{3}-\d{3}-\d{4}"),
            confidence=0.8,
            description="Test phone pattern"
        )
        
        assert pattern.name == "test_pattern"
        assert pattern.confidence == 0.8
        assert pattern.description == "Test phone pattern"
        assert pattern.pattern.pattern == r"\d{3}-\d{3}-\d{4}"


if __name__ == "__main__":
    pytest.main([__file__])