"""
PII (Personally Identifiable Information) detection service.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..models.pii import PIIMatch
from ..exceptions import SecurityException


@dataclass
class PIIPattern:
    """Pattern for detecting PII."""
    name: str
    pattern: re.Pattern
    confidence: float
    description: str


class PIIDetectionService:
    """Service for detecting and masking PII in text."""
    
    def __init__(self, enable_ml_detection: bool = False):
        """
        Initialize PII detection service.
        
        Args:
            enable_ml_detection: Whether to enable ML-based detection (requires presidio)
        """
        self.enable_ml_detection = enable_ml_detection
        
        # Initialize regex patterns
        self.patterns = self._initialize_patterns()
        
        # Initialize ML-based detection if enabled
        self._ml_analyzer = None
        if enable_ml_detection:
            self._init_ml_detection()
    
    def _initialize_patterns(self) -> List[PIIPattern]:
        """Initialize regex patterns for PII detection."""
        patterns = [
            # Email addresses
            PIIPattern(
                name="email",
                pattern=re.compile(
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    re.IGNORECASE
                ),
                confidence=0.9,
                description="Email address"
            ),
            
            # Phone numbers (US format)
            PIIPattern(
                name="phone_us",
                pattern=re.compile(
                    r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
                ),
                confidence=0.8,
                description="US phone number"
            ),
            
            # Social Security Numbers
            PIIPattern(
                name="ssn",
                pattern=re.compile(
                    r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b'
                ),
                confidence=0.95,
                description="Social Security Number"
            ),
            
            # Credit Card Numbers (basic pattern)
            PIIPattern(
                name="credit_card",
                pattern=re.compile(
                    r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
                ),
                confidence=0.85,
                description="Credit card number"
            ),
            
            # IP Addresses
            PIIPattern(
                name="ip_address",
                pattern=re.compile(
                    r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
                ),
                confidence=0.7,
                description="IP address"
            ),
            
            # Driver's License (generic pattern)
            PIIPattern(
                name="drivers_license",
                pattern=re.compile(
                    r'\b[A-Z]{1,2}[0-9]{6,8}\b'
                ),
                confidence=0.6,
                description="Driver's license number"
            ),
            
            # Passport Numbers (US format)
            PIIPattern(
                name="passport_us",
                pattern=re.compile(
                    r'\b[0-9]{9}\b'
                ),
                confidence=0.5,
                description="US passport number"
            ),
            
            # Bank Account Numbers (generic)
            PIIPattern(
                name="bank_account",
                pattern=re.compile(
                    r'\b[0-9]{8,17}\b'
                ),
                confidence=0.4,
                description="Bank account number"
            ),
            
            # Medical Record Numbers
            PIIPattern(
                name="medical_record",
                pattern=re.compile(
                    r'\bMRN[-:\s]*[0-9]{6,10}\b',
                    re.IGNORECASE
                ),
                confidence=0.8,
                description="Medical record number"
            ),
            
            # Date of Birth patterns
            PIIPattern(
                name="date_of_birth",
                pattern=re.compile(
                    r'\b(?:DOB|Date of Birth|Born)[-:\s]*(?:[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}|[0-9]{2,4}[/-][0-9]{1,2}[/-][0-9]{1,2})\b',
                    re.IGNORECASE
                ),
                confidence=0.7,
                description="Date of birth"
            )
        ]
        
        return patterns
    
    def _init_ml_detection(self) -> None:
        """Initialize ML-based PII detection using presidio."""
        try:
            from presidio_analyzer import AnalyzerEngine
            self._ml_analyzer = AnalyzerEngine()
        except ImportError:
            raise SecurityException(
                "ML-based PII detection requires presidio-analyzer. "
                "Install with: pip install presidio-analyzer"
            )
    
    def detect_pii(
        self, 
        text: str, 
        confidence_threshold: float = 0.5
    ) -> List[PIIMatch]:
        """
        Detect PII in text.
        
        Args:
            text: Text to analyze
            confidence_threshold: Minimum confidence threshold for matches
            
        Returns:
            List of PII matches found
        """
        matches = []
        
        # Regex-based detection
        regex_matches = self._detect_pii_regex(text, confidence_threshold)
        matches.extend(regex_matches)
        
        # ML-based detection (if enabled)
        if self.enable_ml_detection and self._ml_analyzer:
            ml_matches = self._detect_pii_ml(text, confidence_threshold)
            matches.extend(ml_matches)
        
        # Remove duplicates and sort by position
        matches = self._deduplicate_matches(matches)
        matches.sort(key=lambda m: m.start_pos)
        
        return matches
    
    def _detect_pii_regex(
        self, 
        text: str, 
        confidence_threshold: float
    ) -> List[PIIMatch]:
        """Detect PII using regex patterns."""
        matches = []
        
        for pattern in self.patterns:
            if pattern.confidence < confidence_threshold:
                continue
            
            for match in pattern.pattern.finditer(text):
                pii_match = PIIMatch(
                    type=pattern.name,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=pattern.confidence
                )
                matches.append(pii_match)
        
        return matches
    
    def _detect_pii_ml(
        self, 
        text: str, 
        confidence_threshold: float
    ) -> List[PIIMatch]:
        """Detect PII using ML-based analysis."""
        if not self._ml_analyzer:
            return []
        
        try:
            # Analyze text with presidio
            results = self._ml_analyzer.analyze(
                text=text,
                language='en',
                score_threshold=confidence_threshold
            )
            
            matches = []
            for result in results:
                pii_match = PIIMatch(
                    type=result.entity_type.lower(),
                    text=text[result.start:result.end],
                    start_pos=result.start,
                    end_pos=result.end,
                    confidence=result.score
                )
                matches.append(pii_match)
            
            return matches
            
        except Exception as e:
            # Don't fail the entire operation if ML detection fails
            return []
    
    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove duplicate matches that overlap significantly."""
        if not matches:
            return matches
        
        # Sort by start position
        matches.sort(key=lambda m: m.start_pos)
        
        deduplicated = []
        for match in matches:
            # Check if this match overlaps significantly with any existing match
            is_duplicate = False
            for existing in deduplicated:
                overlap = self._calculate_overlap(match, existing)
                if overlap > 0.8:  # 80% overlap threshold
                    # Keep the match with higher confidence
                    if match.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(match)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(match)
        
        return deduplicated
    
    def _calculate_overlap(self, match1: PIIMatch, match2: PIIMatch) -> float:
        """Calculate overlap ratio between two matches."""
        # Calculate intersection
        start = max(match1.start_pos, match2.start_pos)
        end = min(match1.end_pos, match2.end_pos)
        
        if start >= end:
            return 0.0  # No overlap
        
        intersection = end - start
        
        # Calculate union
        union = (match1.end_pos - match1.start_pos) + (match2.end_pos - match2.start_pos) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def mask_pii(
        self, 
        text: str, 
        mask_char: str = "*",
        preserve_format: bool = True
    ) -> Tuple[str, List[PIIMatch]]:
        """
        Mask PII in text.
        
        Args:
            text: Text to mask
            mask_char: Character to use for masking
            preserve_format: Whether to preserve the format of masked data
            
        Returns:
            Tuple of (masked_text, detected_pii_matches)
        """
        matches = self.detect_pii(text)
        
        if not matches:
            return text, matches
        
        # Sort matches by position (reverse order to avoid position shifts)
        matches.sort(key=lambda m: m.start_pos, reverse=True)
        
        masked_text = text
        for match in matches:
            if preserve_format:
                # Preserve format for certain types
                if match.type in ["phone_us", "ssn", "credit_card"]:
                    masked_value = self._mask_with_format(match.text, mask_char)
                else:
                    masked_value = mask_char * len(match.text)
            else:
                masked_value = mask_char * len(match.text)
            
            # Replace the PII with masked value
            masked_text = (
                masked_text[:match.start_pos] + 
                masked_value + 
                masked_text[match.end_pos:]
            )
        
        # Re-sort matches by original position
        matches.sort(key=lambda m: m.start_pos)
        
        return masked_text, matches
    
    def _mask_with_format(self, text: str, mask_char: str) -> str:
        """Mask text while preserving format characters."""
        masked = ""
        for char in text:
            if char.isalnum():
                masked += mask_char
            else:
                masked += char
        return masked
    
    def get_pii_summary(self, matches: List[PIIMatch]) -> Dict[str, Any]:
        """
        Get summary of detected PII.
        
        Args:
            matches: List of PII matches
            
        Returns:
            Summary dictionary
        """
        if not matches:
            return {
                "total_matches": 0,
                "pii_types": [],
                "high_confidence_count": 0,
                "coverage_percentage": 0.0
            }
        
        # Count by type
        type_counts = {}
        high_confidence_count = 0
        total_chars_covered = 0
        
        for match in matches:
            type_counts[match.type] = type_counts.get(match.type, 0) + 1
            if match.confidence >= 0.8:
                high_confidence_count += 1
            total_chars_covered += match.end_pos - match.start_pos
        
        return {
            "total_matches": len(matches),
            "pii_types": list(type_counts.keys()),
            "type_counts": type_counts,
            "high_confidence_count": high_confidence_count,
            "average_confidence": sum(m.confidence for m in matches) / len(matches),
            "total_chars_covered": total_chars_covered
        }
    
    def add_custom_pattern(
        self, 
        name: str, 
        pattern: str, 
        confidence: float,
        description: str = ""
    ) -> None:
        """
        Add a custom PII detection pattern.
        
        Args:
            name: Name of the PII type
            pattern: Regex pattern string
            confidence: Confidence score (0.0 to 1.0)
            description: Description of the pattern
        """
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            pii_pattern = PIIPattern(
                name=name,
                pattern=compiled_pattern,
                confidence=confidence,
                description=description or f"Custom pattern: {name}"
            )
            self.patterns.append(pii_pattern)
        except re.error as e:
            raise SecurityException(f"Invalid regex pattern: {str(e)}")
    
    def remove_pattern(self, name: str) -> bool:
        """
        Remove a PII detection pattern.
        
        Args:
            name: Name of the pattern to remove
            
        Returns:
            True if pattern was removed
        """
        original_count = len(self.patterns)
        self.patterns = [p for p in self.patterns if p.name != name]
        return len(self.patterns) < original_count
    
    def get_available_patterns(self) -> List[Dict[str, Any]]:
        """
        Get information about available PII patterns.
        
        Returns:
            List of pattern information
        """
        return [
            {
                "name": pattern.name,
                "confidence": pattern.confidence,
                "description": pattern.description
            }
            for pattern in self.patterns
        ]
    
    def health_check(self) -> bool:
        """
        Perform health check on PII detection service.
        
        Returns:
            True if service is healthy
        """
        try:
            # Test with a simple text containing known PII
            test_text = "Contact me at john.doe@example.com or call 555-123-4567"
            matches = self.detect_pii(test_text)
            
            # Should detect at least email and phone
            detected_types = [m.type for m in matches]
            return "email" in detected_types
            
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of PII detection service."""
        return (
            f"PIIDetectionService(patterns={len(self.patterns)}, "
            f"ml_enabled={self.enable_ml_detection})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of PII detection service."""
        return (
            f"PIIDetectionService(patterns={len(self.patterns)}, "
            f"ml_enabled={self.enable_ml_detection}, "
            f"ml_analyzer={'available' if self._ml_analyzer else 'not_available'})"
        )