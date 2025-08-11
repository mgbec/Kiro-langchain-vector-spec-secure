"""
PII (Personally Identifiable Information) detection data models.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PIIMatch:
    """Represents a detected PII match in text."""
    
    type: str  # "email", "phone", "ssn", "credit_card", etc.
    text: str  # The actual matched text
    start_pos: int  # Start position in the original text
    end_pos: int  # End position in the original text
    confidence: float  # Confidence score (0.0 to 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PII match to dictionary."""
        return {
            "type": self.type,
            "text": self.text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
        }
    
    def get_masked_text(self, mask_char: str = "*") -> str:
        """Get the text with PII masked."""
        return mask_char * len(self.text)