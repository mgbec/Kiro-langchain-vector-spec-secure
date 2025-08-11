"""
Unit tests for EncryptionService.
"""

import pytest
import base64

from langchain_vector_db.services.encryption import EncryptionService
from langchain_vector_db.exceptions import EncryptionException


class TestEncryptionService:
    """Test cases for EncryptionService."""
    
    def test_initialization_with_defaults(self):
        """Test encryption service initialization with defaults."""
        service = EncryptionService()
        
        assert service.algorithm == "AES-256-GCM"
        assert service._encryption_key is not None
        assert len(service._encryption_key) > 0
    
    def test_initialization_with_custom_key(self):
        """Test initialization with custom encryption key."""
        custom_key = "my_custom_encryption_key_12345"
        service = EncryptionService(encryption_key=custom_key)
        
        assert service._encryption_key == custom_key
    
    def test_initialization_with_fernet(self):
        """Test initialization with Fernet algorithm."""
        service = EncryptionService(algorithm="Fernet")
        
        assert service.algorithm == "Fernet"
        assert service._cipher is not None
    
    def test_initialization_with_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with pytest.raises(EncryptionException) as exc_info:
            EncryptionService(algorithm="INVALID")
        
        assert "Unsupported encryption algorithm" in str(exc_info.value)
    
    def test_encrypt_decrypt_string_aes_gcm(self):
        """Test encryption and decryption of string data with AES-GCM."""
        service = EncryptionService(algorithm="AES-256-GCM")
        
        original_data = "This is sensitive data that needs encryption"
        
        # Encrypt data
        encrypted_data = service.encrypt(original_data)
        
        assert isinstance(encrypted_data, str)
        assert encrypted_data != original_data
        assert len(encrypted_data) > 0
        
        # Decrypt data
        decrypted_data = service.decrypt_to_string(encrypted_data)
        
        assert decrypted_data == original_data
    
    def test_encrypt_decrypt_bytes_aes_gcm(self):
        """Test encryption and decryption of bytes data with AES-GCM."""
        service = EncryptionService(algorithm="AES-256-GCM")
        
        original_data = b"Binary data to encrypt"
        
        # Encrypt data
        encrypted_data = service.encrypt(original_data)
        
        assert isinstance(encrypted_data, str)
        assert encrypted_data != original_data.decode('utf-8')
        
        # Decrypt data
        decrypted_data = service.decrypt(encrypted_data)
        
        assert decrypted_data == original_data
    
    def test_encrypt_decrypt_string_fernet(self):
        """Test encryption and decryption with Fernet."""
        service = EncryptionService(algorithm="Fernet")
        
        original_data = "Fernet encryption test data"
        
        # Encrypt data
        encrypted_data = service.encrypt(original_data)
        
        assert isinstance(encrypted_data, str)
        assert encrypted_data != original_data
        
        # Decrypt data
        decrypted_data = service.decrypt_to_string(encrypted_data)
        
        assert decrypted_data == original_data
    
    def test_encrypt_empty_string(self):
        """Test encryption of empty string."""
        service = EncryptionService()
        
        encrypted_data = service.encrypt("")
        decrypted_data = service.decrypt_to_string(encrypted_data)
        
        assert decrypted_data == ""
    
    def test_encrypt_unicode_data(self):
        """Test encryption of unicode data."""
        service = EncryptionService()
        
        original_data = "Unicode test: 你好世界 🌍 émojis"
        
        encrypted_data = service.encrypt(original_data)
        decrypted_data = service.decrypt_to_string(encrypted_data)
        
        assert decrypted_data == original_data
    
    def test_decrypt_invalid_data(self):
        """Test decryption of invalid data."""
        service = EncryptionService()
        
        with pytest.raises(EncryptionException) as exc_info:
            service.decrypt("invalid_encrypted_data")
        
        assert "Decryption failed" in str(exc_info.value)
    
    def test_decrypt_corrupted_data(self):
        """Test decryption of corrupted data."""
        service = EncryptionService()
        
        # Encrypt valid data
        original_data = "Test data"
        encrypted_data = service.encrypt(original_data)
        
        # Corrupt the encrypted data
        corrupted_data = encrypted_data[:-5] + "XXXXX"
        
        with pytest.raises(EncryptionException):
            service.decrypt(corrupted_data)
    
    def test_generate_new_key(self):
        """Test generating a new encryption key."""
        service = EncryptionService()
        
        new_key = service.generate_new_key()
        
        assert isinstance(new_key, str)
        assert len(new_key) > 0
        assert new_key != service._encryption_key
    
    def test_rotate_key(self):
        """Test key rotation."""
        service = EncryptionService()
        
        # Encrypt data with original key
        original_data = "Data encrypted with original key"
        encrypted_with_old_key = service.encrypt(original_data)
        
        # Rotate key
        new_key = service.generate_new_key()
        service.rotate_key(new_key)
        
        assert service._encryption_key == new_key
        
        # Encrypt data with new key
        encrypted_with_new_key = service.encrypt(original_data)
        
        # Should be different ciphertexts
        assert encrypted_with_old_key != encrypted_with_new_key
        
        # New key should decrypt new data
        decrypted_data = service.decrypt_to_string(encrypted_with_new_key)
        assert decrypted_data == original_data
    
    def test_get_key_info(self):
        """Test getting key information."""
        service = EncryptionService()
        
        key_info = service.get_key_info()
        
        assert "algorithm" in key_info
        assert "key_length" in key_info
        assert "key_hash" in key_info
        assert key_info["algorithm"] == "AES-256-GCM"
        assert key_info["key_length"] > 0
        assert len(key_info["key_hash"]) == 16  # First 16 chars of hash
    
    def test_different_keys_produce_different_results(self):
        """Test that different keys produce different encrypted results."""
        service1 = EncryptionService(encryption_key="key1")
        service2 = EncryptionService(encryption_key="key2")
        
        data = "Same data, different keys"
        
        encrypted1 = service1.encrypt(data)
        encrypted2 = service2.encrypt(data)
        
        # Should produce different ciphertexts
        assert encrypted1 != encrypted2
        
        # Each should decrypt correctly with its own key
        assert service1.decrypt_to_string(encrypted1) == data
        assert service2.decrypt_to_string(encrypted2) == data
    
    def test_same_key_different_instances(self):
        """Test that same key in different instances works correctly."""
        key = "shared_encryption_key"
        service1 = EncryptionService(encryption_key=key)
        service2 = EncryptionService(encryption_key=key)
        
        data = "Data encrypted by service1"
        
        # Encrypt with service1
        encrypted_data = service1.encrypt(data)
        
        # Decrypt with service2
        decrypted_data = service2.decrypt_to_string(encrypted_data)
        
        assert decrypted_data == data
    
    def test_large_data_encryption(self):
        """Test encryption of large data."""
        service = EncryptionService()
        
        # Create large data (1MB)
        large_data = "A" * (1024 * 1024)
        
        encrypted_data = service.encrypt(large_data)
        decrypted_data = service.decrypt_to_string(encrypted_data)
        
        assert decrypted_data == large_data
    
    def test_multiple_encrypt_decrypt_cycles(self):
        """Test multiple encryption/decryption cycles."""
        service = EncryptionService()
        
        original_data = "Test data for multiple cycles"
        current_data = original_data
        
        # Perform multiple encrypt/decrypt cycles
        for i in range(5):
            encrypted = service.encrypt(current_data)
            current_data = service.decrypt_to_string(encrypted)
            assert current_data == original_data
    
    def test_base64_encoding_in_output(self):
        """Test that encrypted output is valid base64."""
        service = EncryptionService()
        
        data = "Test data for base64 validation"
        encrypted_data = service.encrypt(data)
        
        # Should be valid base64
        try:
            base64.b64decode(encrypted_data)
        except Exception:
            pytest.fail("Encrypted data is not valid base64")


if __name__ == "__main__":
    pytest.main([__file__])