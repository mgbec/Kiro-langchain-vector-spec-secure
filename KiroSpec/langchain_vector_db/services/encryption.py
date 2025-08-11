"""
Data encryption and decryption services.
"""

import os
import base64
import secrets
from typing import Union, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..exceptions import EncryptionException


class EncryptionService:
    """Service for encrypting and decrypting data."""
    
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        algorithm: str = "AES-256-GCM"
    ):
        """
        Initialize encryption service.
        
        Args:
            encryption_key: Base encryption key (will be derived if needed)
            algorithm: Encryption algorithm to use
        """
        self.algorithm = algorithm
        self._encryption_key = encryption_key or self._generate_key()
        
        # Initialize cipher based on algorithm
        if algorithm == "AES-256-GCM":
            self._cipher = self._init_aes_gcm()
        elif algorithm == "Fernet":
            self._cipher = self._init_fernet()
        else:
            raise EncryptionException(f"Unsupported encryption algorithm: {algorithm}")
    
    def _generate_key(self) -> str:
        """Generate a secure encryption key."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')
    
    def _init_aes_gcm(self) -> AESGCM:
        """Initialize AES-GCM cipher."""
        # Derive 32-byte key from the provided key
        key_bytes = self._derive_key(self._encryption_key.encode('utf-8'), b'aes_gcm_salt')
        return AESGCM(key_bytes)
    
    def _init_fernet(self) -> Fernet:
        """Initialize Fernet cipher."""
        # Derive Fernet-compatible key
        key_bytes = self._derive_key(self._encryption_key.encode('utf-8'), b'fernet_salt')
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        return Fernet(fernet_key)
    
    def _derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive encryption key using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password)
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Base64-encoded encrypted data
            
        Raises:
            EncryptionException: If encryption fails
        """
        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            if self.algorithm == "AES-256-GCM":
                return self._encrypt_aes_gcm(data_bytes)
            elif self.algorithm == "Fernet":
                return self._encrypt_fernet(data_bytes)
            else:
                raise EncryptionException(f"Encryption not implemented for {self.algorithm}")
                
        except Exception as e:
            if isinstance(e, EncryptionException):
                raise
            raise EncryptionException(f"Encryption failed: {str(e)}")
    
    def decrypt(self, encrypted_data: str) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted data as bytes
            
        Raises:
            EncryptionException: If decryption fails
        """
        try:
            if self.algorithm == "AES-256-GCM":
                return self._decrypt_aes_gcm(encrypted_data)
            elif self.algorithm == "Fernet":
                return self._decrypt_fernet(encrypted_data)
            else:
                raise EncryptionException(f"Decryption not implemented for {self.algorithm}")
                
        except Exception as e:
            if isinstance(e, EncryptionException):
                raise
            raise EncryptionException(f"Decryption failed: {str(e)}")
    
    def decrypt_to_string(self, encrypted_data: str) -> str:
        """
        Decrypt data and return as string.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted data as string
        """
        decrypted_bytes = self.decrypt(encrypted_data)
        return decrypted_bytes.decode('utf-8')
    
    def _encrypt_aes_gcm(self, data: bytes) -> str:
        """Encrypt using AES-GCM."""
        # Generate random nonce
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        
        # Encrypt data
        ciphertext = self._cipher.encrypt(nonce, data, None)
        
        # Combine nonce and ciphertext
        encrypted_data = nonce + ciphertext
        
        # Return base64-encoded result
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def _decrypt_aes_gcm(self, encrypted_data: str) -> bytes:
        """Decrypt using AES-GCM."""
        # Decode base64
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        
        # Extract nonce and ciphertext
        nonce = encrypted_bytes[:12]  # First 12 bytes are nonce
        ciphertext = encrypted_bytes[12:]  # Rest is ciphertext
        
        # Decrypt data
        return self._cipher.decrypt(nonce, ciphertext, None)
    
    def _encrypt_fernet(self, data: bytes) -> str:
        """Encrypt using Fernet."""
        encrypted_data = self._cipher.encrypt(data)
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def _decrypt_fernet(self, encrypted_data: str) -> bytes:
        """Decrypt using Fernet."""
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        return self._cipher.decrypt(encrypted_bytes)
    
    def generate_new_key(self) -> str:
        """
        Generate a new encryption key.
        
        Returns:
            New encryption key
        """
        return self._generate_key()
    
    def rotate_key(self, new_key: str) -> None:
        """
        Rotate to a new encryption key.
        
        Args:
            new_key: New encryption key
            
        Note:
            This only updates the key for future operations.
            Existing encrypted data will need to be re-encrypted.
        """
        self._encryption_key = new_key
        
        # Reinitialize cipher with new key
        if self.algorithm == "AES-256-GCM":
            self._cipher = self._init_aes_gcm()
        elif self.algorithm == "Fernet":
            self._cipher = self._init_fernet()
    
    def get_key_info(self) -> dict:
        """
        Get information about the current encryption key.
        
        Returns:
            Dictionary with key information (without exposing the actual key)
        """
        return {
            "algorithm": self.algorithm,
            "key_length": len(self._encryption_key),
            "key_hash": hashlib.sha256(self._encryption_key.encode()).hexdigest()[:16]
        }


# Import hashlib for key info
import hashlib