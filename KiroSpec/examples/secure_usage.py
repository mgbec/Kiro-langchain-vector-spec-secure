#!/usr/bin/env python3
"""
Secure usage example for the LangChain Vector Database.

This example demonstrates:
- Security configuration
- Authentication and authorization
- Encrypted data storage
- PII detection and masking
- Audit logging
"""

import os
from pathlib import Path
from langchain_vector_db.manager import VectorDatabaseManager
from langchain_vector_db.models.config import VectorDBConfig, SecurityConfig
from langchain_vector_db.models.document import Document


def main():
    """Demonstrate secure vector database usage."""
    
    # Configure security settings
    security_config = SecurityConfig(
        auth_enabled=True,
        auth_type="api_key",
        rbac_enabled=True,
        encryption_enabled=True,
        pii_detection_enabled=True,
        audit_logging_enabled=True,
        rate_limiting_enabled=True,
        max_requests_per_minute=60,
        brute_force_threshold=5
    )
    
    # Configure the vector database with security
    config = VectorDBConfig(
        storage_type="local",
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002",
        local_storage_path="./secure_vector_db_data",
        api_key=os.getenv("OPENAI_API_KEY"),
        security=security_config
    )
    
    # Initialize the vector database manager
    manager = VectorDatabaseManager(config)
    
    try:
        print("🔐 Initializing Secure Vector Database...")
        
        # Create API keys for different users with different roles
        print("\n👤 Creating user accounts and API keys...")
        
        # Admin user - full permissions
        admin_api_key = manager.security_manager.create_api_key(
            user_id="admin_user",
            roles=["admin"],
            expires_hours=24
        )
        print(f"   Admin API key created: {admin_api_key[:20]}...")
        
        # Writer user - can create and read documents
        writer_api_key = manager.security_manager.create_api_key(
            user_id="writer_user", 
            roles=["writer"],
            expires_hours=8
        )
        print(f"   Writer API key created: {writer_api_key[:20]}...")
        
        # Reader user - can only read documents
        reader_api_key = manager.security_manager.create_api_key(
            user_id="reader_user",
            roles=["reader"],
            expires_hours=4
        )
        print(f"   Reader API key created: {reader_api_key[:20]}...")
        
        # Authenticate users
        print("\n🔑 Authenticating users...")
        
        admin_token = manager.security_manager.authenticate_api_key(
            admin_api_key, "admin_user", "192.168.1.100"
        )
        print(f"   Admin authenticated: {admin_token.user_id} with roles {admin_token.roles}")
        
        writer_token = manager.security_manager.authenticate_api_key(
            writer_api_key, "writer_user", "192.168.1.101"
        )
        print(f"   Writer authenticated: {writer_token.user_id} with roles {writer_token.roles}")
        
        reader_token = manager.security_manager.authenticate_api_key(
            reader_api_key, "reader_user", "192.168.1.102"
        )
        print(f"   Reader authenticated: {reader_token.user_id} with roles {reader_token.roles}")
        
        # Test PII detection and masking
        print("\n🕵️  Testing PII detection and masking...")
        
        sensitive_text = \"\"\"John Doe's email is john.doe@company.com and his phone number is 555-123-4567. \n        His SSN is 123-45-6789 and credit card number is 4111-1111-1111-1111.\"\"\"\n        \n        pii_matches = manager.security_manager.detect_pii(sensitive_text)\n        print(f\"   Detected {len(pii_matches)} PII items:\")\n        for match in pii_matches:\n            print(f\"     - {match.type}: {match.value} at position {match.start}-{match.end}\")\n        \n        masked_text = manager.security_manager.mask_sensitive_data(sensitive_text)\n        print(f\"   Masked text: {masked_text}\")\n        \n        # Create documents with different sensitivity levels\n        print(\"\\n📄 Creating documents with different security classifications...\")\n        \n        documents = [\n            Document(\n                page_content=\"Public information about artificial intelligence trends.\",\n                metadata={\"classification\": \"public\", \"department\": \"research\"}\n            ),\n            Document(\n                page_content=\"Internal company data about machine learning projects.\",\n                metadata={\"classification\": \"internal\", \"department\": \"engineering\"}\n            ),\n            Document(\n                page_content=masked_text,  # Use masked version of sensitive data\n                metadata={\"classification\": \"confidential\", \"department\": \"hr\"}\n            )\n        ]\n        \n        # Add documents as admin user\n        print(\"\\n✅ Adding documents as admin user...\")\n        doc_ids = manager.add_documents(\n            documents,\n            generate_embeddings=True,\n            auth_token=admin_token,\n            user_id=\"admin_user\",\n            ip_address=\"192.168.1.100\"\n        )\n        print(f\"   Added {len(doc_ids)} documents: {doc_ids}\")\n        \n        # Test writer permissions\n        print(\"\\n✏️  Testing writer permissions...\")\n        writer_document = Document(\n            page_content=\"Writer-created document about natural language processing.\",\n            metadata={\"classification\": \"internal\", \"author\": \"writer_user\"}\n        )\n        \n        writer_doc_ids = manager.add_documents(\n            [writer_document],\n            auth_token=writer_token,\n            user_id=\"writer_user\",\n            ip_address=\"192.168.1.101\"\n        )\n        print(f\"   Writer successfully added document: {writer_doc_ids[0]}\")\n        \n        # Test reader permissions (should fail to write)\n        print(\"\\n🚫 Testing reader permission restrictions...\")\n        try:\n            reader_document = Document(\n                page_content=\"Reader attempting to create document.\",\n                metadata={\"classification\": \"public\"}\n            )\n            \n            manager.add_documents(\n                [reader_document],\n                auth_token=reader_token,\n                user_id=\"reader_user\"\n            )\n            print(\"   ❌ ERROR: Reader should not be able to create documents!\")\n        except Exception as e:\n            print(f\"   ✅ Reader correctly denied write access: {str(e)[:60]}...\")\n        \n        # Test search permissions (all users should be able to search)\n        print(\"\\n🔍 Testing search permissions...\")\n        \n        # Admin search\n        admin_results = manager.similarity_search(\n            \"artificial intelligence\",\n            k=2,\n            auth_token=admin_token,\n            user_id=\"admin_user\"\n        )\n        print(f\"   Admin search returned {len(admin_results)} results\")\n        \n        # Writer search\n        writer_results = manager.similarity_search(\n            \"machine learning\",\n            k=2,\n            auth_token=writer_token,\n            user_id=\"writer_user\"\n        )\n        print(f\"   Writer search returned {len(writer_results)} results\")\n        \n        # Reader search\n        reader_results = manager.similarity_search(\n            \"natural language\",\n            k=2,\n            auth_token=reader_token,\n            user_id=\"reader_user\"\n        )\n        print(f\"   Reader search returned {len(reader_results)} results\")\n        \n        # Test document updates (only admin and writer should succeed)\n        print(\"\\n🔄 Testing document update permissions...\")\n        \n        # Admin update\n        admin_update_success = manager.update_document(\n            doc_ids[0],\n            new_content=\"Updated public information about AI trends and applications.\",\n            auth_token=admin_token,\n            user_id=\"admin_user\"\n        )\n        print(f\"   Admin update successful: {admin_update_success}\")\n        \n        # Writer update\n        writer_update_success = manager.update_document(\n            writer_doc_ids[0],\n            new_content=\"Updated writer document about advanced NLP techniques.\",\n            auth_token=writer_token,\n            user_id=\"writer_user\"\n        )\n        print(f\"   Writer update successful: {writer_update_success}\")\n        \n        # Reader update (should fail)\n        try:\n            manager.update_document(\n                doc_ids[0],\n                new_content=\"Reader attempting to update document.\",\n                auth_token=reader_token,\n                user_id=\"reader_user\"\n            )\n            print(\"   ❌ ERROR: Reader should not be able to update documents!\")\n        except Exception as e:\n            print(f\"   ✅ Reader correctly denied update access: {str(e)[:60]}...\")\n        \n        # Test rate limiting\n        print(\"\\n⏱️  Testing rate limiting...\")\n        rapid_requests = 0\n        rate_limited = 0\n        \n        for i in range(10):\n            try:\n                manager.similarity_search(\n                    f\"test query {i}\",\n                    k=1,\n                    auth_token=reader_token,\n                    user_id=\"reader_user\",\n                    ip_address=\"192.168.1.102\"\n                )\n                rapid_requests += 1\n            except Exception as e:\n                if \"rate limit\" in str(e).lower():\n                    rate_limited += 1\n                    break\n        \n        print(f\"   Completed {rapid_requests} requests before rate limiting\")\n        if rate_limited > 0:\n            print(\"   ✅ Rate limiting is working correctly\")\n        \n        # Check audit logs\n        print(\"\\n📋 Checking audit logs...\")\n        audit_events = manager.security_manager.get_audit_events(limit=10)\n        print(f\"   Found {len(audit_events)} recent audit events:\")\n        \n        for event in audit_events[-5:]:  # Show last 5 events\n            print(f\"     - {event['timestamp']}: {event['user_id']} performed {event['operation']} (Result: {event['result']})\")\n        \n        # Check security alerts\n        print(\"\\n🚨 Checking security alerts...\")\n        security_alerts = manager.security_manager.get_security_alerts()\n        if security_alerts:\n            print(f\"   Found {len(security_alerts)} security alerts:\")\n            for alert in security_alerts[-3:]:  # Show last 3 alerts\n                print(f\"     - {alert['timestamp']}: {alert['alert_type']} - {alert['description']}\")\n        else:\n            print(\"   No security alerts found\")\n        \n        # Test encryption (data at rest)\n        print(\"\\n🔒 Testing data encryption...\")\n        test_data = b\"Sensitive data that needs encryption\"\n        encrypted_data = manager.security_manager.encrypt_data(test_data)\n        decrypted_data = manager.security_manager.decrypt_data(encrypted_data)\n        \n        print(f\"   Original data length: {len(test_data)} bytes\")\n        print(f\"   Encrypted data length: {len(encrypted_data)} bytes\")\n        print(f\"   Decryption successful: {decrypted_data == test_data}\")\n        \n        # Persist with encryption\n        print(\"\\n💾 Persisting encrypted database...\")\n        persist_success = manager.persist()\n        print(f\"   Secure persistence successful: {persist_success}\")\n        \n    except Exception as e:\n        print(f\"❌ Error: {e}\")\n        raise\n    \n    finally:\n        # Clean up\n        print(\"\\n🧹 Cleaning up...\")\n        manager.close()\n        print(\"✅ Secure vector database closed successfully\")\n\n\nif __name__ == \"__main__\":\n    main()