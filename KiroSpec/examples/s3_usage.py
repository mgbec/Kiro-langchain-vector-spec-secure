#!/usr/bin/env python3
"""
AWS S3 usage example for the LangChain Vector Database.

This example demonstrates:
- S3 storage configuration
- Document storage in S3
- S3-specific features and optimizations
- Error handling for S3 operations
"""

import os
from pathlib import Path
from langchain_vector_db.manager import VectorDatabaseManager
from langchain_vector_db.models.config import VectorDBConfig
from langchain_vector_db.models.document import Document


def main():
    """Demonstrate S3 vector database usage."""
    
    # Configure S3 storage
    config = VectorDBConfig(
        storage_type="s3",
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002",
        api_key=os.getenv("OPENAI_API_KEY"),
        
        # S3 configuration
        s3_bucket_name=os.getenv("S3_BUCKET_NAME", "my-vector-db-bucket"),
        s3_key_prefix="vector_db/",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        
        # S3 optimization settings
        s3_batch_size=100,
        s3_retry_attempts=3,
        s3_timeout_seconds=30
    )
    
    # Initialize the vector database manager
    manager = VectorDatabaseManager(config)
    
    try:
        print("☁️  Initializing S3 Vector Database...")
        print(f"   Bucket: {config.s3_bucket_name}")
        print(f"   Key prefix: {config.s3_key_prefix}")
        print(f"   Region: {config.aws_region}")
        
        # Test S3 connection
        print("\\n🔗 Testing S3 connection...")
        health_status = manager.health_check()
        if health_status:
            print("   ✅ S3 connection successful")
        else:
            print("   ❌ S3 connection failed")
            return
        
        # Create sample documents for S3 storage
        print("\\n📄 Creating sample documents for S3 storage...")
        documents = [
            Document(
                page_content="Cloud storage enables scalable and durable data persistence.",
                metadata={"category": "cloud", "storage_type": "s3", "region": config.aws_region}
            ),
            Document(
                page_content="Amazon S3 provides highly available object storage with global accessibility.",
                metadata={"category": "aws", "service": "s3", "availability": "high"}
            ),
            Document(
                page_content="Vector databases benefit from cloud storage for distributed deployments.",
                metadata={"category": "database", "deployment": "distributed", "storage": "cloud"}
            ),
            Document(
                page_content="S3 bucket policies and IAM roles provide fine-grained access control.",
                metadata={"category": "security", "service": "s3", "feature": "access_control"}
            ),
            Document(
                page_content="Cross-region replication ensures data durability and disaster recovery.",
                metadata={"category": "reliability", "feature": "replication", "scope": "cross_region"}
            )
        ]
        
        # Add documents to S3
        print("\\n📤 Uploading documents to S3...")
        upload_start = time.time() if 'time' in globals() else 0
        
        doc_ids = manager.add_documents(documents, generate_embeddings=True)
        
        if 'time' in globals():
            upload_duration = time.time() - upload_start
            print(f"   Uploaded {len(doc_ids)} documents in {upload_duration:.3f} seconds")
        else:
            print(f"   Uploaded {len(doc_ids)} documents")
        
        print(f"   Document IDs: {doc_ids}")
        
        # Get document count from S3
        print("\\n📊 Checking document count in S3...")
        count = manager.get_document_count()
        print(f"   Total documents in S3: {count}")
        
        # Perform similarity searches on S3 data
        print("\\n🔍 Performing similarity searches on S3 data...")
        
        search_queries = [
            "cloud storage scalability",
            "Amazon S3 availability",
            "distributed vector database",
            "S3 security and access control",
            "data replication and recovery"
        ]
        
        for i, query in enumerate(search_queries, 1):
            print(f"\\n   Search {i}: '{query}'")
            results = manager.similarity_search(query, k=2)
            
            for j, doc in enumerate(results, 1):
                category = doc.metadata.get('category', 'N/A')
                print(f"     {j}. {doc.page_content[:60]}... (Category: {category})")
        
        # Test S3-specific operations
        print("\\n🔧 Testing S3-specific operations...")
        
        # Retrieve specific document from S3
        print(f"\\n📖 Retrieving document from S3 (ID: {doc_ids[0]}):")
        retrieved_doc = manager.get_document(doc_ids[0])
        if retrieved_doc:
            print(f"   Content: {retrieved_doc.page_content}")
            print(f"   Metadata: {retrieved_doc.metadata}")
            print(f"   Storage region: {retrieved_doc.metadata.get('region', 'N/A')}")
        
        # Update document in S3
        print(f"\\n✏️  Updating document in S3 (ID: {doc_ids[0]}):")
        update_success = manager.update_document(
            doc_ids[0],
            new_content="Updated: Cloud storage with S3 enables highly scalable and durable data persistence with global accessibility.",
            new_metadata={
                "category": "cloud",
                "storage_type": "s3",
                "region": config.aws_region,
                "updated": True,
                "version": "2.0"
            }
        )
        print(f"   S3 update successful: {update_success}")
        
        # Verify update in S3
        updated_doc = manager.get_document(doc_ids[0])
        if updated_doc:
            print(f"   Updated content: {updated_doc.page_content[:80]}...")
            print(f"   Version: {updated_doc.metadata.get('version', 'N/A')}")
        
        # Test batch operations with S3
        print("\\n📦 Testing batch operations with S3...")
        
        batch_documents = [
            Document(
                page_content=f"Batch document {i} for S3 storage testing.",
                metadata={"batch": True, "doc_number": i, "storage": "s3"}
            )
            for i in range(1, 6)
        ]
        
        batch_doc_ids = manager.add_documents(batch_documents, generate_embeddings=True)
        print(f"   Added {len(batch_doc_ids)} documents in batch to S3")
        
        # Search batch documents
        batch_results = manager.similarity_search("batch document S3", k=3)
        print(f"   Batch search returned {len(batch_results)} results")
        
        # Test S3 persistence and loading
        print("\\n💾 Testing S3 persistence...")
        persist_success = manager.persist()
        print(f"   S3 persistence successful: {persist_success}")
        
        # Test loading from S3 (create new manager instance)
        print("\\n📂 Testing S3 loading with new manager instance...")
        new_manager = VectorDatabaseManager(config)
        
        try:
            load_success = new_manager.load()
            print(f"   S3 load successful: {load_success}")
            
            # Verify loaded data
            loaded_count = new_manager.get_document_count()
            print(f"   Loaded document count from S3: {loaded_count}")
            
            # Test search on loaded S3 data
            loaded_results = new_manager.similarity_search("cloud storage", k=2)
            print(f"   Search on loaded S3 data returned {len(loaded_results)} results")
            
            for i, doc in enumerate(loaded_results, 1):
                print(f"     {i}. {doc.page_content[:50]}...")
        
        finally:
            new_manager.close()
        
        # Test S3 error handling
        print("\\n🚨 Testing S3 error handling...")
        
        # Test with invalid document ID
        try:
            invalid_doc = manager.get_document("invalid_doc_id_12345")
            if invalid_doc is None:
                print("   ✅ Correctly handled invalid document ID")
            else:
                print("   ❌ Should have returned None for invalid ID")
        except Exception as e:
            print(f"   ✅ Correctly handled invalid document ID with exception: {str(e)[:50]}...")
        
        # Test deletion from S3
        print(f"\\n🗑️  Testing document deletion from S3 (ID: {batch_doc_ids[-1]}):")
        delete_success = manager.delete_document(batch_doc_ids[-1])
        print(f"   S3 deletion successful: {delete_success}")
        
        # Verify deletion
        deleted_doc = manager.get_document(batch_doc_ids[-1])
        if deleted_doc is None:
            print("   ✅ Document successfully deleted from S3")
        else:
            print("   ❌ Document still exists in S3")
        
        # Final S3 statistics
        print("\\n📈 Final S3 statistics...")
        final_count = manager.get_document_count()
        print(f"   Final document count in S3: {final_count}")
        
        # Get S3 storage statistics if available
        try:
            stats = manager.get_stats()
            if stats:
                print("   S3 storage statistics:")
                for key, value in stats.items():
                    if key.startswith('s3_') or key in ['storage_size_mb', 'total_documents']:
                        print(f"     - {key}: {value}")
        except Exception as e:
            print(f"   Could not retrieve S3 statistics: {str(e)[:50]}...")
        
        # Test S3 cleanup operations
        print("\\n🧹 Testing S3 cleanup operations...")
        
        # Clear some documents (be careful in production!)
        if len(doc_ids) > 3:
            cleanup_ids = doc_ids[-2:]  # Delete last 2 documents
            for doc_id in cleanup_ids:
                cleanup_success = manager.delete_document(doc_id)
                print(f"   Cleaned up document {doc_id}: {cleanup_success}")
        
        # Final persistence
        print("\\n💾 Final S3 persistence...")
        final_persist = manager.persist()
        print(f"   Final S3 persistence successful: {final_persist}")
        
    except Exception as e:
        print(f"❌ S3 Error: {e}")
        print("\\n💡 Troubleshooting tips:")
        print("   - Ensure AWS credentials are properly configured")
        print("   - Verify S3 bucket exists and is accessible")
        print("   - Check IAM permissions for S3 operations")
        print("   - Confirm AWS region is correct")
        print("   - Validate network connectivity to S3")
        raise
    
    finally:
        # Clean up S3 resources
        print("\\n🧹 Cleaning up S3 resources...")
        manager.close()
        print("✅ S3 vector database closed successfully")


if __name__ == "__main__":
    # Import time if available for timing measurements
    try:
        import time
    except ImportError:
        pass
    
    main()