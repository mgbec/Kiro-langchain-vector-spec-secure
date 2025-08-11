#!/usr/bin/env python3
"""
Basic usage example for the LangChain Vector Database.

This example demonstrates:
- Basic configuration
- Document ingestion
- Similarity search
- Document management
"""

import os
from pathlib import Path
from langchain_vector_db.manager import VectorDatabaseManager
from langchain_vector_db.models.config import VectorDBConfig
from langchain_vector_db.models.document import Document


def main():
    """Demonstrate basic vector database usage."""
    
    # Configure the vector database
    config = VectorDBConfig(
        storage_type="local",  # Use local FAISS storage
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002",
        local_storage_path="./vector_db_data",
        api_key=os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key
    )
    
    # Initialize the vector database manager
    manager = VectorDatabaseManager(config)
    
    try:
        print("🚀 Initializing Vector Database...")
        
        # Create sample documents
        documents = [
            Document(
                page_content="Artificial intelligence is transforming how we work and live.",
                metadata={"category": "AI", "source": "article_1"}
            ),
            Document(
                page_content="Machine learning algorithms can identify patterns in large datasets.",
                metadata={"category": "ML", "source": "article_2"}
            ),
            Document(
                page_content="Natural language processing enables computers to understand human language.",
                metadata={"category": "NLP", "source": "article_3"}
            ),
            Document(
                page_content="Vector databases are essential for similarity search applications.",
                metadata={"category": "Database", "source": "article_4"}
            ),
            Document(
                page_content="Deep learning neural networks can solve complex problems.",
                metadata={"category": "DL", "source": "article_5"}
            )
        ]
        
        # Add documents to the database
        print("📄 Adding documents to the database...")
        doc_ids = manager.add_documents(documents, generate_embeddings=True)
        print(f"✅ Added {len(doc_ids)} documents with IDs: {doc_ids}")
        
        # Get document count
        count = manager.get_document_count()
        print(f"📊 Total documents in database: {count}")
        
        # Perform similarity searches
        print("\n🔍 Performing similarity searches...")
        
        # Search 1: AI-related query
        print("\n1. Searching for 'artificial intelligence applications':")
        results = manager.similarity_search("artificial intelligence applications", k=3)
        for i, doc in enumerate(results, 1):
            print(f"   {i}. {doc.page_content[:60]}... (Category: {doc.metadata.get('category', 'N/A')})")
        
        # Search 2: Machine learning query
        print("\n2. Searching for 'machine learning patterns':")
        results = manager.similarity_search("machine learning patterns", k=2)
        for i, doc in enumerate(results, 1):
            print(f"   {i}. {doc.page_content[:60]}... (Source: {doc.metadata.get('source', 'N/A')})")
        
        # Search with scores
        print("\n3. Searching with similarity scores:")
        scored_results = manager.similarity_search_with_score("neural networks", k=3)
        for i, (doc, score) in enumerate(scored_results, 1):
            print(f"   {i}. Score: {score:.4f} - {doc.page_content[:50]}...")
        
        # Retrieve specific document
        print(f"\n📖 Retrieving specific document (ID: {doc_ids[0]}):")
        specific_doc = manager.get_document(doc_ids[0])
        if specific_doc:
            print(f"   Content: {specific_doc.page_content}")
            print(f"   Metadata: {specific_doc.metadata}")
        
        # Update a document
        print(f"\n✏️  Updating document (ID: {doc_ids[0]}):")
        update_success = manager.update_document(
            doc_ids[0],
            new_content="Artificial intelligence and machine learning are revolutionizing technology.",
            new_metadata={"category": "AI", "source": "article_1", "updated": True}
        )
        print(f"   Update successful: {update_success}")
        
        # Verify update
        updated_doc = manager.get_document(doc_ids[0])
        if updated_doc:
            print(f"   Updated content: {updated_doc.page_content}")
            print(f"   Updated metadata: {updated_doc.metadata}")
        
        # Delete a document
        print(f"\n🗑️  Deleting document (ID: {doc_ids[-1]}):")
        delete_success = manager.delete_document(doc_ids[-1])
        print(f"   Delete successful: {delete_success}")
        
        # Verify deletion
        final_count = manager.get_document_count()
        print(f"   Final document count: {final_count}")
        
        # Persist the database
        print("\n💾 Persisting database to storage...")
        persist_success = manager.persist()
        print(f"   Persistence successful: {persist_success}")
        
        # Demonstrate loading (create new manager instance)
        print("\n📂 Testing database loading...")
        new_manager = VectorDatabaseManager(config)
        load_success = new_manager.load()
        print(f"   Load successful: {load_success}")
        
        loaded_count = new_manager.get_document_count()
        print(f"   Loaded document count: {loaded_count}")
        
        # Test search on loaded database
        loaded_results = new_manager.similarity_search("artificial intelligence", k=1)
        print(f"   Search on loaded database returned {len(loaded_results)} results")
        
        new_manager.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        manager.close()
        print("✅ Vector database closed successfully")


if __name__ == "__main__":
    main()