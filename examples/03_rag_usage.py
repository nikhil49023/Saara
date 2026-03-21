"""
Example 3: RAG (Retrieval-Augmented Generation)

Shows how to use SAARA's RAG engine to build an intelligent
document retrieval and question-answering system.
"""

from pathlib import Path
from saara import RAGEngine, RAGConfig


def main():
    """RAG example."""

    # Create RAG configuration
    config = RAGConfig(
        vector_store="chromadb",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=512,
        top_k=5,
        similarity_threshold=0.5,
        hybrid_search=True,
    )

    # Initialize RAG engine
    rag = RAGEngine(config)

    # Index documents
    documents = [
        "Your document text 1...",
        "Your document text 2...",
        "Your document text 3...",
    ]

    print("Indexing documents...")
    rag.index_documents(documents)
    print(f"✓ Indexed {len(documents)} documents")

    # Query the RAG engine
    query = "What is the main topic of the documents?"
    print(f"\nQuery: {query}")

    results = rag.search(query)
    print(f"✓ Found {len(results)} relevant documents")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.2f}")
        print(f"   Text: {result['text'][:100]}...")


if __name__ == "__main__":
    main()
