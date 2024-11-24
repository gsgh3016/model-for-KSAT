import os
import uuid

from dotenv import load_dotenv
from loaders.document_loader import load_document
from pinecone import Pinecone
from processing.chunk_splitter import split_into_chunks
from processing.embedding_generator import generate_embeddings


def rag_preprocessing():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX")

    if not api_key or not environment or not index_name:
        raise ValueError("Pinecone API_KEY, ENVIRONMENT, or INDEX name is missing from environment variables.")

    # Initialize Pinecone and connect to the existing index
    pc = Pinecone(api_key=api_key, environment=environment)
    pinecone_index = pc.Index(index_name)
    print(f"Connected to Pinecone index: {index_name}")

    # Load and process document
    file_path = "../data/test.md"
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    embeddings = generate_embeddings(chunks, embedding_type="huggingface")

    # Prepare and upsert embeddings with unique IDs
    upsert_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        unique_id = f"{chunk.metadata.get('document_id', 'unknown')}_{i}_{uuid.uuid4()}"
        upsert_data.append(
            {
                "id": unique_id,  # 고유한 ID 생성
                "values": embedding,
                "metadata": {
                    "text": chunk.page_content,
                    "document_id": chunk.metadata.get("document_id", "unknown"),
                    "chunk_index": i,
                },
            }
        )

    # Upload embeddings
    pinecone_index.upsert(vectors=upsert_data)
    print(f"Upserted {len(embeddings)} embeddings with metadata to Pinecone index '{index_name}'.")


if __name__ == "__main__":
    rag_preprocessing()
