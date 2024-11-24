import os
import uuid

from dotenv import load_dotenv
from pinecone import Pinecone

from rag_preprocess import generate_embeddings, load_document, split_into_chunks


def rag_preprocessing():
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
    file_path = "data/test.md"
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
    load_dotenv()
    rag_preprocessing()
