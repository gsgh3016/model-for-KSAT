import os

from dotenv import load_dotenv
from loaders.document_loader import load_document
from pinecone import Pinecone
from processing.chunk_splitter import split_into_chunks
from processing.embedding_generator import generate_embeddings


def rag_preprocessing():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX")

    if not api_key or not environment or not index_name:
        raise ValueError("Pinecone API_KEY, ENVIRONMENT, or INDEX name is missing from environment variables.")

    # 1. Initialize Pinecone and connect to the existing index
    pc = Pinecone(api_key=api_key, environment=environment)

    pinecone_index = pc.Index(index_name)
    print(f"Connected to Pinecone index: {index_name}")

    # 2. Load and process document
    file_path = "../data/test.md"
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    embeddings = generate_embeddings(chunks, embedding_type="huggingface")

    # 3. Prepare and upsert embeddings with metadata
    upsert_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        upsert_data.append(
            {
                "id": str(i),
                "values": embedding,
                "metadata": {
                    "text": chunk.page_content,  # 메타데이터에 문단 텍스트 추가
                    "document_id": chunk.metadata.get("document_id", "unknown"),  # 문서 ID가 있다면 추가
                    "chunk_index": i,  # 각 chunk의 인덱스
                },
            }
        )

    pinecone_index.upsert(vectors=upsert_data)
    print(f"Upserted {len(embeddings)} embeddings with metadata to Pinecone index '{index_name}'.")


if __name__ == "__main__":
    rag_preprocessing()
