import uuid

from rag_modules import generate_embeddings, load_document, split_into_chunks
from utils import get_pinecone_index


def rag_preprocessing():
    # Pinecone index에 연결
    pinecone_index, index_name = get_pinecone_index()

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
    rag_preprocessing()
