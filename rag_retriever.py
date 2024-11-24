import os

from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as pine
from pinecone import Pinecone


def rag_retrieval():
    # 환경 변수에서 Pinecone 설정 정보 가져오기
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX")

    # 필수 환경 변수 확인
    if not api_key or not environment or not index_name:
        raise ValueError("Pinecone API_KEY, ENVIRONMENT, or INDEX name is missing from environment variables.")

    # 1. Pinecone 초기화 및 연결
    pc = Pinecone(api_key=api_key, environment=environment)
    pinecone_index = pc.Index(index_name)
    print(f"Connected to Pinecone index: {index_name}")

    # 2. HuggingFace Embeddings 모델 초기화
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. LangChain의 Pinecone VectorStore 생성
    vector_store = pine(index=pinecone_index, embedding=embeddings.embed_query, text_key="text")

    # 4. 간단한 Query를 사용해 문서 검색
    query = "명탐정 코난은 무슨 만화야?"
    retrieved_docs = vector_store.similarity_search(query, k=5)  # k는 반환할 문서 개수

    # 5. 검색된 문서 출력
    print("Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i + 1}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")


if __name__ == "__main__":
    load_dotenv()
    rag_retrieval()
