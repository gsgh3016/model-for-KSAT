from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore

from utils import get_pinecone_index


def get_retriever(embedding_model: str = "BAAI/bge-m3", k: int = 5):
    """
    Pinecone에 연결된 Retriever를 생성하고 반환합니다.

    Args:
        embedding_model (str): 임베딩에 사용할 허깅페이스 모델 이름
        k (int): similarity search로 불러올 문서 개수

    Returns:
        LangChain retriever
    """

    # Pinecone index 연결
    pinecone_index, index_name = get_pinecone_index()

    # HuggingFace Embeddings 모델 초기화
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model)

    # LangChain의 Pinecone VectorStore 생성
    vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key="text")

    # Retriever 반환
    return vector_store.as_retriever(search_kwargs={"k": k})
