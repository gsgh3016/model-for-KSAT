from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings


def generate_embeddings(chunks, embedding_type="huggingface"):
    if embedding_type == "openai":
        embeddings = OpenAIEmbeddings()
    elif embedding_type == "huggingface":
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError("Invalid embedding type specified.")
    return embeddings.embed_documents([chunk.page_content for chunk in chunks])
