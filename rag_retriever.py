import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils import get_pinecone_index


def rag_retrieval():
    # Pinecone index에 연결
    pinecone_index, index_name = get_pinecone_index()

    # HuggingFace Embeddings 모델 초기화
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # LangChain의 Pinecone VectorStore 생성
    vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key="text")

    # 간단한 Query를 사용해 문서 검색
    query = "초전도체가 뭐야?"
    retrieved_docs = vector_store.similarity_search(query, k=5)  # k는 반환할 문서 개수

    # 검색된 문서 출력
    print("Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i + 1}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")

    # VectorStore로 Retriever 객체 생성
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 생성 모델 불러오기
    model_id = "google/gemma-2-2b-it"

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device=0 if torch.cuda.is_available() else -1,
    )

    llm = HuggingFacePipeline(pipeline=gen)

    template = (
        "당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. "
        "당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.\n"
        "검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, "
        "답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요. "
        "한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.\n\n"
        "# Question:\n"
        "{question}\n\n"
        "# Context:\n"
        "{context}\n\n"
        "# Answer:\n"
    )

    prev_chat = []

    chat = [*prev_chat, {"role": "user", "content": template}]

    prompt_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

    # 체인 생성
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    answer = rag_chain.invoke(query)

    print(answer)


if __name__ == "__main__":
    rag_retrieval()
