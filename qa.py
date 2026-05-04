from dotenv import load_dotenv
import os

import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


def build_rag_chain():
    load_dotenv()
    os.environ["LANGCHAIN_PROJECT"] = "RAG TUTORIAL"

    local_model_name = "qwen3-8b"
    local_base_url = "http://202.31.200.130:8001/v1"
    local_api_key = "not_used"
    embedding_model_name = "BAAI/bge-m3"

    loader = WebBaseLoader(
        web_paths=("https://n.news.naver.com/article/437/0000378416",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
            )
        ),
    )

    docs = loader.load()
    print(f"문서의 수: {len(docs)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question}\nContext: {context}\nAnswer:"
    )

    llm = ChatOpenAI(
        model_name=local_model_name,
        base_url=local_base_url,
        api_key=local_api_key,
        temperature=0,
    )

    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    rag_chain = build_rag_chain()

    q1 = "부영그룹의 출산 장려 정책에 대해 설명해주세요."
    q2 = "부영그룹에 대해 설명해주세요."

    print("\nQ1:", q1)
    print("A1:", rag_chain.invoke(q1))

    print("\nQ2:", q2)
    print("A2:", rag_chain.invoke(q2))


if __name__ == "__main__":
    main()
