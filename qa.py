import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_BASE"] = "http://202.31.200.88:8111/v1"
os.environ["OPENAI_API_KEY"] = "not_used"
def q_agent(d):
    loader = WebBaseLoader(
        web_paths=(d,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={"class": ["newsct_article _article_body",
                                "media_end_head_title"]},
            )
        ),
    )
    docs = loader.load()
    print(f"문서의 수: {len(docs)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)


    page_idx = 10
    if len(docs) > page_idx:
        print(f"\n[페이지내용]\n{docs[page_idx].page_content[:500]}")
        print(f"\n[metadata]\n{docs[page_idx].metadata}\n")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
    splits = text_splitter.split_documents(docs)
    print(f"분할된 문서의 수: {len(splits)}")
    # 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
    # 벡터스토어를 생성합니다.
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )

    vectorstore = FAISS.from_documents(splits, embedding_model)

    # 단계 4: 검색(Search)
    # 뉴스에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()

    def format_docs(ds):
        # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
        return "\n\n".join(doc.page_content for doc in ds)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 질문-답변 도우미다. 반드시 제공된 context 안에서만 답해라. 모르면 모른다고 말해라."),
        ("human", "질문: {question}\n\n문맥: {context}\n\n답변:")
    ])
    # 단계 7: 체인 생성(Create Chain)
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    def format_docs(ds):
        return "\n\n".join(d.page_content for d in ds)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("rag_chain 생성 완료")
    return rag_chain

d = input("문서 url을 입력하세요 (종료하려면 'exit' 입력): ")
rag_chain = q_agent(d)
q = ""
while(q != "exit"):
    q = input("질문을 입력하세요(exit 입력 시 종료): ")
    answer = rag_chain.invoke(q)
    print("응답 본문:\n", answer)