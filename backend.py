import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest_pdf(pdf_path: str, db_path: str = "faiss_index") -> FAISS:
    """
    Load a PDF, chunk its content, create embeddings, and store in FAISS DB.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embedding_model)
    vectordb.save_local(db_path)
    return vectordb


def load_vectordb(db_path: str = "faiss_index") -> FAISS:
    """
    Reload FAISS index from local storage.
    """
    return FAISS.load_local(
        db_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )


def build_qa_chain(vectordb: FAISS, groq_api_key: str):
    """
    Build a RetrievalQA chain using FAISS retriever + Groq LLM.
    The Groq API key is passed from the Streamlit app.
    """
    llm = ChatGroq(
        model="llama3-70b-8192",  # you can swap to "llama-3-70b-chat" if enabled on your Groq account
        api_key=groq_api_key,
        temperature=0
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # simple retrieval pipeline
        return_source_documents=True
    )
    return chain
