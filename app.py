import os
import streamlit as st
from backend import ingest_pdf, load_vectordb, build_qa_chain


st.set_page_config(page_title="LegalEagle", layout="wide")
st.title("🦅 LegalEagle — AI Contract Assistant")

st.markdown(
    """
    **LegalEagle** is an AI-powered assistant for legal documents... Upload contracts, agreements, or policies and ask questions in plain English...The app retrieves the most relevant clauses and explains them clearly with citations.     
    """
)


os.makedirs("contracts", exist_ok=True)


groq_api_key = st.text_input("🔑 Enter your Groq API Key", type="password")
if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()


uploaded = st.file_uploader("📂 Upload a contract / legal document", type=["pdf"])
if uploaded:
    file_path = f"contracts/{uploaded.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"✅ {uploaded.name} uploaded & indexed.")
    vectordb = ingest_pdf(file_path)
else:
    try:
        vectordb = load_vectordb()
    except:
        vectordb = None
        st.warning("⚠️ No documents indexed yet. Please upload a PDF.")


if vectordb:
    qa = build_qa_chain(vectordb, groq_api_key)  # pass API key
    query = st.text_input("💬 Ask a question about your contract")
    if query:
        result = qa(query)

        st.subheader("📖 Answer")
        st.write(result["result"])

        st.subheader("📑 Sources")
        for doc in result["source_documents"]:
            st.markdown(
                f"📄 **Page {doc.metadata.get('page', '?')}**\n\n"
                f"{doc.page_content[:500]}..."
            )
