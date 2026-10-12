import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="RAG Document Bot", page_icon="📄", layout="centered")
st.title("📄 RAG Document Bot")
st.caption("Upload any document — ask anything from it | By Annu")

api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

@st.cache_resource
def get_llm():
    return ChatGroq(api_key=api_key, model="openai/gpt-oss-120b", temperature=0.3)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    print(f"DEBUG: {len(chunks)} chunks created")
    print(f"DEBUG: First chunk: {chunks[0].page_content[:100]}")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("DEBUG: Vectorstore built successfully")
    return vectorstore, len(chunks)

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on this document."

Context:
{context}

Question: {question}

Answer:
""")

llm = get_llm()
parser = StrOutputParser()
chain = prompt | llm | parser

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# Sidebar — upload document
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.doc_name:
            with st.spinner("Building knowledge base..."):
                text = uploaded_file.read().decode("utf-8")
                vs, num_chunks = build_vectorstore(text)
                st.session_state.vectorstore = vs
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []
            st.success(f"✅ Ready! {num_chunks} chunks indexed")

    if st.session_state.doc_name:
        st.info(f"📄 {st.session_state.doc_name}")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main area — chat
if not st.session_state.vectorstore:
    st.info("👈 Upload a .txt document in the sidebar to get started")
else:
    # Show chat history
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])

    # Question input
    question = st.chat_input("Ask anything about your document...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                chunks = st.session_state.vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([c.page_content for c in chunks])
                answer = chain.invoke({"context": context, "question": question})
            st.write(answer)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })