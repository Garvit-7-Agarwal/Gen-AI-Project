import os
from pathlib import Path
import uuid

import streamlit as st
st.set_page_config(
    page_title="Agentic AI RAG",
    page_icon="📚"
)

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ==========================
# LangChain Gemini Models
# ==========================
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.tools import DuckDuckGoSearchResults

# ==========================
# Load Environment Variables
# ==========================
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite',temperature=0.1)

embeddings_model = GoogleGenerativeAIEmbeddings( model="models/gemini-embedding-001" )

search = DuckDuckGoSearchResults(output_format="list")

answer = ""
# ==========================
# Create Data Folder and vector store directory 
# ==========================

# Create a unique ID for this Streamlit session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id

DATA_DIR = Path("data") / SESSION_ID
VECTOR_DB = Path("vector_store") / SESSION_ID

DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB.mkdir(parents=True, exist_ok=True)


# ==========================
# Streamlit UI
# ==========================

if "web_search_approved" not in st.session_state: 
    st.session_state.web_search_approved = False 

st.title("📚 Agentic AI RAG")
st.subheader("Step 1 : Document Ingestion")

uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True,
)


if uploaded_files and st.button("Process Documents"):

    # ==========================
    # Save Uploaded PDFs
    # ==========================

    saved_files = []

    for uploaded_file in uploaded_files:

        file_path = DATA_DIR / uploaded_file.name
        

        with open(file_path, "wb") as f:  
            f.write(uploaded_file.getbuffer()) 

        saved_files.append(uploaded_file.name) 

    st.success(f"{len(saved_files)} PDF(s) uploaded successfully!")

    st.write("### Saved Files")

    for file in saved_files:
        st.write(f"✅ {file}")

    # -------------------------
    # Chunk Documents
    # -------------------------

    documents = []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    for pdf in pdf_files:

        loader = PyPDFLoader(str(pdf))
        documents.extend(loader.load())

    st.success(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    st.success(f"Created {len(chunks)} chunks.")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings_model,
    ) 

    vector_store.save_local(str(VECTOR_DB)) 
    st.success("FAISS Vector Store Created!")

# ==========================
# Display Existing PDFs
# ==========================
existing_pdfs = sorted(DATA_DIR.glob("*.pdf"))

if existing_pdfs:
    st.divider()
    st.write("### PDFs Currently Available in This Session")

    for pdf in existing_pdfs:
        st.write(f"📄 {pdf.name}")

# ==========================
# Load FAISS Vector Store
# ==========================

if (VECTOR_DB / "index.faiss").exists():

    vector_store = FAISS.load_local(
        str(VECTOR_DB),
        embeddings_model,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    ) 

st.divider()

st.subheader("Ask Questions")

query = st.text_input("Enter your question")

context = ""
retrieved_docs=""

if query.strip():

    if not (VECTOR_DB / "index.faiss").exists():
        st.warning("⚠️ Please upload and process PDF documents first.")
        st.stop()


    retrieved_docs = retriever.invoke(query)
    for doc in retrieved_docs:
        context += doc.page_content + "\n\n"

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant.

    Answer ONLY using the provided context.

    If the context does not contain enough information,
    reply exactly with:

    NOT_FOUND

    Context:
    {context}

    Question:
    {question}
    """)

    formatted_prompt = prompt.format(
        context=context,
        question=query
    )

    response = model.invoke(formatted_prompt)
    answer = response.text().strip()

    if answer == "NOT_FOUND":

        st.warning(
            "I couldn't find sufficient information in the uploaded documents."
        )

        if st.button("Search the Web"):
            st.session_state.web_search_approved = True

if st.session_state.web_search_approved:

    web_results = search.invoke(query)

    web_context = ""

    for result in web_results:
        web_context += (
            f"Title: {result['title']}\n"
            f"Content: {result['snippet']}\n"
            f"URL: {result['link']}\n\n"
        )

    web_prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant.

    Answer the user's question ONLY using the web search results provided below.

    If the search results do not contain enough information, clearly say so.

    Do NOT invent facts.
    Do NOT invent sources.
    Do NOT include URLs or a Sources section in your answer because they will be displayed separately.

    Web Results:
    {context}

    Question:
    {question}
    """)

    formatted_prompt = web_prompt.format(
        context=web_context,
        question=query
    )
    st.write("Web Search Completed. ✅")
    response = model.invoke(formatted_prompt)

if st.session_state.web_search_approved:
    final_answer = response.text().strip()
    final_context = web_context
else:
    final_answer = answer
    final_context = context

if query.strip():
    verification_prompt = ChatPromptTemplate.from_template("""
    You are verifying an AI-generated answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    {answer}

    Determine whether every important claim in the answer is supported by the context.

    Reply with ONLY one word:

    SUPPORTED

    or

    UNSUPPORTED
    """)

    verification_response = model.invoke(
        verification_prompt.format(
            context=final_context,
            question=query,
            answer=final_answer,
        )
    )

    verification = verification_response.text().strip()

    if verification == "UNSUPPORTED":

        if st.session_state.web_search_approved:
            st.info("🔄 Self-correction triggered...")

            web_results = search.invoke(query)

            # rebuild web_context

            response = model.invoke(
                web_prompt.format(
                    context=web_context,
                    question=query,
                )
            )

            final_answer = response.text().strip()
            st.success("Answer regenerated using additional Web Search.")

        else:
            st.info("🔄 Self-correction triggered...")

            # Retrieveing more chunks this time
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 8}
            )

            retrieved_docs = retriever.invoke(query)

            final_context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

            formatted_prompt = prompt.format(
                context=final_context,
                question=query
            )

            response = model.invoke(formatted_prompt)

            final_answer = response.text().strip()

            st.success("Answer regenerated using additional retrieved context.")

# Final answer
if st.button("Answer"):
    if query.strip():
        if not (VECTOR_DB / "index.faiss").exists():
            st.warning("⚠️ Please upload and process PDF documents first.")
            st.stop()

        
        st.subheader("Answer")
        st.write(final_answer)
        
        # Sources

        if st.session_state.web_search_approved:

            st.subheader("Sources")
            
            for result in web_results:
                st.write(f"🔗 {result['link']}")
        else:
            st.subheader("Sources")
            
            for doc in retrieved_docs:
            
                source = Path(doc.metadata["source"]).name
                page = doc.metadata["page"] + 1
            
                st.write(f"📄 {source} | Page {page}")
        st.session_state.web_search_approved = False
    else:
        st.warning("⚠️ Write question first.")
        st.stop()