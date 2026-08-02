import os
from pathlib import Path

import streamlit as st
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
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

VECTOR_DB = Path("vector_store")
VECTOR_DB.mkdir(exist_ok=True)


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Agentic AI RAG", page_icon="📚")

if "web_search_approved" not in st.session_state: # st.session_state is like a dictionary that stores variables for a user's current session.
    st.session_state.web_search_approved = False # Need to initialse it first otherwise we will get key error 

st.title("📚 Agentic AI RAG")
st.subheader("Step 1 : Document Ingestion")

uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True,
)


if uploaded_files and st.button("Process Documents"): # i add this button because it reduces the reuse of api key like when we run this code then every time the embedding will be generated but if it was generated earlier then it is not neccesary to generate again so this button will help 

    # ==========================
    # Save Uploaded PDFs
    # ==========================

    saved_files = []

    for uploaded_file in uploaded_files:

        file_path = DATA_DIR / uploaded_file.name
        # It does not create the file. It only creates the path where the file should be stored.
        # Ex: data/Machine_Learning.pdf

        with open(file_path, "wb") as f:  # Here it create an empty file like an empty file is created in data folder and wb mean write in binary mode
            f.write(uploaded_file.getbuffer()) # This returns the raw binary content of the uploaded file.

        saved_files.append(uploaded_file.name) # saved files only contains the name of all the files 

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
    ) # Vector store both : the original text,its corresponding vector

    vector_store.save_local(str(VECTOR_DB)) # str(VECTOR_DB) ensures you're passing a plain string path, which is compatible with libraries that expect strings. It's a common practice to avoid compatibility issues.
    # Now vector store will contain two files 
    # 1) index.faiss : Vector embeddings in a binary FAISS index
    # 2) index.pkl : Original document chunks, metadata, and mapping information

    st.success("FAISS Vector Store Created!")

# ==========================
# Display Existing PDFs
# ==========================
existing_pdfs = sorted(DATA_DIR.glob("*.pdf"))

# DATA_DIR.glob("*.pdf") searches the DATA_DIR (i.e., the "data/" folder)
# and returns all files whose extension is ".pdf".

# Example:
# existing_pdfs = [
#     Path("data/AI.pdf"),
#     Path("data/ML.pdf"),
#     Path("data/Resume.pdf")
# ]

# The glob() method returns an iterator of Path objects.
# When wrapped with sorted(), it becomes a list of Path objects
# sorted alphabetically by their file names.

if existing_pdfs:
    st.divider()
    st.write("### PDFs Currently Available in `/data`")

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
    ) # means retrieve the 4 most relevant chunks.

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
    # retrieved_docs is a list of Document objects.
    for doc in retrieved_docs:
        context += doc.page_content + "\n\n"
    # context = "\n\n".join(doc.page_content for doc in retrieved_docs)

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

# Step 6 

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