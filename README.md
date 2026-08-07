# 📚 Agentic AI RAG using LangChain, Gemini & FAISS

An intelligent **Retrieval-Augmented Generation (RAG)** application built with **Python, Streamlit, LangChain, Google Gemini, and FAISS**. Users can upload PDF documents, ask questions in natural language, and receive accurate, context-aware answers based on the uploaded content. If the required information is unavailable, the application can perform a web search, verify the generated response, and automatically regenerate the answer when necessary, providing a more reliable and transparent AI experience.

---

## ✨ Features

- 📄 Upload and process multiple PDF documents.
- 🔍 Automatic document loading and intelligent text chunking.
- 🧠 Generate vector embeddings using Google Gemini Embeddings.
- ⚡ Store and retrieve document embeddings with FAISS.
- 💬 Ask questions in natural language.
- 📖 Generate context-aware answers using Google Gemini.
- 🌐 Optional web search when information is unavailable in the uploaded documents.
- ✅ AI-powered answer verification.
- 🔄 Self-correction mechanism for unsupported answers.
- 📑 Display document and web sources for transparency.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini
- FAISS
- DuckDuckGo Search
- PyPDFLoader
- RecursiveCharacterTextSplitter

---

## 🚀 Workflow

1. Upload one or more PDF documents.
2. Extract and split document text into chunks.
3. Generate embeddings and create a FAISS vector database.
4. Retrieve the most relevant document chunks based on the user's query.
5. Generate an answer using Gemini.
6. If the answer is not found, optionally perform a web search.
7. Verify whether the generated answer is supported by the available context.
8. Automatically trigger self-correction if verification fails.
9. Display the final answer along with its sources.

---

## 🎯 Project Highlights

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- FAISS Vector Database
- Agentic AI Workflow
- Answer Verification
- Self-Correction Pipeline
- Web Search Integration
- Explainable AI with Source Attribution

---

## 📌 Future Improvements

- 🧠 **Conversation Memory**  
  Implement conversational memory so users can ask follow-up questions without repeating the complete context, making interactions more natural and continuous.

- 📄 **Optimized Document Processing**  
  Detect duplicate PDFs and avoid unnecessary re-embedding. Incremental indexing can also be introduced so that only newly uploaded or modified documents are processed, reducing processing time and API usage.

- 🖥️ **Advanced User Interface**  
  Improve the interface with features such as document management, chat history, download options, clearer source citations, and real-time processing status.

- 🛡️ **Enhanced Logging & Error Handling**  
  Implement better logging and exception handling to simplify debugging and prevent application crashes caused by API, network, or document-processing errors.

Overall, these improvements would make the system more **scalable, reliable, accurate, and suitable for real-world GenAI applications**.

---

## 🎥 Demo

Check out the project demo below to see the Agentic AI RAG application in action.

[▶️ Watch Demo Video](./demo.mp4)

---

## 📄 License

This project is intended for educational and learning purposes.
