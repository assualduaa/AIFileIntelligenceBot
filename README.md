# RAG-Based AI Intelligence Bot 🤖

An AI-powered file-based intelligent assistant that allows users to upload documents and ask contextual questions based on the uploaded content. The system uses embeddings and LLM-based retrieval to deliver accurate, context-aware responses from user-provided files.

---

## 🚀 Overview

This project is a Retrieval-Augmented Generation (RAG) based AI assistant. Users can upload files (PDF, TXT, DOCX, etc.), and the bot processes the content to enable natural language question answering directly from the uploaded data.

It eliminates manual searching by allowing users to interact with their documents conversationally.

---

## ✨ Key Features

- 📂 Upload and process multiple file formats (PDF, TXT, DOCX, etc.)
- 🧠 Context-aware question answering from uploaded documents
- 🔍 Semantic search using embeddings (FAISS-based retrieval)
- 💬 Conversational AI interface
- ⚡ Fast and lightweight local or cloud deployment
- 🔐 Secure file handling and processing
- 🧩 Modular architecture for easy extension

---

## 🏗️ System Architecture

User Upload → File Parser → Text Chunking → Embedding Model → Vector DB (FAISS)
                                                           ↓
                                              User Query → Embedding Search
                                                           ↓
                                               Context + LLM (Mistral/Ollama/OpenAI)
                                                           ↓
                                                   Final Answer Output

---

## 🛠️ Tech Stack

- Backend: Python (FastAPI / Flask)
- AI/LLM: Mistral / Ollama / OpenAI (configurable)
- Embeddings: SentenceTransformers / OpenAI Embeddings
- Vector Database: FAISS
- File Processing: PyPDF2, python-docx, unstructured.io
- Orchestration: Custom pipeline / optional n8n integration
- Frontend (optional): React / Streamlit

---

## 📦 Installation

### 1. Clone the Repository

git clone https://github.com/your-username/ai-intelligence-bot.git
cd ai-intelligence-bot

---

### 2. Create Virtual Environment

python -m venv venv

Mac/Linux:
source venv/bin/activate

Windows:
venv\Scripts\activate

---

### 3. Install Dependencies

pip install -r requirements.txt
python -m pip install easyocr

---

## ⚙️ Environment Variables

Create a `.env` file:

LLM_MODEL=mistral
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_PATH=./vectorstore
UPLOAD_FOLDER=./uploads

---

## ▶️ Run the Application

python main.py

Server will run at:
http://localhost:8000

---

## 📂 Project Structure

ai-intelligence-bot/

├── backend/
│   ├── main.py
│   ├── file_loader.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm_chain.py
│
├── uploads/
├── vectorstore/
├── requirements.txt
├── .env
└── README.md

---

## 💡 How It Works

1. User uploads a document  
2. System extracts and chunks text  
3. Text is converted into embeddings  
4. Stored in FAISS vector database  
5. User asks a question  
6. Relevant chunks are retrieved  
7. LLM generates final contextual answer  

---

## 📌 Example Usage

### Upload File

POST /upload

---

### Ask Question

POST /ask
{
  "question": "What is the summary of the document?"
}

---

## 🧠 Future Improvements

- Multi-file knowledge merging  
- Chat history memory  
- Role-based document access  
- UI dashboard for analytics  
- n8n automation workflow integration  
- Multi-agent AI reasoning layer  

---

## 👨‍💻 Author

Built by Asna Sherin  
AI Engineer | Data Analyst | Automation Specialist  

---

## 📄 License

This project is open-source and available under the MIT License.
