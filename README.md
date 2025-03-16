# rag_chatbot_groq-api
RAG-based chatbot using OpenAI &amp; Groq API, built with LangChain &amp; Streamlit. Supports PDF processing, vector search, real-time query handling, step-by-step reasoning and streaming responses.  GROQ API is executed asynchronously for faster response times.
## Demo Video
[Watch the video](https://drive.google.com/file/d/1lx8npUI1FD-rJvMY75pel7lR3oZ_d7cd/view)

## 🌟 Features  

- **PDF Upload & Processing**: Upload PDF files and extract meaningful text for AI-driven analysis.  
- **RAG-Based Retrieval**: Uses Retrieval-Augmented Generation (RAG) to provide accurate and contextual responses.  
- **Vector-Based Semantic Search**: Stores and retrieves document chunks using an In-Memory VectorStore.  
- **Streaming AI Responses**: Get real-time, token-by-token responses with GROQ models.
- **Asynchronous API Calls**: GROQ API is executed asynchronously for faster response times and improved efficiency.
- **Step-by-Step Query Breakdown**: AI explains its reasoning in multiple structured steps before answering.  
- **Performance Monitoring**: Displays response time, chunk count, and PDF processing duration.  
- **Advanced Model Settings**: Customize temperature, max tokens, chunk size, and overlap via UI controls.  
- **Persistent Chat History**: Maintains previous queries and AI responses for seamless conversations.  
- **Clean & Interactive UI**: Built with Streamlit for a user-friendly and responsive interface.

## 🚀 Quick Start
### Prerequisites
- Python 3.8 or higher
- Groq API key
- OpenAI API key (for embeddings)
### Installation

**1️. Clone the Repository:**

      git clone https://github.com/enginsancak/rag_chatbot_groq-api

      cd rag_chatbot_groq-api

**2. Create and activate a virtual environment:**
   
      python -m venv venv

      source venv/bin/activate  # On Windows: .\venv\Scripts\activate

**3. Install dependencies:**

      pip install -r requirements.txt

### Running the Application

**1. Start the Streamlit app:**

      streamlit run main.py

**2. Open your browser and navigate to http://localhost:8501**

## 📁 Project Structure

     deepseek-rag-chatbot/
     ├── main.py                             # Main application file
     ├── requirements.txt                    # Project dependencies
     ├── README.md                           # Project documentation
     └── document_store/                     # Document storage directory
         └── pdfs/                           # PDF storage directory

## 💡 Usage Guide

**1. Setup**
- Enter your Groq API key
- Enter your OpenAI key
- Select a model
- Configure advanced settings

**2. Document Upload**
- Upload a PDF document
- Documents are stored in document_store/pdfs/
- System processes and indexes the document

**3.Chatting**
- Ask questions about your document
- View the AI's thinking process
- View the Answer

**4.Advanced Settings**
- Temperature: Control response randomness (0.0-1.0)
- Chunk Size: Adjust text processing (500-2000)
- Chunk Overlap: Adjust context overlap (0-500)

## 🔧 Available Models
- **llama-3.3-70b-versatile:** Versatile and powerful
- **Qwen-2.5-32b:** Balanced performance
- **deepseek-r1-distill-llama-70b:** Best for complex tasks
- **llama3-70b-8192:** Extended context window

## 🔍 Technical Details
**Components**
- **Frontend:** Streamlit
- **RAG Implementation:** LangChain
- **Embeddings:** OpenAI Text Embeddings
- **LLM Provider:** Groq
- **PDF Processing:** PDFPlumber
- **Text Splitting:** RecursiveCharacterTextSplitter
- **Vector Store:** InMemoryVectorStore

**Process Flow**
1. Document Upload → PDF Processing → Text Chunking
2. Chunk Embedding → Vector Storage
3. Query Processing → Context Retrieval
4. Asynchronous LLM Processing → Streaming Response



