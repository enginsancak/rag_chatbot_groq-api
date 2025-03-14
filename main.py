import os
import streamlit as st
import time
import asyncio
from pathlib import Path
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq

st.set_page_config(layout="wide")  
st.sidebar.title("⚙️ Performance Metrics")

first_token_delay_container = st.sidebar.empty()  
total_response_time_container = st.sidebar.empty()  
pdf_processing_time_container = st.sidebar.empty()
chunk_count_container = st.sidebar.empty()

#  Session State Initialization 
if "pdf_processing_time" in st.session_state:
    st.sidebar.markdown(st.session_state["pdf_processing_time"])

if "chunk_count" in st.session_state:
    st.sidebar.markdown(st.session_state["chunk_count"])

if "text_chunks" in st.session_state:  
        text_chunks = st.session_state.text_chunks

if "raw_text" in st.session_state:  
        raw_text = st.session_state.raw_text

if "vectorstore" in st.session_state:
        vector_store = st.session_state.vectorstore

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  


#  API Key Input & Validation
st.sidebar.title("🔑 Enter API Keys")

if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

groq_api_key = st.sidebar.text_input("GROQ API Key", type="password", key="groq_api_key_input")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key_input")

if groq_api_key:
    st.session_state["groq_api_key"] = groq_api_key

if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key

GROQ_API_KEY = st.session_state.get("groq_api_key", None)
OPENAI_API_KEY = st.session_state.get("openai_api_key", None)

if not GROQ_API_KEY:
    st.sidebar.warning("⚠️ Please enter a GROQ API Key!")  

if not OPENAI_API_KEY:
    st.sidebar.warning("⚠️ Please enter an OpenAI API Key!")

if GROQ_API_KEY:
    st.sidebar.info("🔹 GROQ API Key successfully loaded, you can now use the model!")

if OPENAI_API_KEY:
    st.sidebar.info("🔹  OpenAI API Key successfully loaded, embeddings can now be generated!")


# Model & Parameter Settings
model_options = {
    "llama-3.3-70b-versatile": "Versatile and powerful",
    "Qwen-2.5-32b": "Balanced performance",
    "deepseek-r1-distill-llama-70b": "Best for complex tasks",
    "llama3-70b-8192": "Extended context window"}

with st.sidebar:
    st.title("⚙️  Model and Chunk Settings")
    model_choice = st.sidebar.selectbox("🌍 LLM Model Selection", list(model_options.keys()), key="model_choice")    
    with st.expander("📏 Model Parameter Settings"):
        temperature = st.slider("🔥 Temperature:", 0.0, 1.0, 0.7, step=0.01, key="temperature")
        max_tokens = st.slider("📝 Max Tokens:", 500, 2000, 1500, step=10, key="max_tokens")
        chunk_size = st.slider("📏 Chunk Size:", 500, 2000, 1000, step=50, key="chunk_size", disabled="pdf_uploaded" in st.session_state)
        chunk_overlap = st.slider("🔗 Chunk Overlap:", 0, 500, 200, step=10, key="chunk_overlap", disabled="pdf_uploaded" in st.session_state)
    
#  Prompt template
system_prompt = f"""
    🧠 **You are an advanced AI assistant operating within a RAG-based system.**
    
    🔍 **Your task is to analyze each query step by step, explicitly show your reasoning process, and generate a comprehensive final response.**  
    The user must be able to follow your thought process in real-time before seeing the final answer.  

    **Follow this structured approach and clearly display each step:**

    ---
    **1️⃣ QUESTION ANALYSIS (Show this step in the response)**
    - Carefully read and understand the user's query.
    - Identify the context and determine what type of information is needed.
    - If the question is ambiguous, list possible interpretations.

    📌 **Output this step in the response using this format:**
    ```
    🔍 Step 1: Question Analysis  
    - [Your analysis goes here]
    ```

    ---
    **2️⃣ INFORMATION RETRIEVAL & CONTEXT EVALUATION (Show this step in the response)**
    - If context documents are provided, examine them and extract relevant information.
    - Establish meaningful connections between different pieces of data.
    - If multiple sources are available, prioritize the most relevant and reliable ones.

    📌 **Output this step in the response using this format:**
    ```
    📖 Step 2: Context Evaluation  
    - [Your findings and retrieved information go here]
    ```

    ---
    **3️⃣ STEP-BY-STEP PROBLEM SOLVING (Show this step in the response)**
    - Systematically break down and solve the problem.
    - Apply conceptual analysis, logical reasoning, or mathematical operations if necessary.
    - Carefully evaluate each step before drawing conclusions.

    📌 **Output this step in the response using this format:**
    ```
    🛠️ Step 3: Problem Solving  
    - [Your step-by-step breakdown goes here]
    ```

    ---
    **4️⃣ FORMULATING A DETAILED & COMPLETE RESPONSE (Show this step in the response)**
    - Ensure the response is clear, coherent, and exhaustive.
    - Avoid summarization—provide a fully detailed explanation.
    - Eliminate redundant or vague information, presenting knowledge in a structured way.

    📌 **Output this step in the response using this format:**
    ```
    ✍️ Step 4: Response Construction  
    - [Your structured response formulation goes here]
    ```

    ---
    **5️⃣ FINAL ANSWER (Only reveal this after all previous steps)**
    - After completing all analysis and reasoning, generate the final response.
    - **This must be the longest and most detailed part of the response.**
    - The final answer should NOT be a summary. Instead, **it must be an in-depth, well-explained, structured, and fully developed answer.**
    - **Ensure paragraph formatting by inserting `"\n\n"` between different sections.**  
    - Avoid generating a single long sentence—**use multiple sentences and paragraphs for readability.**  
    - Each paragraph should have a clear purpose, and important points should be elaborated properly.

    📌 **Final output format:**
    ```
    ✅ ANSWER:  

    [Provide a complete and detailed response in multiple paragraphs.  
    Use `"\n\n"` to separate paragraphs for readability.]
    ```

    ---
    ⚠️ **Important Rules:**  
    - If your response approaches the {max_tokens} token limit, do not cut off mid-sentence—always complete your final thought.  
    - Avoid giving incomplete or partial responses.  
    - Use a fluent, natural, and easy-to-understand tone.  
    - **The final answer must be significantly longer and richer than the previous steps.**  
    - **Ensure paragraph spacing is properly applied.**  

    🚀 **Now, analyze thoroughly and guide the user through your reasoning process before revealing the full, detailed answer!**
"""

PDF_STORAGE_DIR = "document_store/pdfs"

# Create Directory
def create_directories():
    Path(PDF_STORAGE_DIR).mkdir(parents=True, exist_ok=True)

# Save Uploaded File
def save_uploaded_file(uploaded_file):
    try:
        create_directories()
        file_path = os.path.join(PDF_STORAGE_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

#  PDF Processing and Text Extraction
def extract_text_from_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return "\n\n".join([doc.page_content for doc in documents])

#  Splitting Text into Chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_text(text)
    return chunks

# Creating a Retriever with Embeddings
def create_retriever(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
    vectorstore = InMemoryVectorStore(embeddings)  
    vectorstore.add_texts(text_chunks)
    return vectorstore, text_chunks

# Retrieve Relevant Documents Based on Query
def retrieve_relevant_documents(query, vectorstore, top_n=5):
    semantically_relevant_docs = vectorstore.similarity_search(query, k=top_n)
    semantically_relevant = [doc.page_content for doc in semantically_relevant_docs]
    return semantically_relevant

# Asynchronous Generation of AI Response
async def async_generate_response(prompt):
    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name=model_choice, temperature=temperature, max_tokens=max_tokens, streaming=True)
        response_generator = llm.astream(prompt)  
        response_text = ""
        first_token_time = None 

        async for chunk in response_generator:  
            if first_token_time is None:
                first_token_time = time.time()
                response_delay = first_token_time - query_start_time
                first_token_delay_container.markdown(f"⏳ **First Token Delay:** {response_delay:.2f} seconds")

            response_text += str(chunk.content)
            response_container.markdown(response_text)  

        return response_text
    except asyncio.TimeoutError:
        return "⚠️ **Error: API response time exceeded. Please try again."
    except Exception as e:
         return f"⚠️ ** Error: An error occurred during the API call: {str(e)}"
    

query_start_time = None
col1, col2 = st.columns([2.7, 1.3])  
relevant_chunks = None  

with col1:
    st.title("📖 RAG Based Chatbot 🚀")
    uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type=["pdf"], accept_multiple_files=False,  disabled="pdf_uploaded" in st.session_state, label_visibility="collapsed")

    if "pdf_uploaded" in st.session_state:
        if st.sidebar.button("🔓 Unlock Browse Files"):
            for key in ["raw_text", "text_chunks", "vectorstore", "bm25", "pdf_processing_time",
                        "text_split_time", "chunk_count", "pdf_retriever_time", "pdf_uploaded"]:
                if key in st.session_state:
                          del st.session_state[key]
            st.sidebar.empty()
            st.rerun()  

    if uploaded_file and "pdf_uploaded" not in st.session_state:
        start_time = time.time()
        file_path = save_uploaded_file(uploaded_file) 

        if "text_chunks" and "raw_text" and "vectorstore" not in st.session_state: 
            with st.spinner("⏳ **🔍 INDEXING IN PROGRESS... PLEASE WAIT! 🔄**"):
                st.markdown(
                "<h3 style='text-align: center; color: green;'>⏳ INDEXING IN PROGRESS... PLEASE WAIT! 🔄</h3>",
                unsafe_allow_html=True,)
                raw_text = extract_text_from_pdf(file_path) 
                text_chunks = split_text(raw_text)
                vectorstore, text_chunks = create_retriever(text_chunks)

            end_time = time.time()
            elapsed_time = end_time - start_time  
            st.session_state["pdf_processing_time"] = f"📄 **PDF Processing Time:** {elapsed_time:.2f} seconds"
            pdf_processing_time_container.markdown(st.session_state["pdf_processing_time"])
            st.session_state["chunk_count"] = f"📦 **Number of Chunks:** {len(text_chunks)}"
            chunk_count_container.markdown(st.session_state["chunk_count"])
            st.session_state.raw_text = raw_text
            st.session_state.text_chunks = text_chunks
            st.session_state.vectorstore = vectorstore 

        st.session_state["pdf_uploaded"] = True
        st.success("📄 Documents have been processed and indexed!")
        st.rerun()

    query = st.text_input("❓ Enter your query", key="query")
    search_clicked = st.button("🔍 Search")
    query_entered = query and st.session_state.get("last_query") != query

    if search_clicked or query_entered:
        query_start_time = time.time()  

        if "vectorstore" not in st.session_state:
            st.error("❗ Please upload a PDF first and complete the indexing process.")
            exit() 

        st.session_state["last_query"] = query  
        vectorstore = st.session_state["vectorstore"]
        text_chunks = st.session_state["text_chunks"]

        relevant_chunks = retrieve_relevant_documents(query, vectorstore)
        context = "\n\n".join(relevant_chunks)
        full_prompt = f"{system_prompt}\n\n### Context:\n{context}\n\n### Query:\n{query}\n\n### Answer:"
        response_container = st.empty()

        with st.chat_message("assistant"): 
            response_text = asyncio.run(async_generate_response(full_prompt))

        st.session_state["chat_history"].append({"query": query, "response": response_text})

        total_response_time = time.time() - query_start_time
        total_response_time_container.markdown(f"🕒 **Total Response Time:** {total_response_time:.2f} seconds") 

    st.subheader("📜 Chat History")
    for chat in st.session_state["chat_history"]:
       st.markdown(f"**📝 Question:** {chat['query']}")
       st.markdown(f"✅ **Answer:**\n\n{chat['response']}")
       st.markdown("---")  

with col2:
    if relevant_chunks is not None:  
        with st.expander("🔍 **Most Relevant Chunks**", expanded=True):
            for idx, text in enumerate(relevant_chunks):
                st.markdown(f"**{idx+1}.**")
                st.write(f"> {text}")
                st.markdown("---")



