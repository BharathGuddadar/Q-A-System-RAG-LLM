import streamlit as st
import faiss
import os
from io import BytesIO
from docx import Document
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEndpoint
from api_key import huggingface_api_key
import time

# Set HuggingFace API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

st.set_page_config(
    page_title="AI Knowledge Assistant",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# ---------------- Simple Styling ----------------
def apply_simple_style():
    st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main app styling */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
        color: white;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        color: #2d3748;
    }
    
    .info-card h3 {
        color: #2d3748;
        margin-top: 0;
    }
    
    .info-card p {
        color: #4a5568;
        margin-bottom: 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #9ae6b4;
        margin: 1rem 0;
        color: #22543d;
    }
    
    .success-card h4 {
        color: #22543d;
        margin-top: 0;
    }
    
    .question-card {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #fc8181;
        color: #742a2a;
    }
    
    .question-card h3 {
        color: #742a2a;
        margin-top: 0;
    }
    
    .question-card p {
        color: #9c4221;
    }
    
    .answer-card {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #4299e1;
        color: #2a4365;
    }
    
    .answer-card h4 {
        color: #2a4365;
        margin-top: 0;
    }
    
    .answer-card p {
        color: #2c5282;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress indicator */
    .progress-step {
        display: flex;
        align-items: center;
        padding: 0.8rem;
        margin: 0.5rem 0;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        color: #4a5568;
    }
    
    .step-number {
        background: #e2e8f0;
        color: #4a5568;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .step-number.completed {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
    }
    
    .step-number.current {
        background: linear-gradient(135deg, #ed8936, #dd6b20);
        color: white;
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        background-color: white;
        color: #2d3748;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #f7fafc, #edf2f7);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #4a5568;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

def show_progress_steps(current_step):
    steps = [
        ("📁", "Upload Content"),
        ("⚙️", "Process Data"),
        ("❓", "Ask Questions"),
        ("💡", "Get Answers")
    ]
    
    for i, (icon, label) in enumerate(steps):
        if i < current_step:
            step_class = "completed"
        elif i == current_step:
            step_class = "current"
        else:
            step_class = ""
        
        st.markdown(f"""
        <div class="progress-step">
            <div class="step-number {step_class}">{icon}</div>
            <div>{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- Data Processing ----------------
def process_input(input_type, input_data, chunk_size=1000, chunk_overlap=100):
    progress_bar = st.progress(0)
    status = st.empty()
    
    try:
        status.info("📄 Reading your content...")
        progress_bar.progress(20)
        
        if input_type == "PDF" and isinstance(input_data, UploadedFile):
            pdf_reader = PdfReader(BytesIO(input_data.read()))
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
            texts = [text]

        elif input_type == "Link":
            documents = []
            valid_urls = [url for url in input_data if url.strip()]
            for i, url in enumerate(valid_urls):
                status.info(f"🌐 Loading website {i+1}/{len(valid_urls)}...")
                loader = WebBaseLoader(url)
                documents.extend(loader.load())
            texts = [doc.page_content for doc in documents]

        elif input_type == "Text" and isinstance(input_data, str):
            texts = [input_data]

        elif input_type == "DOCX" and isinstance(input_data, UploadedFile):
            doc = Document(BytesIO(input_data.read()))
            texts = ["\n".join([para.text for para in doc.paragraphs])]

        elif input_type == "TXT" and isinstance(input_data, UploadedFile):
            text = input_data.read().decode("utf-8")
            texts = [text]

        else:
            raise ValueError("Unsupported or invalid input type")

        status.info("✂️ Breaking text into chunks...")
        progress_bar.progress(40)
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []
        for t in texts:
            chunks.extend(text_splitter.split_text(t))

        if not chunks:
            raise ValueError("No text content found to process")

        status.info("🧠 Creating AI embeddings...")
        progress_bar.progress(70)
        
        # Create embeddings and FAISS index
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
        dimension = sample_embedding.shape[0]
        index = faiss.IndexFlatL2(dimension)

        vector_store = FAISS(
            embedding_function=hf_embeddings.embed_query,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        status.info("💾 Building knowledge base...")
        progress_bar.progress(90)
        
        vector_store.add_texts(chunks)
        
        progress_bar.progress(100)
        status.success("✅ Knowledge base ready!")
        time.sleep(1)
        
        progress_bar.empty()
        status.empty()
        
        return vector_store, len(chunks)

    except Exception as e:
        progress_bar.empty()
        status.empty()
        raise e

# ---------------- Question Answering ----------------
def answer_question(vectorstore, query, temperature=0.6):
    try:
        llm = HuggingFaceEndpoint(
            repo_id='HuggingFaceH4/zephyr-7b-beta',
            task='conversational',
            temperature=temperature,
            provider='featherless-ai'
        )
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        result = qa.invoke({"query": query})
        return result.get("result", "No answer found.")
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ---------------- Main App ----------------
def main():
    apply_simple_style()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Knowledge Assistant</h1>
        <p>Upload documents, paste text, or add links to create your personal AI assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        st.session_state.chunks_count = 0
        st.session_state.current_step = 0

    # Sidebar for settings
    with st.sidebar:
        st.markdown("### 🛠️ Settings")
        
        with st.expander("⚙️ Advanced Options", expanded=False):
            chunk_size = st.slider("Text Chunk Size", 500, 2000, 1000, 100,
                                  help="Larger chunks = more context, smaller chunks = more precise")
            chunk_overlap = st.slider("Chunk Overlap", 0, 200, 100, 25,
                                     help="Overlap between chunks for better context")
            temperature = st.slider("AI Creativity", 0.1, 1.0, 0.6, 0.1,
                                   help="Lower = more focused, Higher = more creative")
        
        # Progress tracker
        st.markdown("### 📋 Progress")
        show_progress_steps(st.session_state.current_step)
        
        # Statistics
        if st.session_state.vectorstore:
            st.markdown("### 📊 Stats")
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{st.session_state.chunks_count}</div>
                <div class="metric-label">Text Chunks</div>
            </div>
            """, unsafe_allow_html=True)

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Step 1: Input Selection
        st.markdown("""
        <div class="info-card">
            <h3>📁 Step 1: Choose Your Content</h3>
            <p>Select what type of content you want to upload and process</p>
        </div>
        """, unsafe_allow_html=True)
        
        input_type = st.selectbox("Content Type", ["PDF", "Text", "DOCX", "TXT", "Link"],
                                 help="Choose the type of content you want to process")

        # Input fields based on type
        if input_type == "Link":
            number_input = st.number_input("Number of Links", min_value=1, max_value=5, step=1, value=1)
            input_data = []
            for i in range(number_input):
                url = st.text_input(f"🔗 URL {i+1}", placeholder="https://example.com")
                input_data.append(url)
        elif input_type == "Text":
            input_data = st.text_area("📝 Enter your text here", height=150,
                                     placeholder="Paste your text content here...")
        else:
            input_data = st.file_uploader(f"📎 Upload your {input_type} file", 
                                         type=[input_type.lower()])

        # Process button
        if st.button("🚀 Process Content", use_container_width=True):
            if input_data and (input_type != "Link" or any(url.strip() for url in input_data)):
                try:
                    st.session_state.current_step = 1
                    vectorstore, chunks_count = process_input(input_type, input_data, chunk_size, chunk_overlap)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chunks_count = chunks_count
                    st.session_state.current_step = 2
                    
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>✅ Success!</h4>
                        <p>Your content has been processed into <strong>{chunks_count} searchable chunks</strong>. 
                        You can now ask questions about it!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.current_step = 0
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please provide valid content to process.")

    with col2:
        # Quick tips
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #2d3748; margin-top: 0;">💡 Quick Tips</h4>
            <ul style="margin: 0; padding-left: 1.2rem; color: #4a5568;">
                <li>PDFs work best with text-based content</li>
                <li>For websites, make sure URLs are accessible</li>
                <li>Larger chunk sizes capture more context</li>
                <li>Higher temperature makes AI more creative</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Step 2: Question Section (only if knowledge base is ready)
    if st.session_state.vectorstore:
        st.markdown("---")
        st.markdown("""
        <div class="question-card">
            <h3>❓ Step 2: Ask Your Questions</h3>
            <p>Your knowledge base is ready! Ask any question about your content.</p>
        </div>
        """, unsafe_allow_html=True)
        
        query = st.text_input("🤔 What would you like to know?", 
                             placeholder="Ask anything about your uploaded content...")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Get Answer", use_container_width=True):
                if query.strip():
                    st.session_state.current_step = 3
                    
                    with st.spinner("🤖 AI is thinking..."):
                        answer = answer_question(st.session_state.vectorstore, query, temperature)
                    
                    st.markdown(f"""
                    <div class="answer-card">
                        <h4>💡 Answer</h4>
                        <p><strong>Question:</strong> {query}</p>
                        <p><strong>Answer:</strong> {answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Please enter a question first.")

if __name__ == "__main__":
    main()