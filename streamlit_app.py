import streamlit as st
import tempfile
import os
import base64
import uuid
import gc
import time
import numpy as np
from pdf2image import convert_from_path
import logging


# Import your existing RAG components - fixed imports
from src.embedder import EmbedData, ColPaliEmbedder
from src.vector_store import QdrantVectorStore
from src.rag_pipeline import RAG, Retriever
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="Multimodal RAG with Janus",
    page_icon="🔍",
    layout="wide"
)

# Configuration
collection_name = "multimodal_rag_with_deepseek-new"

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    st.session_state.context = None

session_id = st.session_state.id

def reset_chat():
    """Reset chat history and clear memory"""
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    """Display PDF preview in sidebar"""
    st.markdown("### PDF Preview")
    # Reset file pointer to beginning
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}" 
            width="400" height="100%" type="application/pdf"
            style="height:100vh; width:100%">
    </iframe>
    """
    
    st.markdown(pdf_display, unsafe_allow_html=True)

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Save uploaded file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                
                # Check if file already processed
                if file_key not in st.session_state.get('file_cache', {}):
                    with st.spinner("🔄 Processing your document..."):
                        # Convert PDF to images
                        images = convert_from_path(file_path)
                        
                        # Create images directory
                        os.makedirs('./images', exist_ok=True)
                        
                        # Save images with consistent naming
                        image_paths = []
                        for i, image in enumerate(images):
                            image_path = f'./images/page{i}.jpg'
                            image.save(image_path, 'JPEG')
                            image_paths.append(image_path)
                        
                        # Initialize embedding properly
                        embedder = ColPaliEmbedder(
                            use_hpc=True,  # Enable HPC
                            hpc_kwargs={'k_centroids': 256, 'prune_ratio': 0.6}
                        )
                        
                        # Build codebook if using HPC
                        if embedder.use_hpc:
                            with st.spinner("Building HPC codebook... (this might take a moment)"):
                                # Use a smaller sample of images to build the codebook
                                sample_images = images[:min(200, len(images))]
                                
                                # Use the new, dedicated method to get raw patch embeddings
                                raw_sample_embeddings = embedder.get_raw_patch_embeddings_for_codebook(sample_images)
                                
                                # Fit the codebook with the raw embeddings
                                if raw_sample_embeddings.size > 0:
                                    embedder.build_hpc_codebook(raw_sample_embeddings)
                                else:
                                    st.warning("Could not generate sample embeddings for HPC codebook. HPC will be disabled.")
                                    embedder.use_hpc = False # Fallback
                        
                        # Then proceed with normal embedding and ingestion
                        with st.spinner("Embedding all pages..."):
                            embeddings = embedder.embed_images(images)
                        
                        vector_store = QdrantVectorStore(
                            collection_name=collection_name,
                            vector_dim=128,  # Always 128 for Janus compatibility
                            batch_size=4,
                            use_hpc=embedder.use_hpc and getattr(embedder, 'codebook_built', False)
                        )

                        vector_store.create_collection()
                        vector_store.ingest_data(embeddings, images, image_paths=image_paths)

                        logger.info(f"Vector store created with {len(embeddings)} embeddings")

                        # Verify collection info
                        collection_info = vector_store.collection_info()
                        if collection_info:
                            logger.info(f"Collection vectors count: {collection_info.vectors_count}")
                        
                        # Set up retriever and RAG correctly
                        retriever = Retriever(vector_store=vector_store, embedder=embedder)
                        query_engine = RAG(retriever=retriever)
                        
                        # Cache the query engine
                        st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]
                
                st.success("✅ Ready to Chat!")
                display_pdf(uploaded_file)
                
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.stop()

# Main chat interface
col1, col2 = st.columns([6, 1])

with col1:
    st.header("# 🤖 Multimodal RAG powered by Janus")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history if not exists
if "messages" not in st.session_state:
    reset_chat()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your document..."):
    # Check if document is uploaded
    if not st.session_state.file_cache:
        st.warning("⚠️ Please upload a PDF document first!")
        st.stop()
    
    # Get the query engine (assumes single document for now)
    query_engine = next(iter(st.session_state.file_cache.values()))
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Get response from RAG - handle both streaming and non-streaming
            response = query_engine.query(prompt)
            
            # Check if it's a generator (streaming) or string
            if hasattr(response, '__iter__') and not isinstance(response, str):
                # Stream the response
                for chunk in response:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
            else:
                # Non-streaming response
                full_response = str(response)
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            message_placeholder.markdown(error_msg)
            full_response = error_msg
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})