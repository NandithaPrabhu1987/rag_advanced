"""
Streamlit UI for Real-Time RAG Agent
===================================

This module provides a user-friendly web interface for the Real-Time RAG agent.
"""

import os
import time
import json
import subprocess
import threading
import streamlit as st
from typing import Dict, List, Any

from real_time_rag_agent import RealTimeRAGAgent

# Start API server in background
def start_api_server():
    try:
        # Try to import Flask to make sure it's installed
        import flask
        import flask_cors
        
        # Start server in a separate process
        process = subprocess.Popen(
            ["python", "mock_api_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the server to start
        time.sleep(2)
        return process
    except ImportError:
        st.error("Flask is not installed. Please install flask and flask-cors.")
        return None
    except Exception as e:
        st.error(f"Failed to start API server: {e}")
        return None

# Initialize session state
def init_session_state():
    if "agent" not in st.session_state:
        st.session_state.agent = RealTimeRAGAgent()
        
    if "api_server" not in st.session_state:
        st.session_state.api_server = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "api_connected" not in st.session_state:
        st.session_state.api_connected = False
        
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = []

# Format the source documents for display
def format_sources(sources):
    source_text = ""
    
    for source in sources:
        freshness = ""
        if source["source_type"] == "API Data" and source["data_freshness_minutes"] is not None:
            freshness = f"(Data from {source['data_freshness_minutes']:.1f} minutes ago)"
            
        source_text += f"**Source {source['id']}** ({source['source_type']}) {freshness}\n\n"
        source_text += f"{source['content']}\n\n"
        
    return source_text

# Add a document to the agent
def add_document(file, doc_id=None):
    # Save uploaded file temporarily
    with open(f"temp_{file.name}", "wb") as f:
        f.write(file.getbuffer())
        
    # Add to agent
    if not doc_id:
        doc_id = os.path.basename(file.name)
        
    result = st.session_state.agent.add_document(f"temp_{file.name}", doc_id)
    
    # Clean up temp file
    os.remove(f"temp_{file.name}")
    
    if result["status"] == "success":
        if doc_id not in st.session_state.documents_loaded:
            st.session_state.documents_loaded.append(doc_id)
        return True, result
    else:
        return False, result

# Connect to API
def connect_api():
    if not st.session_state.api_server:
        st.session_state.api_server = start_api_server()
        
    if not st.session_state.api_server:
        return False, "Failed to start API server"
        
    # Configure API connector
    result = st.session_state.agent.add_api_connector(
        "mock_api",
        "http://localhost:8000",
        {
            "stocks": "/api/stocks",
            "news": "/api/news",
            "weather": "/api/weather"
        },
        refresh_interval=60  # Refresh every minute
    )
    
    if result["status"] == "success":
        # Refresh initial data
        for endpoint in ["stocks", "news", "weather"]:
            st.session_state.agent.refresh_api_data("mock_api", endpoint)
            
        st.session_state.api_connected = True
        return True, result
    else:
        return False, result

# Process user query
def process_query(question, refresh_api=True, active_source=None):
    # Set active source if specified
    if active_source:
        st.session_state.agent.set_active_document(active_source)
        
    # Process the query
    with st.spinner("Processing your question..."):
        result = st.session_state.agent.query_with_explanation(
            question,
            refresh_api_data=refresh_api and st.session_state.api_connected
        )
        
    return result

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Real-Time RAG Agent",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize session
    init_session_state()
    
    # Header
    st.title("📚 Real-Time RAG with API Integration")
    st.markdown("Combine static documents with real-time API data for powerful question answering")
    
    # Sidebar for settings and controls
    with st.sidebar:
        st.header("Settings")
        
        # Document upload
        st.subheader("Document Management")
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf"])
        doc_id = st.text_input("Document ID (optional)")
        
        if uploaded_file is not None:
            if st.button("Add Document"):
                success, result = add_document(uploaded_file, doc_id or None)
                if success:
                    st.success(f"Document added: {result['document_id']}")
                else:
                    st.error(f"Error: {result['message']}")
        
        # API connection            
        st.subheader("API Connection")
        if not st.session_state.api_connected:
            if st.button("Connect to Mock API"):
                success, result = connect_api()
                if success:
                    st.success("Connected to Mock API")
                else:
                    st.error(f"Connection failed: {result['message']}")
        else:
            st.success("✅ Connected to Mock API")
            
            if st.button("Refresh API Data"):
                with st.spinner("Refreshing data..."):
                    for endpoint in ["stocks", "news", "weather"]:
                        st.session_state.agent.refresh_api_data("mock_api", endpoint, force=True)
                st.success("API data refreshed")
        
        # Knowledge source selector
        if st.session_state.documents_loaded or st.session_state.api_connected:
            st.subheader("Knowledge Source")
            
            # Get available sources
            sources = st.session_state.agent.get_available_knowledge_sources()
            all_sources = []
            
            if sources.get("document"):
                all_sources.extend(sources["document"])
            if sources.get("api_data"):
                all_sources.extend(sources["api_data"])
            if sources.get("merged"):
                all_sources.extend(sources["merged"])
                
            # Source selection
            selected_source = st.selectbox(
                "Active Knowledge Source",
                all_sources,
                index=0 if all_sources else None
            )
            
            if selected_source and st.button("Set Active Source"):
                st.session_state.agent.set_active_document(selected_source)
                st.success(f"Set active source: {selected_source}")
                
            # Merge sources
            if len(all_sources) > 1:
                st.subheader("Merge Knowledge Sources")
                merge_sources = st.multiselect(
                    "Select sources to merge",
                    all_sources
                )
                merge_id = st.text_input("Merged source ID", "merged_source")
                
                if merge_sources and st.button("Merge Sources"):
                    with st.spinner("Merging sources..."):
                        result = st.session_state.agent.merge_knowledge_sources(
                            merge_sources,
                            merge_id
                        )
                    
                    if result["status"] == "success":
                        st.success(f"Sources merged: {merge_id}")
                    else:
                        st.error(f"Error: {result['message']}")
            
        # System statistics
        st.subheader("System Statistics")
        if st.button("View Statistics"):
            stats = st.session_state.agent.get_statistics()
            
            st.write("Knowledge Sources:")
            for source_type, source_list in stats["knowledge_sources"].items():
                st.write(f"- {source_type}: {len(source_list)}")
                
            st.write("Metrics:")
            for metric, value in stats["metrics"].items():
                st.write(f"- {metric}: {value}")
                
            st.write("Configuration:")
            for key, value in stats["current_configuration"].items():
                st.write(f"- {key}: {value}")
    
    # Main content area
    if not st.session_state.documents_loaded and not st.session_state.api_connected:
        st.info("👈 Please add a document or connect to the API to get started.")
    else:
        # Chat interface
        st.header("Ask a Question")
        
        # Optional settings
        col1, col2 = st.columns(2)
        with col1:
            refresh_api = st.checkbox("Refresh API data before answering", value=True)
        with col2:
            show_explanation = st.checkbox("Show RAG explanation", value=False)
            
        # User input
        user_question = st.text_input("Your question:")
        
        if user_question:
            if st.button("Ask"):
                # Add question to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Process the question
                result = process_query(user_question, refresh_api)
                
                if result["status"] == "success":
                    # Add answer to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "sources": result["sources"],
                        "explanation": result["explanation"] if show_explanation else None
                    })
                else:
                    # Add error to chat history
                    st.session_state.chat_history.append({
                        "role": "system", 
                        "content": f"Error: {result['message']}"
                    })
        
        # Display chat history
        st.header("Chat History")
        
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You**: {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Assistant**: {msg['content']}")
                
                # Show sources if available
                if "sources" in msg and msg["sources"]:
                    with st.expander("View sources"):
                        st.markdown(format_sources(msg["sources"]))
                        
                # Show explanation if available and enabled
                if "explanation" in msg and msg["explanation"] and show_explanation:
                    with st.expander("RAG process explanation"):
                        for step in msg["explanation"]["process"]:
                            st.subheader(step["step"])
                            for key, value in step["details"].items():
                                st.write(f"- {key}: {value}")
            else:
                st.warning(msg["content"])

if __name__ == "__main__":
    main()
