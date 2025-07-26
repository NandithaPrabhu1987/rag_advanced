# Advanced RAG Agent with LangChain

This repository contains an advanced implementation of Retrieval-Augmented Generation (RAG) using LangChain. The implementation demonstrates key concepts and techniques that enhance traditional RAG pipelines.

## Features

1. **Multi-document handling**:
   - Load individual documents or entire directories
   - Switch between document collections
   - Merge multiple document collections into unified knowledge bases

2. **Advanced chunking strategies**:
   - Recursive character splitting (semantically-aware)
   - Token-based splitting (optimized for LLM context windows)
   - Toggle between strategies depending on content

3. **Sophisticated retrieval pipeline**:
   - Query transformation to improve search effectiveness
   - Contextual compression to extract most relevant information
   - Re-ranking of search results for better context selection

4. **Transparency and explainability**:
   - Detailed explanation of the RAG process
   - Source attribution and citations
   - Confidence scoring
   - Performance metrics and statistics

5. **Interactive CLI interface**:
   - Load and manage documents
   - Switch between strategies
   - Ask questions and see detailed results
   - View system statistics

## Getting Started

### Prerequisites

- Python 3.8+
- LangChain libraries
- A Groq API key (in .env file)
- HuggingFace libraries for embeddings
- Flask and Flask-CORS (for the API server)
- Streamlit (for the web interface)

### Installation

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install all required packages:
   ```bash
   pip install langchain langchain-community langchain-groq python-dotenv faiss-cpu sentence-transformers flask flask-cors streamlit
   ```

   Or install from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variable by creating a `.env` file from the example:
   ```bash
   cp example.env .env
   ```

4. Edit the `.env` file to add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

### Running the Agent

There are several ways to use the RAG agent:

#### Option 1: Interactive CLI

Run the basic CLI interface:

```bash
python advanced_rag_agent.py
```

#### Option 2: Demo Script

Run the demonstration script that shows various features:

```bash
python real_time_rag_demo.py
```

#### Option 3: Web Interface with Streamlit

1. First, start the mock API server in one terminal:
   ```bash
   python mock_api_server.py
   ```

2. Then, in another terminal, start the Streamlit web interface:
   ```bash
   streamlit run real_time_rag_streamlit.py
   ```

3. Open your browser and go to http://localhost:8501 to use the web interface.

You can also let the Streamlit app start the API server automatically by clicking the "Connect to Mock API" button in the sidebar.

## Real-Time RAG with API Integration

This implementation includes a sophisticated Real-Time RAG system that integrates both static documents and live API data.

### Key Features of Real-Time RAG

1. **API Data Integration**:
   - Connect to external APIs for real-time information
   - Automatically refresh data at configurable intervals
   - Transform API responses into searchable documents

2. **Unified Knowledge Base**:
   - Seamlessly combine static documents and API data
   - Query across all knowledge sources simultaneously
   - Merge knowledge sources for comprehensive answers

3. **Web Interface with Streamlit**:
   - User-friendly web application
   - Upload and manage documents
   - Connect to APIs with a single click
   - View detailed sources and explanations

4. **Mock API Server**:
   - Sample API for demonstration purposes
   - Endpoints for stocks, news, and weather data
   - Simulated real-time data updates

## Understanding the RAG Pipeline

This implementation showcases several important concepts in modern RAG systems:

### 1. Document Processing
Documents are loaded, chunked using different strategies, and stored for later retrieval.

### 2. Embedding and Indexing
Text chunks are converted to vector embeddings using the HuggingFace embedding model and indexed in a FAISS vector database.

### 3. Query Processing
When a question is asked:
   - The question is transformed to optimize retrieval
   - The transformed query is converted to a vector
   - Similar document chunks are retrieved from the vector store
   - A contextual compressor extracts the most relevant information

### 4. Answer Generation
The LLM generates an answer based on:
- The original question
- The retrieved context
- Instructions to cite sources and provide confidence levels

### 5. Explanation Generation
The system provides a detailed explanation of how the answer was generated, including:
- How the query was transformed
- Which retrieval methods were used
- Which document chunks were referenced
- Performance metrics and statistics

## Sample Data

The system includes sample documents on AI ethics and AI applications that demonstrate the RAG capabilities. You can also use your own text documents.

## Using the Real-Time RAG Streamlit Interface

The Streamlit interface provides an intuitive way to interact with the Real-Time RAG system:

### Document Management
1. **Upload Documents**:
   - Use the file uploader in the sidebar to add documents
   - Optionally provide a custom document ID
   - Click "Add Document" to process and index the file

2. **API Connection**:
   - Click "Connect to Mock API" to start the API server and connect
   - Use "Refresh API Data" to force an update of all API endpoints
   - API data is automatically refreshed at the configured interval

3. **Knowledge Source Selection**:
   - Select a specific document or API data source from the dropdown
   - Click "Set Active Source" to focus queries on that source
   - Use "Merge Sources" to combine multiple knowledge sources

4. **Asking Questions**:
   - Type your question in the main text input
   - Toggle "Refresh API data before answering" to get the latest information
   - Enable "Show RAG explanation" for detailed process information
   - View source documents by expanding the "View sources" section

## Troubleshooting

### Common Issues

1. **API Connection Fails**:
   - Ensure Flask and Flask-CORS are installed
   - Check that port 8000 is available
   - Make sure the mock_api_server.py file is in the same directory

2. **LLM API Errors**:
   - Verify your Groq API key is correctly set in the .env file
   - Check your internet connection
   - Make sure the API service is operational

3. **Streamlit Interface Issues**:
   - Run Streamlit with the --server.headless option to avoid registration prompts
   - If the interface appears without styling, try a different browser
   - For port conflicts, add `--server.port=XXXX` to use a different port

## Extending the System

This implementation can be extended in several ways:
- Add additional embedding models
- Implement more sophisticated re-ranking strategies
- Add support for different document types (PDFs, websites, etc.)
- Implement memory or conversation history
- Connect to real external APIs instead of the mock server
- Add authentication for multi-user support

## License

This project is available under --- license
