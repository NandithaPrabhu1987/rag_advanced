"""
Real-Time RAG Agent with API Integration
======================================

This module implements a RAG agent that can integrate with real-time API data sources
and combine them with static documents.
"""

import os
import time
import json
import datetime
import requests
from typing import List, Dict, Any, Optional, Union

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document

# Load environment variables
load_dotenv()

class APIConnector:
    """A connector for retrieving data from APIs and converting to documents"""
    
    def __init__(self, base_url: str, endpoints: Dict[str, str], refresh_interval: int = 300):
        """
        Initialize the API connector
        
        Args:
            base_url: Base URL for the API
            endpoints: Dictionary mapping endpoint names to paths
            refresh_interval: How often to refresh data in seconds (default: 5 minutes)
        """
        self.base_url = base_url
        self.endpoints = endpoints
        self.refresh_interval = refresh_interval
        self.last_refresh = {}
        self.cached_data = {}
        
    def get_data(self, endpoint_name: str, force_refresh: bool = False) -> Dict:
        """
        Get data from the specified API endpoint
        
        Args:
            endpoint_name: Name of the endpoint (must be in self.endpoints)
            force_refresh: Whether to force refresh regardless of interval
            
        Returns:
            Dictionary with API response data
        """
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")
            
        current_time = time.time()
        
        # Check if we need to refresh
        needs_refresh = (
            force_refresh or 
            endpoint_name not in self.last_refresh or
            current_time - self.last_refresh.get(endpoint_name, 0) > self.refresh_interval
        )
        
        if needs_refresh:
            url = f"{self.base_url}{self.endpoints[endpoint_name]}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                self.cached_data[endpoint_name] = response.json()
                self.last_refresh[endpoint_name] = current_time
                print(f"Refreshed data from {endpoint_name}")
            except Exception as e:
                print(f"Error refreshing data from {endpoint_name}: {e}")
                # Return cached data if available, otherwise raise
                if endpoint_name not in self.cached_data:
                    raise
                    
        return self.cached_data[endpoint_name]
    
    def to_documents(self, endpoint_name: str, document_format_func=None) -> List[Document]:
        """
        Convert API data to Document objects for RAG
        
        Args:
            endpoint_name: Name of the endpoint to get data from
            document_format_func: Optional function to format the documents
                If not provided, uses a default formatting based on the endpoint
                
        Returns:
            List of Document objects
        """
        data = self.get_data(endpoint_name)
        
        if document_format_func:
            return document_format_func(data)
            
        # Default formatting based on endpoint
        documents = []
        
        if endpoint_name == "stocks":
            # Format stock data
            for ticker, stock_info in data.items():
                content = f"""Stock: {ticker} ({stock_info['name']})
Sector: {stock_info['sector']}
Current Price: ${stock_info['current_price']}
Previous Close: ${stock_info.get('previous_close', 'N/A')}
"""
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": f"API:{endpoint_name}",
                        "ticker": ticker,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                documents.append(doc)
                
        elif endpoint_name == "news":
            # Format news data
            for article in data.get("articles", []):
                content = f"""Title: {article['title']}
Source: {article['source']}
Date: {article['date']}

{article['content']}
"""
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": f"API:{endpoint_name}",
                        "id": article["id"],
                        "title": article["title"],
                        "date": article["date"],
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                documents.append(doc)
                
        elif endpoint_name == "weather":
            # Format weather data
            for city, weather_info in data.items():
                if city == "last_updated":
                    continue
                    
                content = f"""Weather for {city}:
Condition: {weather_info['condition']}
Temperature: {weather_info['temperature']}°C
"""
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": f"API:{endpoint_name}",
                        "city": city,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                documents.append(doc)
                
        return documents

class RealTimeRAGAgent:
    """A RAG agent capable of integrating real-time API data with static documents"""
    
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        """Initialize the RAG agent with models and configurations"""
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize LLM
        self.llm = ChatGroq(
            temperature=0.2,
            model_name=model_name
        )
        
        # Document chunking strategies
        self.chunking_strategies = {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            "token": TokenTextSplitter(
                chunk_size=300,
                chunk_overlap=50
            )
        }
        
        # Track metrics for transparency
        self.metrics = {
            "queries_processed": 0,
            "chunks_retrieved": 0,
            "avg_response_time": 0,
            "api_calls": 0
        }
        
        # Document store
        self.documents = {}
        self.vector_stores = {}
        self.current_vector_store = None
        self.active_strategy = "recursive"
        
        # API connectors
        self.api_connectors = {}
        self.api_refresh_intervals = {}
        
    def add_document(self, file_path: str, document_id: str = None):
        """Add a static document to the RAG system"""
        if document_id is None:
            document_id = os.path.basename(file_path)
            
        # Load document
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            
            # Store original document
            self.documents[document_id] = documents
            
            # Process with current strategy and create vector store
            chunks = self.chunking_strategies[self.active_strategy].split_documents(documents)
            
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_stores[document_id] = vector_store
            
            # Set as current if first document
            if self.current_vector_store is None:
                self.current_vector_store = vector_store
                
            return {
                "status": "success",
                "document_id": document_id,
                "num_chunks": len(chunks),
                "strategy": self.active_strategy
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def add_api_connector(self, 
                          connector_id: str, 
                          base_url: str, 
                          endpoints: Dict[str, str],
                          refresh_interval: int = 300):
        """
        Add an API connector as a data source
        
        Args:
            connector_id: Unique identifier for this connector
            base_url: Base URL for the API
            endpoints: Dictionary mapping endpoint names to paths
            refresh_interval: How often to refresh in seconds
        """
        try:
            connector = APIConnector(base_url, endpoints, refresh_interval)
            self.api_connectors[connector_id] = connector
            self.api_refresh_intervals[connector_id] = refresh_interval
            
            return {
                "status": "success",
                "connector_id": connector_id,
                "endpoints": list(endpoints.keys())
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def refresh_api_data(self, connector_id: str, endpoint_name: str, force: bool = False):
        """
        Refresh data from an API endpoint and update the vector store
        
        Args:
            connector_id: ID of the connector to use
            endpoint_name: Name of the endpoint to refresh
            force: Whether to force refresh even if interval hasn't passed
        """
        if connector_id not in self.api_connectors:
            return {
                "status": "error",
                "message": f"Connector {connector_id} not found"
            }
            
        try:
            # Get API connector
            connector = self.api_connectors[connector_id]
            
            # Get documents from API
            documents = connector.to_documents(endpoint_name)
            
            # Create document ID for this endpoint
            doc_id = f"api_{connector_id}_{endpoint_name}"
            
            # Store documents
            self.documents[doc_id] = documents
            
            # Process documents and update vector store
            chunks = self.chunking_strategies[self.active_strategy].split_documents(documents)
            
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_stores[doc_id] = vector_store
            
            # Set as current if no current store
            if self.current_vector_store is None:
                self.current_vector_store = vector_store
            
            # Update metrics
            self.metrics["api_calls"] += 1
            
            return {
                "status": "success",
                "document_id": doc_id,
                "num_chunks": len(chunks),
                "data_timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def set_active_document(self, document_id: str):
        """Set which document to use for retrievals"""
        if document_id in self.vector_stores:
            self.current_vector_store = self.vector_stores[document_id]
            return {"status": "success", "document_id": document_id}
        else:
            return {
                "status": "error",
                "message": f"Document {document_id} not found. Available: {list(self.vector_stores.keys())}"
            }
    
    def merge_knowledge_sources(self, source_ids: List[str], output_id: str = "merged"):
        """
        Merge multiple knowledge sources (static docs and API data) into a unified view
        
        Args:
            source_ids: List of document IDs to merge
            output_id: ID for the merged result
        """
        if not all(doc_id in self.vector_stores for doc_id in source_ids):
            missing = [doc_id for doc_id in source_ids if doc_id not in self.vector_stores]
            return {
                "status": "error",
                "message": f"Documents not found: {missing}"
            }
            
        try:
            # Combine all documents
            all_documents = []
            for doc_id in source_ids:
                all_documents.extend(self.documents[doc_id])
                
            # Create new vector store
            chunks = self.chunking_strategies[self.active_strategy].split_documents(all_documents)
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            # Store merged collection
            self.documents[output_id] = all_documents
            self.vector_stores[output_id] = vector_store
            self.current_vector_store = vector_store
            
            return {
                "status": "success",
                "document_id": output_id,
                "num_chunks": len(chunks),
                "sources": source_ids
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _create_qa_chain(self):
        """Create the RAG chain with the current configuration"""
        
        # Enhance retriever with contextual compression
        compressor = LLMChainExtractor.from_llm(self.llm)
        retriever = self.current_vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        
        # Create custom prompt template with instructions to cite sources
        template = """You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        1. Answer the question based only on the provided context
        2. If you cannot answer the question with the context, say "I don't have enough information to answer that question."
        3. Cite the specific parts of the context you used in your answer
        4. Provide a confidence level (low, medium, high) for your answer
        5. Be sure to indicate if you're using real-time API data in your answer
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=compression_retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
    
    def query_with_explanation(self, question: str, refresh_api_data: bool = True):
        """Process a question and provide the answer with explanation of the RAG process"""
        
        # Record start time for metrics
        start_time = time.time()
        
        if not self.current_vector_store:
            return {
                "status": "error",
                "message": "No documents loaded. Please add a document or API data first."
            }
            
        # Refresh API data if requested
        if refresh_api_data:
            api_sources_refreshed = []
            for connector_id, connector in self.api_connectors.items():
                for endpoint in connector.endpoints:
                    refresh_result = self.refresh_api_data(connector_id, endpoint)
                    if refresh_result["status"] == "success":
                        api_sources_refreshed.append(f"{connector_id}:{endpoint}")
            
            # If we refreshed any APIs and we're using a merged source, recreate it
            current_doc_id = next((k for k, v in self.vector_stores.items() 
                                if v == self.current_vector_store), None)
            if current_doc_id and current_doc_id.startswith("merged_") and api_sources_refreshed:
                # Get the sources that were merged
                source_ids = [s for s in self.vector_stores.keys() 
                             if not s.startswith("merged_")]
                self.merge_knowledge_sources(source_ids, current_doc_id)
            
        # Create QA chain
        qa_chain = self._create_qa_chain()
        
        # Process question
        try:
            # Perform query transformation to improve retrieval
            query_transformer_prompt = f"""
            Transform the following question into a search query that will help retrieve 
            relevant context from a vector store. Focus on key entities and concepts:
            
            Question: {question}
            
            Transformed Query:"""
            
            transformed_query = self.llm.invoke(query_transformer_prompt).content
            
            # Run main query
            result = qa_chain({"query": transformed_query})
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Update metrics
            self.metrics["queries_processed"] += 1
            self.metrics["chunks_retrieved"] += len(source_docs)
            
            elapsed_time = time.time() - start_time
            self.metrics["avg_response_time"] = (
                (self.metrics["avg_response_time"] * (self.metrics["queries_processed"] - 1) + elapsed_time) / 
                self.metrics["queries_processed"]
            )
            
            # Generate explanation of RAG process
            explanation = self._explain_rag_process(question, transformed_query, source_docs, elapsed_time, 
                                                 api_refreshed=refresh_api_data)
            
            return {
                "status": "success",
                "answer": answer,
                "sources": [self._format_source(i, doc) for i, doc in enumerate(source_docs, 1)],
                "explanation": explanation,
                "metrics": {
                    "elapsed_time": elapsed_time,
                    "num_sources": len(source_docs),
                    "api_data_refreshed": refresh_api_data
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _format_source(self, idx: int, doc: Document):
        """Format a source document for display"""
        # Limit content length for display
        content = doc.page_content
        if len(content) > 200:
            content = content[:200] + "..."
            
        # Check if it's an API source and add freshness info
        source_type = "Document"
        freshness = None
        if "source" in doc.metadata and doc.metadata["source"].startswith("API:"):
            source_type = "API Data"
            if "timestamp" in doc.metadata:
                timestamp = datetime.datetime.fromisoformat(doc.metadata["timestamp"])
                now = datetime.datetime.now()
                freshness = (now - timestamp).total_seconds() / 60  # minutes
        
        return {
            "id": idx,
            "content": content,
            "metadata": doc.metadata,
            "source_type": source_type,
            "data_freshness_minutes": freshness
        }
    
    def _explain_rag_process(self, original_question: str, transformed_query: str, 
                         source_docs: List[Document], time_taken: float, api_refreshed: bool = False):
        """Generate an explanation of the RAG process for transparency"""
        
        # Check what types of sources were used
        api_sources_used = []
        static_sources_used = []
        
        for doc in source_docs:
            if "source" in doc.metadata:
                if doc.metadata["source"].startswith("API:"):
                    api_source = doc.metadata["source"].split(":", 1)[1]
                    if api_source not in api_sources_used:
                        api_sources_used.append(api_source)
                else:
                    static_source = doc.metadata["source"]
                    if static_source not in static_sources_used:
                        static_sources_used.append(static_source)
        
        explanation = {
            "process": [
                {"step": "API Data Refresh", "details": {
                    "performed": api_refreshed,
                    "api_sources": list(self.api_connectors.keys()) if api_refreshed else []
                }},
                {"step": "Query Transformation", "details": {
                    "original": original_question,
                    "transformed": transformed_query
                }},
                {"step": "Vector Search", "details": {
                    "embedding_model": self.embeddings.model_name,
                    "retrieval_strategy": "Similarity search with contextual compression",
                    "num_chunks_retrieved": len(source_docs)
                }},
                {"step": "Source Analysis", "details": {
                    "api_sources_used": api_sources_used,
                    "static_sources_used": static_sources_used,
                    "total_sources_used": len(source_docs)
                }},
                {"step": "LLM Generation", "details": {
                    "model": self.llm.model_name,
                    "context_window_used": sum(len(doc.page_content) for doc in source_docs)
                }}
            ],
            "performance": {
                "time_taken": f"{time_taken:.2f} seconds",
                "total_queries": self.metrics["queries_processed"],
                "avg_response_time": f"{self.metrics['avg_response_time']:.2f} seconds",
                "total_api_calls": self.metrics["api_calls"]
            }
        }
        return explanation
    
    def get_available_knowledge_sources(self):
        """Get list of available knowledge sources"""
        sources = {}
        
        # Categorize sources
        for doc_id in self.vector_stores.keys():
            if doc_id.startswith("api_"):
                source_type = "api_data"
            elif doc_id.startswith("merged_"):
                source_type = "merged"
            else:
                source_type = "document"
                
            if source_type not in sources:
                sources[source_type] = []
                
            sources[source_type].append(doc_id)
            
        return sources
    
    def get_statistics(self):
        """Get statistics about the system"""
        return {
            "knowledge_sources": self.get_available_knowledge_sources(),
            "metrics": self.metrics,
            "api_connectors": {
                connector_id: {
                    "base_url": connector.base_url,
                    "endpoints": list(connector.endpoints.keys()),
                    "refresh_interval": self.api_refresh_intervals[connector_id],
                    "last_refresh": {
                        endpoint: datetime.datetime.fromtimestamp(timestamp).isoformat()
                        for endpoint, timestamp in connector.last_refresh.items()
                    } if hasattr(connector, 'last_refresh') else {}
                }
                for connector_id, connector in self.api_connectors.items()
            },
            "current_configuration": {
                "active_source": next((k for k, v in self.vector_stores.items() 
                                    if v == self.current_vector_store), None),
                "chunking_strategy": self.active_strategy,
                "embedding_model": self.embeddings.model_name,
                "llm": self.llm.model_name
            }
        }

# Interactive CLI interface
def main():
    """Run the interactive CLI for the Real-Time RAG Agent"""
    import os
    
    print("="*50)
    print("Real-Time RAG Agent with API Integration - Interactive CLI")
    print("="*50)
    
    # Initialize agent
    agent = RealTimeRAGAgent()
    
    # Sample data directory
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
    
    # Main loop
    while True:
        print("\nWhat would you like to do?")
        print("1. Load document")
        print("2. Connect to API")
        print("3. Refresh API data")
        print("4. View knowledge sources")
        print("5. Merge knowledge sources")
        print("6. Ask a question")
        print("7. View system statistics")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == "1":
            # Load document
            file_path = input("Enter document path: ")
            doc_id = input("Enter document ID (or press Enter for default): ") or None
            result = agent.add_document(file_path, doc_id)
            print(f"\nResult: {result}")
            
        elif choice == "2":
            # Connect to API
            print("\nEnter API details:")
            connector_id = input("Connector ID: ")
            base_url = input("Base URL: ")
            
            # Get endpoints
            endpoints = {}
            print("\nEnter endpoints (leave blank to finish):")
            while True:
                name = input("Endpoint name: ")
                if not name:
                    break
                path = input("Endpoint path: ")
                endpoints[name] = path
                
            if not endpoints:
                print("No endpoints specified. Using defaults for demo API.")
                endpoints = {
                    "stocks": "/api/stocks",
                    "news": "/api/news",
                    "weather": "/api/weather"
                }
                
            refresh_interval = input("Refresh interval in seconds (default: 300): ")
            if not refresh_interval:
                refresh_interval = 300
            else:
                refresh_interval = int(refresh_interval)
                
            result = agent.add_api_connector(connector_id, base_url, endpoints, refresh_interval)
            print(f"\nResult: {result}")
            
            # Refresh initial data
            if result["status"] == "success":
                print("\nFetching initial data...")
                for endpoint in endpoints:
                    refresh_result = agent.refresh_api_data(connector_id, endpoint)
                    print(f"Loaded {endpoint}: {refresh_result}")
            
        elif choice == "3":
            # Refresh API data
            api_connectors = agent.api_connectors
            if not api_connectors:
                print("\nNo API connectors available.")
                continue
                
            print("\nAvailable API connectors:")
            for i, connector_id in enumerate(api_connectors.keys(), 1):
                print(f"{i}. {connector_id}")
                
            try:
                idx = int(input("\nEnter connector number: ")) - 1
                connector_ids = list(api_connectors.keys())
                if 0 <= idx < len(connector_ids):
                    connector_id = connector_ids[idx]
                    
                    # Get endpoints
                    connector = api_connectors[connector_id]
                    print(f"\nEndpoints for {connector_id}:")
                    for i, endpoint in enumerate(connector.endpoints.keys(), 1):
                        print(f"{i}. {endpoint}")
                        
                    endpoint_idx = int(input("\nEnter endpoint number: ")) - 1
                    endpoints = list(connector.endpoints.keys())
                    if 0 <= endpoint_idx < len(endpoints):
                        endpoint = endpoints[endpoint_idx]
                        
                        print(f"\nRefreshing data for {connector_id}:{endpoint}...")
                        result = agent.refresh_api_data(connector_id, endpoint, force=True)
                        print(f"Result: {result}")
                    else:
                        print("Invalid endpoint selection.")
                else:
                    print("Invalid connector selection.")
            except ValueError:
                print("Please enter a valid number.")
                
        elif choice == "4":
            # View knowledge sources
            sources = agent.get_available_knowledge_sources()
            if not any(sources.values()):
                print("\nNo knowledge sources available.")
                continue
                
            print("\nAvailable knowledge sources:")
            
            if sources.get("document"):
                print("\nDocuments:")
                for i, doc_id in enumerate(sources["document"], 1):
                    print(f"{i}. {doc_id}")
                    
            if sources.get("api_data"):
                print("\nAPI Data:")
                for i, doc_id in enumerate(sources["api_data"], 1):
                    print(f"{i}. {doc_id}")
                    
            if sources.get("merged"):
                print("\nMerged Sources:")
                for i, doc_id in enumerate(sources["merged"], 1):
                    print(f"{i}. {doc_id}")
                    
            # Set active source
            change = input("\nChange active source? (y/n): ")
            if change.lower() == "y":
                source_id = input("Enter source ID: ")
                result = agent.set_active_document(source_id)
                print(f"Result: {result}")
                
        elif choice == "5":
            # Merge knowledge sources
            sources = list(agent.vector_stores.keys())
            if not sources:
                print("\nNo knowledge sources available to merge.")
                continue
                
            print("\nAvailable knowledge sources:")
            for i, source_id in enumerate(sources, 1):
                print(f"{i}. {source_id}")
                
            print("\nEnter source numbers to merge (comma-separated):")
            try:
                selections = input("> ")
                indices = [int(idx.strip()) - 1 for idx in selections.split(",")]
                
                if all(0 <= idx < len(sources) for idx in indices):
                    source_ids = [sources[idx] for idx in indices]
                    
                    output_id = input("Enter output ID for merged source: ")
                    if not output_id:
                        output_id = "merged_" + "_".join(source_ids)
                        
                    result = agent.merge_knowledge_sources(source_ids, output_id)
                    print(f"\nResult: {result}")
                else:
                    print("Invalid source selection.")
            except ValueError:
                print("Please enter valid numbers.")
                
        elif choice == "6":
            # Ask a question
            if not agent.current_vector_store:
                print("\nNo active knowledge source. Please load a document or API data first.")
                continue
                
            question = input("\nEnter your question: ")
            
            refresh = input("Refresh API data before answering? (y/n, default: y): ")
            refresh_api = refresh.lower() != "n"
            
            print("\nProcessing...")
            result = agent.query_with_explanation(question, refresh_api_data=refresh_api)
            
            if result["status"] == "success":
                print("\n" + "="*50)
                print("ANSWER:")
                print(result["answer"])
                
                print("\n" + "="*50)
                print("SOURCES:")
                for source in result["sources"]:
                    freshness = ""
                    if source["source_type"] == "API Data" and source["data_freshness_minutes"] is not None:
                        minutes = source["data_freshness_minutes"]
                        freshness = f"[Data from {minutes:.1f} minutes ago]"
                    
                    print(f"\n[{source['id']}] {source['source_type']} {freshness}")
                    print(f"Content: {source['content']}")
                
                print("\n" + "="*50)
                print("RAG PROCESS EXPLANATION:")
                for step in result["explanation"]["process"]:
                    print(f"\n{step['step']}:")
                    for k, v in step["details"].items():
                        print(f"  - {k}: {v}")
                
                print(f"\nPerformance:")
                for k, v in result["explanation"]["performance"].items():
                    print(f"  - {k}: {v}")
            else:
                print(f"\nError: {result['message']}")
                
        elif choice == "7":
            # View system statistics
            stats = agent.get_statistics()
            print("\n" + "="*50)
            print("SYSTEM STATISTICS:")
            
            print("\nKnowledge Sources:")
            for source_type, sources in stats["knowledge_sources"].items():
                print(f"  - {source_type}: {len(sources)} sources")
                for s in sources:
                    print(f"    - {s}")
            
            print("\nAPI Connectors:")
            for connector_id, details in stats["api_connectors"].items():
                print(f"  - {connector_id}:")
                print(f"    - URL: {details['base_url']}")
                print(f"    - Endpoints: {', '.join(details['endpoints'])}")
                print(f"    - Refresh interval: {details['refresh_interval']} seconds")
                
            print("\nMetrics:")
            for k, v in stats['metrics'].items():
                print(f"  - {k}: {v}")
                
            print("\nCurrent Configuration:")
            for k, v in stats['current_configuration'].items():
                print(f"  - {k}: {v}")
                
        elif choice == "8":
            # Exit
            print("\nThank you for using the Real-Time RAG Agent!")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 8.")

if __name__ == "__main__":
    main()
