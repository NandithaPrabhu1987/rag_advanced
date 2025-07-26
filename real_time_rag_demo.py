"""
Real-Time RAG Agent Demo Script

This script demonstrates how to use the Real-Time RAG agent
with both static documents and real-time API data.
"""

import os
import sys
import time
import threading
import subprocess
from typing import Dict, Any

from real_time_rag_agent import RealTimeRAGAgent

def start_mock_api_server():
    """Start the mock API server in a separate process"""
    print("Starting mock API server...")
    try:
        # Check if Flask is installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors"])
        
        # Start server in a separate process
        server_process = subprocess.Popen(
            [sys.executable, "mock_api_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for server to start
        time.sleep(2)
        return server_process
    except Exception as e:
        print(f"Error starting mock API server: {e}")
        return None

def demo_static_documents(agent: RealTimeRAGAgent):
    """Demonstrate using the agent with static documents"""
    print("\n" + "="*50)
    print("DEMO: STATIC DOCUMENT RAG")
    print("="*50)
    
    # Add a document
    sample_doc_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "sample_data", 
        "ai_overview.txt"
    )
    
    print(f"\nAdding document: {sample_doc_path}")
    result = agent.add_document(sample_doc_path, "ai_doc")
    print(f"Result: {result}")
    
    # Ask a question about the document
    question = "What are the main types of artificial intelligence?"
    print(f"\nQuestion: {question}")
    
    answer_result = agent.query_with_explanation(question, refresh_api_data=False)
    
    if answer_result["status"] == "success":
        print("\nAnswer:")
        print(answer_result["answer"])
        
        print("\nSources used:")
        for source in answer_result["sources"]:
            print(f"- Source {source['id']}: {source['source_type']}")
    else:
        print(f"Error: {answer_result['message']}")
        
    return answer_result

def demo_api_integration(agent: RealTimeRAGAgent):
    """Demonstrate integrating real-time API data"""
    print("\n" + "="*50)
    print("DEMO: REAL-TIME API INTEGRATION")
    print("="*50)
    
    # Add API connector
    print("\nAdding API connector...")
    connector_result = agent.add_api_connector(
        "mock_api",
        "http://localhost:8000",
        {
            "stocks": "/api/stocks",
            "news": "/api/news",
            "weather": "/api/weather"
        },
        refresh_interval=60  # refresh every minute
    )
    
    print(f"Result: {connector_result}")
    
    # Refresh API data
    print("\nRefreshing API data...")
    for endpoint in ["stocks", "news", "weather"]:
        result = agent.refresh_api_data("mock_api", endpoint)
        print(f"Refreshed {endpoint}: {result}")
        
    # Set API data as active source
    print("\nSetting stocks as active source...")
    agent.set_active_document("api_mock_api_stocks")
    
    # Ask a question about API data
    question = "What is the current price of Apple stock?"
    print(f"\nQuestion: {question}")
    
    answer_result = agent.query_with_explanation(question)
    
    if answer_result["status"] == "success":
        print("\nAnswer:")
        print(answer_result["answer"])
        
        print("\nSources used:")
        for source in answer_result["sources"]:
            print(f"- Source {source['id']}: {source['source_type']}")
            if source["data_freshness_minutes"] is not None:
                print(f"  Data freshness: {source['data_freshness_minutes']:.1f} minutes")
    else:
        print(f"Error: {answer_result['message']}")
        
    return answer_result

def demo_combined_sources(agent: RealTimeRAGAgent):
    """Demonstrate using both static documents and API data together"""
    print("\n" + "="*50)
    print("DEMO: COMBINED KNOWLEDGE SOURCES")
    print("="*50)
    
    # Merge knowledge sources
    print("\nMerging document and API data...")
    merge_result = agent.merge_knowledge_sources(
        ["ai_doc", "api_mock_api_news", "api_mock_api_stocks"],
        "merged_all_sources"
    )
    print(f"Result: {merge_result}")
    
    # Set merged source as active
    agent.set_active_document("merged_all_sources")
    
    # Ask a question that requires both document and API knowledge
    question = "What are the different types of AI and how might they affect the stock market?"
    print(f"\nQuestion: {question}")
    
    answer_result = agent.query_with_explanation(question)
    
    if answer_result["status"] == "success":
        print("\nAnswer:")
        print(answer_result["answer"])
        
        print("\nSources used:")
        for source in answer_result["sources"]:
            print(f"- Source {source['id']}: {source['source_type']}")
            if "source" in source["metadata"]:
                print(f"  Source: {source['metadata']['source']}")
    else:
        print(f"Error: {answer_result['message']}")
    
    # Display explanation
    print("\nRAG Process Explanation:")
    for step in answer_result["explanation"]["process"]:
        print(f"\n{step['step']}:")
        for k, v in step["details"].items():
            print(f"  - {k}: {v}")
    
    return answer_result

def main():
    """Run the complete demo"""
    print("\nReal-Time RAG Agent Demo")
    print("=======================\n")
    
    # Start mock API server
    server_process = start_mock_api_server()
    if not server_process:
        print("Failed to start mock API server. Exiting.")
        return
        
    try:
        # Initialize agent
        agent = RealTimeRAGAgent()
        
        # Demo with static document
        demo_static_documents(agent)
        
        # Demo with API data
        demo_api_integration(agent)
        
        # Demo with combined sources
        demo_combined_sources(agent)
        
        # Show system statistics
        print("\n" + "="*50)
        print("SYSTEM STATISTICS")
        print("="*50)
        
        stats = agent.get_statistics()
        
        print("\nKnowledge Sources:")
        for source_type, sources in stats["knowledge_sources"].items():
            print(f"  - {source_type}: {len(sources)} sources")
            for s in sources:
                print(f"    - {s}")
        
        print("\nMetrics:")
        for k, v in stats["metrics"].items():
            print(f"  - {k}: {v}")
            
    finally:
        # Terminate the server process
        if server_process:
            print("\nShutting down mock API server...")
            server_process.terminate()
            
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
