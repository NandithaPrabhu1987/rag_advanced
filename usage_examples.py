"""
Usage Examples for the Advanced RAG Agent
========================================

This script demonstrates how to use the AdvancedRAGAgent programmatically.
"""

from advanced_rag_agent import AdvancedRAGAgent
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))

def main():
    print("="*50)
    print("Advanced RAG Agent - Usage Examples")
    print("="*50)
    
    # Initialize the agent
    agent = AdvancedRAGAgent(model_name="llama-3.3-70b-versatile")
    print("Agent initialized with model: llama-3.3-70b-versatile")
    
    # Get sample data path
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
    ethics_file = os.path.join(sample_dir, "ai_ethics.txt")
    applications_file = os.path.join(sample_dir, "ai_applications.txt")
    
    # Example 1: Load individual documents
    print("\nExample 1: Loading individual documents")
    result = agent.add_document(ethics_file, "ethics")
    print_json(result)
    
    result = agent.add_document(applications_file, "applications")
    print_json(result)
    
    # Example 2: Switch between documents
    print("\nExample 2: Switching active document")
    result = agent.set_active_document("applications")
    print_json(result)
    
    # Example 3: Change chunking strategy
    print("\nExample 3: Changing chunking strategy")
    result = agent.set_chunking_strategy("token")
    print_json(result)
    
    # Example 4: Ask a question about AI in healthcare
    print("\nExample 4: Querying about AI in healthcare")
    result = agent.query_with_explanation("How is AI used in healthcare?")
    print("\nAnswer:")
    print(result["answer"])
    
    print("\nSources used:")
    for source in result["sources"]:
        print(f"Source {source['id']}: {source['content'][:100]}...")
    
    print("\nRAG Process:")
    print_json(result["explanation"])
    
    # Example 5: Merge documents and ask a question that spans both
    print("\nExample 5: Merging documents and asking a cross-document question")
    result = agent.merge_documents(["ethics", "applications"], "ai_knowledge")
    print_json(result)
    
    result = agent.query_with_explanation(
        "What ethical considerations are relevant to AI applications in healthcare?"
    )
    print("\nAnswer (from merged documents):")
    print(result["answer"])
    
    print("\nSources used (should include content from both documents):")
    for source in result["sources"]:
        print(f"Source {source['id']}: {source['content'][:100]}...")
    
    # Example 6: Get system statistics
    print("\nExample 6: System statistics")
    stats = agent.get_statistics()
    print_json(stats)
    
    print("\nExamples completed successfully!")

if __name__ == "__main__":
    main()
