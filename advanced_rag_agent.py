"""
Advanced RAG Agent with LangChain
=================================

This script demonstrates a more sophisticated implementation of RAG (Retrieval-Augmented Generation)
using LangChain. It features:

1. Multi-document handling
2. Different chunking strategies
3. Query transformations
4. Advanced retrieval methods
5. Result reranking
6. Source attribution and citations
7. Self-evaluation and explanation of RAG process
"""

import os
import time
from typing import List, Dict, Any

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

class AdvancedRAGAgent:
    """An advanced RAG agent with multiple strategies and introspection capabilities"""
    
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
            "avg_response_time": 0
        }
        
        # Document store
        self.documents = {}
        self.vector_stores = {}
        self.current_vector_store = None
        self.active_strategy = "recursive"
        
    def add_document(self, file_path: str, document_id: str = None):
        """Add a document to the RAG system"""
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
    
    def add_document_directory(self, dir_path: str):
        """Add all text documents from a directory"""
        try:
            loader = DirectoryLoader(dir_path, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            
            # Use directory name as document ID
            document_id = os.path.basename(dir_path)
            
            # Store original documents
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
                "num_files": len(documents),
                "strategy": self.active_strategy
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def set_chunking_strategy(self, strategy: str):
        """Set the active chunking strategy"""
        if strategy in self.chunking_strategies:
            self.active_strategy = strategy
            return {"status": "success", "strategy": strategy}
        else:
            return {
                "status": "error",
                "message": f"Strategy {strategy} not found. Available: {list(self.chunking_strategies.keys())}"
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
    
    def merge_documents(self, document_ids: List[str], output_id: str = "merged"):
        """Merge multiple document collections into a single vector store"""
        if not all(doc_id in self.vector_stores for doc_id in document_ids):
            missing = [doc_id for doc_id in document_ids if doc_id not in self.vector_stores]
            return {
                "status": "error",
                "message": f"Documents not found: {missing}"
            }
            
        try:
            # Combine all documents
            all_documents = []
            for doc_id in document_ids:
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
                "merged_from": document_ids
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
    
    def query_with_explanation(self, question: str):
        """Process a question and provide the answer along with explanation of the RAG process"""
        
        # Record start time for metrics
        start_time = time.time()
        
        if not self.current_vector_store:
            return {
                "status": "error",
                "message": "No documents loaded. Please add a document first."
            }
            
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
            explanation = self._explain_rag_process(question, transformed_query, source_docs, elapsed_time)
            
            return {
                "status": "success",
                "answer": answer,
                "sources": [self._format_source(i, doc) for i, doc in enumerate(source_docs, 1)],
                "explanation": explanation,
                "metrics": {
                    "elapsed_time": elapsed_time,
                    "num_sources": len(source_docs)
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
            
        return {
            "id": idx,
            "content": content,
            "metadata": doc.metadata
        }
    
    def _explain_rag_process(self, original_question: str, transformed_query: str, source_docs: List[Document], time_taken: float):
        """Generate an explanation of the RAG process for transparency"""
        explanation = {
            "process": [
                {"step": "Query Transformation", "details": {
                    "original": original_question,
                    "transformed": transformed_query
                }},
                {"step": "Vector Search", "details": {
                    "embedding_model": self.embeddings.model_name,
                    "retrieval_strategy": "Similarity search with contextual compression",
                    "num_chunks_retrieved": len(source_docs)
                }},
                {"step": "Document Processing", "details": {
                    "chunking_strategy": self.active_strategy,
                    "sources_used": len(source_docs)
                }},
                {"step": "LLM Generation", "details": {
                    "model": self.llm.model_name,
                    "context_window_used": sum(len(doc.page_content) for doc in source_docs)
                }}
            ],
            "performance": {
                "time_taken": f"{time_taken:.2f} seconds",
                "total_queries": self.metrics["queries_processed"],
                "avg_response_time": f"{self.metrics['avg_response_time']:.2f} seconds"
            }
        }
        return explanation
    
    def get_available_documents(self):
        """Get list of available documents"""
        return list(self.vector_stores.keys())
    
    def get_statistics(self):
        """Get statistics about the system"""
        return {
            "documents": {
                "count": len(self.vector_stores),
                "ids": list(self.vector_stores.keys())
            },
            "metrics": self.metrics,
            "current_configuration": {
                "active_document": next((k for k, v in self.vector_stores.items() if v == self.current_vector_store), None),
                "chunking_strategy": self.active_strategy,
                "embedding_model": self.embeddings.model_name,
                "llm": self.llm.model_name
            }
        }

# Interactive CLI interface
def main():
    """Run the interactive CLI for the Advanced RAG Agent"""
    print("="*50)
    print("Advanced RAG Agent - Interactive CLI")
    print("="*50)
    
    # Initialize agent
    agent = AdvancedRAGAgent()
    
    # Sample data directory
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
    
    # Create sample data if it doesn't exist
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        
        # Create sample documents
        with open(os.path.join(sample_dir, "ai_ethics.txt"), "w") as f:
            f.write("""# AI Ethics and Governance

Artificial Intelligence ethics addresses the moral and societal implications of AI systems. As AI becomes increasingly integrated into critical domains, ethical considerations have become paramount.

## Key Ethical Concerns

### Bias and Fairness
AI systems can perpetuate and amplify existing societal biases when trained on biased data. For instance, facial recognition systems have shown higher error rates for women and people with darker skin tones. Addressing bias requires diverse training data and regular auditing of AI outputs.

### Transparency and Explainability
Many advanced AI systems, particularly deep learning models, function as "black boxes" where the reasoning behind specific outputs is not easily understood. Explainable AI (XAI) seeks to create systems that can justify their decisions in human-understandable terms.

### Privacy
AI often requires vast amounts of data, raising concerns about privacy, surveillance, and data protection. Techniques like federated learning and differential privacy offer ways to train models while preserving privacy.

### Accountability
As AI systems make increasingly consequential decisions, questions of responsibility become critical. Who is liable when an AI system causes harm—the developer, the user, or the system itself?

## Governance Approaches

### Regulation
Governments worldwide are developing regulatory frameworks for AI. The EU's AI Act proposes a risk-based approach, with stricter requirements for high-risk applications like healthcare or law enforcement.

### Industry Self-regulation
Many technology companies have established AI ethics boards and principles. However, critics question the effectiveness of self-regulation without external oversight.

### Technical Solutions
Researchers are developing technical methods to address ethical concerns, including adversarial testing for robustness, interpretability techniques, and fairness constraints in algorithms.

### Multi-stakeholder Governance
Some advocate for collaborative governance involving industry, government, academia, and civil society to develop standards and best practices.

The field of AI ethics continues to evolve as technology advances and society grapples with the profound implications of increasingly autonomous and capable AI systems.""")
            
        with open(os.path.join(sample_dir, "ai_applications.txt"), "w") as f:
            f.write("""# AI Applications Across Industries

Artificial Intelligence is transforming industries across the global economy. This document explores key applications and impacts in various sectors.

## Healthcare

### Diagnosis and Treatment
AI systems are increasingly effective at analyzing medical images, sometimes outperforming human specialists in detecting conditions like cancer. Machine learning algorithms can identify patterns in patient data to suggest diagnoses or predict disease progression.

### Drug Discovery
AI accelerates drug discovery by predicting how different compounds will behave and identifying promising candidates for clinical trials. This can potentially reduce the time and cost of bringing new treatments to market.

### Personalized Medicine
By analyzing genetic information and health records, AI enables more personalized treatment plans tailored to individual patients' unique characteristics.

## Finance

### Algorithmic Trading
Financial institutions use AI for high-frequency trading, market prediction, and portfolio management. These systems can analyze vast amounts of market data and execute trades at speeds impossible for human traders.

### Risk Assessment
AI improves credit scoring and fraud detection by identifying subtle patterns that might indicate risk or fraudulent activity. This enables more accurate lending decisions and better security for financial transactions.

### Customer Service
Virtual assistants and chatbots handle routine customer queries, while more sophisticated AI systems provide personalized financial advice based on customer data and market trends.

## Transportation

### Autonomous Vehicles
Self-driving technology is advancing rapidly, with potential applications in personal vehicles, trucking, delivery services, and public transportation. These systems combine computer vision, sensor fusion, and decision-making algorithms.

### Traffic Management
Smart traffic systems use AI to optimize signal timing, reduce congestion, and improve fuel efficiency across transportation networks.

### Logistics and Supply Chain
AI optimizes shipping routes, warehouse operations, and inventory management, reducing costs and environmental impact while improving delivery times.

The integration of AI across these industries is not without challenges, including technical limitations, regulatory hurdles, and workforce impacts. However, the potential benefits in efficiency, accuracy, and new capabilities continue to drive adoption and innovation.""")
    
    # Main loop
    while True:
        print("\nWhat would you like to do?")
        print("1. Load document")
        print("2. Load sample documents")
        print("3. View available documents")
        print("4. Change active document")
        print("5. Change chunking strategy")
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
            # Load sample documents
            print("Loading sample documents from:", sample_dir)
            for file in ["ai_ethics.txt", "ai_applications.txt"]:
                file_path = os.path.join(sample_dir, file)
                doc_id = file.split(".")[0]  # Use filename without extension as ID
                result = agent.add_document(file_path, doc_id)
                print(f"Loaded {file}: {result}")
            
        elif choice == "3":
            # View available documents
            docs = agent.get_available_documents()
            if docs:
                print("\nAvailable documents:")
                for i, doc_id in enumerate(docs, 1):
                    print(f"{i}. {doc_id}")
            else:
                print("\nNo documents loaded.")
                
        elif choice == "4":
            # Change active document
            docs = agent.get_available_documents()
            if not docs:
                print("\nNo documents loaded.")
                continue
                
            print("\nAvailable documents:")
            for i, doc_id in enumerate(docs, 1):
                print(f"{i}. {doc_id}")
                
            try:
                idx = int(input("\nEnter document number: ")) - 1
                if 0 <= idx < len(docs):
                    result = agent.set_active_document(docs[idx])
                    print(f"Result: {result}")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")
                
        elif choice == "5":
            # Change chunking strategy
            print("\nAvailable chunking strategies:")
            for i, strategy in enumerate(agent.chunking_strategies.keys(), 1):
                print(f"{i}. {strategy}")
                
            try:
                idx = int(input("\nEnter strategy number: ")) - 1
                strategies = list(agent.chunking_strategies.keys())
                if 0 <= idx < len(strategies):
                    result = agent.set_chunking_strategy(strategies[idx])
                    print(f"Result: {result}")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")
                
        elif choice == "6":
            # Ask a question
            if not agent.get_available_documents():
                print("\nPlease load a document first.")
                continue
                
            question = input("\nEnter your question: ")
            print("\nProcessing...")
            result = agent.query_with_explanation(question)
            
            if result["status"] == "success":
                print("\n" + "="*50)
                print("ANSWER:")
                print(result["answer"])
                
                print("\n" + "="*50)
                print("SOURCES:")
                for source in result["sources"]:
                    print(f"\n[{source['id']}] {source['content']}")
                
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
            
            print("\nDocuments:")
            print(f"  - Count: {stats['documents']['count']}")
            print(f"  - IDs: {', '.join(stats['documents']['ids'])}")
            
            print("\nMetrics:")
            for k, v in stats['metrics'].items():
                print(f"  - {k}: {v}")
                
            print("\nCurrent Configuration:")
            for k, v in stats['current_configuration'].items():
                print(f"  - {k}: {v}")
                
        elif choice == "8":
            # Exit
            print("\nThank you for using the Advanced RAG Agent!")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 8.")

if __name__ == "__main__":
    main()
