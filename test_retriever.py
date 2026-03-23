from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def test_retrieval(query: str, db_dir: str = "./chroma_db"):
    """
    Loads the persisted Vector DB and retrieves the most relevant chunks for a given query.
    """
    print("1. Loading Embedding Model...")
    #Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("2. Loading existing ChromaDB...")
    #  Load ـ Vector DB 
    vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
    print(f"3. Searching for: '{query}'\n")
    #  Retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    # (Retrieval)
    retrieved_docs = retriever.invoke(query)
    
    return retrieved_docs

if __name__ == "__main__":
    # 
    test_query = "What are the best exercises to prevent lower back pain?"
    
    try:
        results = test_retrieval(test_query)
        
        print(f"✅ Found {len(results)} relevant chunks.\n")
        print("--- Top Result (Most Relevant) ---")
        # 
        print(f"Source: {results[0].metadata.get('source', 'Unknown')}")
        print(f"Page: {results[0].metadata.get('page', 'Unknown')}\n")
        print(results[0].page_content)
        print("\n----------------------------------")
        
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")