from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
import os
import shutil

def create_and_persist_vectordb(folder_path: str, persist_dir: str = "./chroma_db"):
    # 0.
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"🧹 Old DB removed from {persist_dir}")

    # 1.  
    print(f"📂 Loading PDFs from {folder_path}...")
    loader = PyPDFDirectoryLoader(folder_path)
    raw_documents = loader.load()
    
    if not raw_documents:
        print("❌ No documents found. Exiting.")
        return None

    # 2. Embeddings (BGE)
    print("🧠 Loading Pro Embedding Model (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 3. Semantic Chunking 
    print("✂️ Splitting documents by MEANING (Semantic Chunking)...")
    #
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile", # 
        breakpoint_threshold_amount=85 # 
    )
    
    # 
    semantic_chunks = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(semantic_chunks)} smart chunks.")

    # 4. 
    print(f"📦 Embedding and saving to ChromaDB...")
    vectordb = Chroma.from_documents(
        documents=semantic_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"🚀 SUCCESS! Vector DB is now 'Semantic-Aware'.")
    return vectordb

if __name__ == "__main__":
    TARGET_FOLDER = "fitness" 
    DB_DIRECTORY = "./chroma_db"
    create_and_persist_vectordb(TARGET_FOLDER, DB_DIRECTORY)