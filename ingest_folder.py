from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_and_chunk_directory(directory_path: str):
    """
    Loads all PDF documents from a directory and splits them into optimal chunks.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # 1. (Directory Loading)
    print(f"Loading PDFs from directory: {directory_path}...")
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    print(f"Loaded a total of {len(documents)} pages from all PDFs.")
    
    # 2. (Character-based Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # 3.Chunks
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Successfully generated {len(chunks)} chunks from the entire library.")
    
    return chunks

# ---------------------------------------------------------
# (Testing)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 
    # 
    folder_path = "fitness" 
    
    try:
        all_chunks = load_and_chunk_directory(folder_path)
        # 
        print("\n--- Sample Chunk (Index 0) ---")
        if len(all_chunks) > 0:
            print(all_chunks[0].page_content)
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")