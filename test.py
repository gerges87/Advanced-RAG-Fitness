from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

def test_local_stack():
    print("1. Testing Local Embeddings (HuggingFace)...")
    #  Embeddings Memory
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector = embeddings.embed_query("تمرين الرفعة الميتة ممتاز للظهر")
    print(f"✅ Embeddings working! Vector length: {len(vector)}")

    print("\n2. Testing Local LLM (Llama 3 via Ollama)...")
    # ـ Ollama 
    llm = Ollama(model="llama3.1:8b")
    response = llm.invoke("What are the primary muscles used in a deadlift? Answer in one short sentence.")
    print(f"✅ Llama 3 Response: {response}")

if __name__ == "__main__":
    test_local_stack()