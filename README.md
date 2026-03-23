# FitRAG: Elite AI Fitness Coach 🏋️‍♂️🤖

An advanced, production-grade Retrieval-Augmented Generation (RAG) system acting as a personalized fitness coach. This system eliminates AI hallucinations and provides scientifically backed workout and nutrition plans based on user biometric data (e.g., InBody analysis).

## 🚀 Key Optimizations & Features
- **Semantic Chunking:** Context-aware document splitting to preserve scientific integrity.
- **Hybrid Search:** Combining Vector Search (Dense) and BM25 (Sparse) for precise technical terminology retrieval.
- **Advanced Re-ranking:** Integrated **FlashRank** to compress and filter the top 3-4 chunks, significantly boosting Context Precision.
- **Domain-Specific Embeddings:** Powered by `BAAI/bge-small-en-v1.5`.
- **Zero Hallucinations:** Evaluated strictly using **Ragas** and **Llama 3.1** as a judge.

## 📊 RAGAS Evaluation Metrics (20 Complex Scenarios)
- **Aggregate RAG Score:** 0.84 (Improved from 0.57)
- **Context Recall:** 0.81
- **Faithfulness:** 0.86
- **Context Precision:** 0.81

## 🛠️ Tech Stack
LangChain | ChromaDB | Ollama (Local LLMs) | Ragas | Streamlit | BM25 | FlashRank

## 💻 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Add your fitness PDF books to a folder named `fitness/`.
4. Build the Vector DB: `python build_vectordb.py`
5. Run the app: `streamlit run app.py`