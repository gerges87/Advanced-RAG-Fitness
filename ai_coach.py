from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def build_fitrag_coach():
    print("🚀 Initializing FitRAG Coach...")
    
    # 1. تحميل الـ Vector DB (The Memory)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # ضبط الـ Retriever ليجلب أفضل 4 فقرات
    retriever = vectordb.as_retriever(search_kwargs={"k": 15})
    
    # 2. إعداد الـ Local LLM (The Brain)
    # قللنا الـ temperature لـ 0.1 عشان نمنع الموديل من التأليف (Hallucination)
    llm = OllamaLLM(model="llama3.1:8b", temperature=0.1)
    
    # 3. هندسة الـ System Prompt (Strict Guardrails)
    template = """
    You are an expert, professional AI Fitness and Rehabilitation Coach.
    Use ONLY the following pieces of retrieved context to answer the user's question.
    If you don't know the answer or the context doesn't contain the information, simply say "I cannot answer this based on my current trusted medical/fitness knowledge base." 
    DO NOT try to make up an answer. Keep the answer highly professional, clear, and structured.

    Context: {context}
    
    Question: {input}
    
    Coach's Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 4. بناء سلسلة المعالجة (The RAG Chain)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain

# ---------------------------------------------------------
# (Execution)
# ---------------------------------------------------------
if __name__ == "__main__":
    coach_chain = build_fitrag_coach()
    
    print("\n" + "="*50)
    print("🤖 FitRAG Coach is Ready! Type your question below.")
    print("="*50 + "\n")
    
    # حلقة تكرارية تفاعلية لسؤال الموديل
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Coach: Stay strong! Goodbye.")
            break
            
        if not user_query.strip():
            continue
            
        print("Coach is thinking... 🤔")
        
        try:
            # تمرير السؤال للـ Pipeline
            response = coach_chain.invoke({"input": user_query})
            
            print("\n💪 Coach:")
            print(response["answer"])
            print("\n" + "-"*50)
            
        except Exception as e:
            print(f"\n❌ An error occurred: {e}\n")