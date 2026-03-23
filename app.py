import streamlit as st
import ollama
from PIL import Image
import io
import os
import numpy as np
import easyocr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever

# ==========================================
# 1.(Caching)
# ==========================================
st.set_page_config(page_title="FitRAG Pro: Ultimate AI Coach", page_icon="💪", layout="wide")

@st.cache_resource
def get_essentials():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = OllamaLLM(model="llama3.1:8b", temperature=0.1)
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    ocr_reader = easyocr.Reader(['en'])
    return embeddings, llm, vectordb, ocr_reader

# ==========================================
# 2.(Categorization Logic)
# ==========================================
def categorize_books(book_list):
    """تصنيف الكتب تلقائياً بناءً على الكلمات المفتاحية"""
    categories = {
        "🏥 Rehab & Pain Relief": ["back", "shoulder", "knee", "neck", "milo", "pain", "fms", "starret", "mckenzie"],
        "🍎 Nutrition & Diet": ["nutrition", "nutricion", "pyramid-nutricion", "sports-nutrition"],
        "🏋️ Strength & Hypertrophy": ["strength", "muscle", "hypertrophy", "entrenamiento", "conditioning", "basics", "schoenfeld"],
        "🧠 Habits & Psychology": ["habits", "mind", "gym", "atomic"],
        "🧬 Anatomy & Science": ["anatomy", "trains", "scientific", "biomechanics"]
    }
    classified = {cat: [] for cat in categories}
    classified["📁 Others"] = []
    
    for book_path in book_list:
        book_name = os.path.basename(book_path).lower()
        found = False
        for cat, keywords in categories.items():
            if any(key in book_name for key in keywords):
                classified[cat].append(book_path)
                found = True
                break
        if not found: classified["📁 Others"].append(book_path)
    return classified

# ==========================================
# 3. (Vision & Full Segmental OCR)
# ==========================================
def analyze_with_llava(image_bytes):
    try:
        prompt = "Analyze this InBody report. Extract Weight, Muscle Mass, Fat %, and BMR."
        response = ollama.generate(model='llava', prompt=prompt, images=[image_bytes])
        return response['response']
    except Exception as e: return f"Error: {e}"

def analyze_with_hybrid_ocr(image_pil, llm, ocr_reader):
    try:
        image_np = np.array(image_pil)
        raw_text = " ".join(ocr_reader.readtext(image_np, detail=0))
        prompt = f"""Extract ALL metrics from this InBody text:
        - Basic: Weight, SMM, PBF, BMR.
        - Segmental Lean Analysis (kg): Right/Left Arm, Trunk, Right/Left Leg.
        - Segmental Fat Analysis (%): Right/Left Arm, Trunk, Right/Left Leg.
        Raw Text: {raw_text}
        Return strictly as a structured summary."""
        return llm.invoke(prompt)
    except Exception as e: return f"Error: {e}"

# ==========================================
# 4. (Query Router)
# ==========================================
def route_query_to_categories(llm, query):
    """الـ AI يقرر أي قسم من المكتبة سيفتح بناءً على السؤال"""
    prompt = f"""Analyze this fitness query: "{query}"
    Categories: [🏥 Rehab & Pain Relief, 🍎 Nutrition & Diet, 🏋️ Strength & Hypertrophy, 🧠 Habits & Psychology, 🧬 Anatomy & Science]
    NOTE: If the user needs a plan, motivation, or is struggling, ALWAYS include '🧠 Habits & Psychology'.
    Return ONLY a comma-separated list of category names."""
    return llm.invoke(prompt).strip()

# ==========================================
# 5. (FlashRank + Persona)
# ==========================================
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

def get_pro_rag_chain(llm, vectordb, user_data, user_goal, auto_selected_books):
    # 1. 
    search_kwargs = {"k": 20}
    if auto_selected_books:
        search_kwargs["filter"] = {"source": {"$in": auto_selected_books}}
    
    vector_retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

    # 2. 
    # BM25
    all_data = vectordb.get()
    all_docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    
    
    if auto_selected_books:
        filtered_docs = [doc for doc in all_docs if doc.metadata.get("source") in auto_selected_books]
    else:
        filtered_docs = all_docs

    # BM25 Retriever 
    if filtered_docs:
        bm25_retriever = BM25Retriever.from_documents(filtered_docs)
        bm25_retriever.k = 5
    else:
        # Fallback 
        bm25_retriever = vector_retriever 

    # 3. (Ensemble) -
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    # 4. Re-ranker (FlashRank) 
    compressor = FlashrankRerank(top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    # ---Prompts  ---
    system_prompt = f"""You are an Elite AI Fitness Coach & Human Performance Expert. 
    
    YOUR CORE PERSONALITY:
    - Encouraging and disciplined. Use 'Habit Science' to motivate.
    - Analyze Segmental InBody data to detect muscle imbalances (e.g., if one leg is weaker, prioritize it).

    STRICT OUTPUT RULES:
    1. If the user asks for a workout or "building muscle", you MUST provide a structured 4-day or 5-day split (like Upper/Lower or PPL).
    2. Format EVERY workout in a high-quality MARKDOWN TABLE with exactly these columns: 
       | Exercise | Sets x Reps | Rest | Coach's Notes (Form & Tempo) |
    3. For Hypertrophy (Muscle Building): Use a rep range of 8-12 for compound movements and 12-15 for isolation.
    4. MANDATORY: Include a "Habit Insight" section at the end based on 'Atomic Habits'.
    5. DO NOT mention book names or authors. Speak as the coach.
    6. Use ONLY the provided context for technical advice.

    USER DATA: {user_data}
    USER GOAL: {user_goal}
    CONTEXT: {{context}}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "Formulate a standalone question based on history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, compression_retriever, rephrase_prompt)
    
    return create_retrieval_chain(
        history_aware_retriever, 
        create_stuff_documents_chain(llm, qa_prompt)
    )
# ==========================================
# 6.(The UI)
# ==========================================
st.title("🤖 FitRAG Pro: The Agentic Coach")
embeddings, llm, vectordb, ocr_reader = get_essentials()

with st.sidebar:
    st.header("👤 Analysis Engine")
    uploaded_file = st.file_uploader("Upload InBody Report", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True)
        analysis_mode = st.radio("Method:", ["Llava Vision", "Hybrid OCR"])
        if st.button("Run Analysis"):
            with st.spinner("Processing..."):
                if "Vision" in analysis_mode:
                    buf = io.BytesIO(); img.save(buf, format='JPEG')
                    result = analyze_with_llava(buf.getvalue())
                else: result = analyze_with_hybrid_ocr(img, llm, ocr_reader)
                st.session_state.profile_summary = result

    if "profile_summary" in st.session_state:
        st.session_state.profile_summary = st.text_area("InBody Data:", st.session_state.profile_summary, height=150)
        st.divider()
        st.subheader("🎯 Coaching Goal")
        st.session_state.user_goal = st.selectbox("What's our focus?", 
            ["Hypertrophy", "Fat Loss", "Rehab & Mobility", "Mental Toughness"])

# ==========================================
# 7. (The Agent Flow)
# ==========================================
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if user_input := st.chat_input("Ask your coach..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("🧠 AI Routing & Consulting Library...", expanded=False) as status:
            # 1. 
            detected_cats = route_query_to_categories(llm, user_input)
            st.write(f"Routing to: {detected_cats}")
            
            # 2. 
            all_books = [os.path.join("fitness", f) for f in os.listdir("fitness") if f.endswith(".pdf")]
            book_categories = categorize_books(all_books)
            final_selection = []
            for cat in book_categories:
                if cat in detected_cats: final_selection.extend(book_categories[cat])
            
            if not final_selection: final_selection = all_books # Fallback
            status.update(label="✅ Context Found. Generating Plan...", state="complete")

        with st.spinner("Coach is thinking..."):
            profile = st.session_state.get("profile_summary", "No stats.")
            goal = st.session_state.get("user_goal", "General")
            
            chain = get_pro_rag_chain(llm, vectordb, profile, goal, final_selection)
            
            history = [("human" if m["role"] == "user" else "ai", m["content"]) 
                       for m in st.session_state.messages[:-1]]
            
            response = chain.invoke({"input": user_input, "chat_history": history})
            st.markdown(response["answer"])
            
            with st.expander("📚 Verified Science (Behind the Scenes)"):
                for doc in response["context"]:
                    st.write(f"📖 {os.path.basename(doc.metadata['source'])} (Page {doc.metadata.get('page', 'N/A')})")
            
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})