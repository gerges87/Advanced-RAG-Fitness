import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import time
import os
import math 
from ragas.run_config import RunConfig 

# ---  app.py ---
from app import get_essentials, get_pro_rag_chain

st.set_page_config(page_title="RAGAS Eval Pro: 20-Question Benchmark", layout="wide")

def load_evaluation_chain():
    embeddings, llm, vectordb, ocr_reader = get_essentials()
    if os.path.exists("fitness"):
        all_books = [os.path.join("fitness", f) for f in os.listdir("fitness") if f.endswith(".pdf")]
    else:
        all_books = []
    return get_pro_rag_chain(llm, vectordb, "Evaluation Mode", "General Fitness", all_books)

@st.cache_resource
def get_judge():
    return OllamaLLM(model="llama3.1:8b", temperature=0)

# ==========================================
# (The Ultimate 20)
# ==========================================
TEST_DATA = [
    # 
    {"question": "I want to start a gym habit but I'm struggling to stay consistent. How can I change my mindset?", "ground_truth": "Focus on identity-based habits rather than outcome-based goals. Start with the two-minute rule."},
    {"question": "What is the most important factor in a nutrition plan for fat loss?", "ground_truth": "Energy balance (calories in vs calories out) is the most critical foundation."},
    {"question": "How can I protect my spine during daily movements if I have back pain?", "ground_truth": "Maintain a neutral spine and use the 'hip hinge' pattern. Build core stiffness using the McGill Big 3."},
    {"question": "Is it better to lift heavy weights or light weights for muscle growth?", "ground_truth": "Hypertrophy can be achieved across a wide range of loading zones, provided sets are taken close to failure."},
    {"question": "My shoulder hurts when doing overhead presses. What should I check first?", "ground_truth": "Check scapular stability and rotator cuff strength. Ensure thoracic spine mobility."},
    {"question": "Can mental practice actually improve my physical performance in the gym?", "ground_truth": "Yes, visualization primes the nervous system and enhances motor skill acquisition."},
    {"question": "What is the benefit of the Tibialis Raise exercise?", "ground_truth": "It strengthens the muscle that acts as a first line of defense for the knee joint."},
    {"question": "How can I stop eating junk food after work when I'm tired?", "ground_truth": "Design your environment to make good habits easy and bad habits difficult by removing cues."},
    {"question": "What is the common cause of shin splints in runners?", "ground_truth": "Often caused by sudden increase in mileage, poor calf flexibility, or inadequate foot core strength."},
    {"question": "What should I eat immediately after a high-intensity workout for best recovery?", "ground_truth": "A combination of high-quality protein and carbohydrates to replenish glycogen."},
   
    {"question": "Explain the 'SAID' principle in strength and conditioning.", "ground_truth": "Specific Adaptation to Imposed Demands: the body adapts specifically to the type of stress/stimulus placed on it."},
    {"question": "What are the key differences between 'Functional' and 'Structural' leg length discrepancies?", "ground_truth": "Structural is an actual bone length difference; Functional is caused by soft tissue tightness or pelvic tilting."},
    {"question": "According to habit science, how do 'implementation intentions' help in sticking to a diet?", "ground_truth": "By creating an 'If-Then' plan (If I feel hungry at 4pm, Then I will eat an apple), you automate the decision-making process."}
]

# ==========================================
# 3.
# ==========================================
st.title("📊 RAGAS Evaluation Dashboard (Ultimate 20)")
st.caption("Testing 20 Scenarios | Semantic Chunking | BGE Embeddings")

if st.button("🚀 Run 20-Question Stress Test"):
    judge_llm = get_judge()
    coach_chain = load_evaluation_chain()
    
    results_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, item in enumerate(TEST_DATA):
        status_text.text(f"Inference: Processing question {i+1}/20...")
        response = coach_chain.invoke({"input": item["question"], "chat_history": []})
        results_data["question"].append(item["question"])
        results_data["answer"].append(response["answer"])
        results_data["contexts"].append([doc.page_content for doc in response["context"]])
        results_data["ground_truth"].append(item["ground_truth"])
        progress_bar.progress((i + 1) / len(TEST_DATA))

    status_text.text("Evaluating: Calculating Metrics (Sequential Protection Mode) ⚖️")
    dataset = Dataset.from_dict(results_data)
    
    eval_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge_llm,
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        run_config=RunConfig(max_workers=1, timeout=240) 
    )
    
    df_results = eval_result.to_pandas()
    st.divider()
    
    metrics_names = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
    scores = []
    for m in metrics_names:
        val = eval_result[m]
        clean_list = [x for x in val if not math.isnan(x)]
        scores.append(sum(clean_list)/len(clean_list) if clean_list else 0.0)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("### 🕸️ Performance Radar")
        fig = go.Figure(data=go.Scatterpolar(r=scores, theta=metrics_names, fill='toself', marker=dict(color='#00FFCC')))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, template="plotly_dark")
        st.plotly_chart(fig)

    with col2:
        st.write("### 📊 Metrics Summary")
        avg_score = sum(scores) / len(scores)
        st.metric("Aggregate RAG Score", f"{avg_score:.2f}")
        for m, s in zip(metrics_names, scores):
            st.progress(s, text=f"{m}: {s:.2f}")

    st.write("### 🔎 Detailed Evaluation Table")
    st.dataframe(df_results, use_container_width=True)
    status_text.success("Evaluation of 20 questions complete!")