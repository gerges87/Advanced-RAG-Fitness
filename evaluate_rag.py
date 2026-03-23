from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
#
from app import load_rag_chain 


judge_llm = OllamaLLM(model="llama3.1:8b", temperature=0)

def run_evaluation():
    chain = load_rag_chain()
    

    test_questions = [
        "What are the Big 3 exercises by Stuart McGill?",
        "How can I treat shoulder impingement?",
        "What is the recommended reps for squats in a rehab program?"
    ]
    

    ground_truths = [
        "The Big 3 are the modified curl-up, side bridge, and bird-dog.",
        "Treatment involves avoiding triggers, improving mobility, and strengthening rotator cuff.",
        "It depends on the phase, but often starts with 4 sets of 15 reps."
    ]

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": ground_truths
    }

    # 3. 
    print("Running RAG on test questions...")
    for q in test_questions:
        response = chain.invoke({"input": q, "chat_history": []})
        data["question"].append(q)
        data["answer"].append(response["answer"])
        # ـ Documents
        data["contexts"].append([doc.page_content for doc in response["context"]])

    #    Dataset
    dataset = Dataset.from_dict(data)

    # 4. 
    print("Evaluating metrics...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=judge_llm,
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    return result

if __name__ == "__main__":
    results = run_evaluation()
    print("\n--- RAG Evaluation Results ---")
    print(results)
    