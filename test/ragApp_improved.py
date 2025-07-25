import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI 
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

def load_index_and_metadata(index_path="index.faiss", meta_path="chunks.json"):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def get_top_k_chunks(query_embedding, index, chunks, k=5):
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    return [chunks[i] for i in I[0]], D[0]


def ask_openai_with_bangla_context(question, bangla_chunks, temperature=0.2):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    chat = ChatOpenAI(temperature=temperature, model="gpt-4", api_key=api_key)

    context = "\n\n".join([f"Chunk {i+1}: {chunk['text']}" for i, chunk in enumerate(bangla_chunks)])
    
    system_prompt = """You are an educational assistant for Bengali literature. 
You will be given Bengali text chunks and a question in Bengali. 
Answer the question in Bengali based on the provided Bengali context.
If you cannot find the answer in the context, say so in Bengali."""
    
    user_prompt = f"""Context (Bengali text):\n{context}\n\nQuestion: {question}\n\nAnswer in Bengali:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = chat.invoke(messages)
    return response.content


def improved_rag_qa(user_question_bn):
    try:
        print(f"🔎 Question: {user_question_bn}")
        
        print("🔍 Embedding question and retrieving context...")
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        index, chunks = load_index_and_metadata()

        question_embedding = embedder.encode([user_question_bn])[0]
        top_chunks, distances = get_top_k_chunks(question_embedding, index, chunks, k=5)
        
        print(f"   Retrieved {len(top_chunks)} relevant chunks")
        print(f"   Similarity scores: {[f'{1/(1+d):.3f}' for d in distances[:3]]}")
        
        print("💬 Asking OpenAI...")
        bangla_answer = ask_openai_with_bangla_context(user_question_bn, top_chunks)
        
        print("\n" + "="*60)
        print("📚 ANSWER:")
        print(bangla_answer)
        print("="*60)
        
        print("\n📖 RELEVANT CONTEXT:")
        for i, chunk in enumerate(top_chunks[:3]):
            print(f"\n[{i+1}] {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}")
        
        return {
            "question": user_question_bn,
            "answer": bangla_answer,
            "context_chunks": top_chunks[:3]
        }
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None

def interactive_chat():
    print("🇧🇩 Bengali RAG System - Ask questions in Bengali!")
    print("Type 'exit' to quit.\n")
    
    while True:
        question = input("প্রশ্ন: ").strip()
        if question.lower() in ["exit", "quit", "বের হও", "বেরিয়ে যাও"]:
            print("👋 বিদায়!")
            break
        
        if question:
            result = improved_rag_qa(question)
            print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
 
    test_question = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    print("🧪 Testing with sample question:")
    improved_rag_qa(test_question)
    
    print("\n" + "="*60)
    print("Starting interactive mode...")
    interactive_chat()
