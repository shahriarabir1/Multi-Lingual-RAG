from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, List
from rag_chain import load_rag_chain

app = FastAPI(title="Multilingual RAG API")

# load RAG Chain
chain = load_rag_chain("answered_output.txt")

#memory
chat_memory: Dict[str, List[Dict[str, str]]] = {}

def get_client_id(request: Request) -> str:
    """Get session key (client IP or custom header)"""
    return request.client.host  


class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QueryRequest, raw_request: Request):
    client_id = get_client_id(raw_request)
    question = request.question.strip()

    if not question:
        return {"error": "Question is empty."}

    
    history = chat_memory.get(client_id, [])
    history_context = "\n".join(
        [f"User: {pair['question']}\nAI: {pair['answer']}" for pair in history[-3:]]
    )

    full_query = f"{history_context}\nUser: {question}" if history_context else question

    try:
        result = chain.invoke({"query": full_query})
        answer = result["result"]

        history.append({"question": question, "answer": answer})
        chat_memory[client_id] = history[-5:]  # Keep only last 5 turns

        return {
            "question": question,
            "answer": answer,
            "short_term_memory": history[-3:]
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/clear_memory")
def clear_memory(raw_request: Request):
    client_id = get_client_id(raw_request)
    chat_memory.pop(client_id, None)
    return {"status": f"Memory cleared for client {client_id}"}
