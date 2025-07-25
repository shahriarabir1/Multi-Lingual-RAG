from rag_chain import load_rag_chain
import os
from dotenv import load_dotenv

load_dotenv()

file_path = "answered_output.txt"
chain = load_rag_chain(file_path)

print("\n💬 Bengali Multilingual RAG 10 Minutws School")
print("Type 'exit' to quit.\n")

while True:
    question = input(" You: ")
    if question.strip().lower() in ['exit', 'quit']:
        break
    result = chain(question)
    print(f"🤖 Answer: {result['result']}\n")
