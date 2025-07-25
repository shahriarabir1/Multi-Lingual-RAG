import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI 
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv()

def improved_search_test():
    print("🔍 Testing improved search for 'সুপুরুষ' question...")
    
   
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
   
    relevant_chunks = []
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        if 'সুপুরুষ' in text or ('অনুপম' in text and 'মামা' in text):
            relevant_chunks.append((i, chunk))
    
    print(f"Found {len(relevant_chunks)} relevant chunks")
    

    for i, (idx, chunk) in enumerate(relevant_chunks[:5]):
        print(f"\n--- Chunk {i+1} (Index: {idx}) ---")
        print(f"Text: {chunk['text'][:300]}...")
        if 'সুপুরুষ' in chunk['text']:
            print("🎯 Contains 'সুপুরুষ'!")
    
    return relevant_chunks

if __name__ == "__main__":
    improved_search_test()
