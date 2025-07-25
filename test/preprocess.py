from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

try:
  
    if not os.path.exists("output_text.txt"):
        raise FileNotFoundError("output_text.txt not found. Please run ocr_PDF.py first.")

    with open("output_text.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    if not text.strip():
        raise ValueError("The text file is empty. Please check OCR output.")
    
    print(" Text file loaded successfully")
    

    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    
    print(f"Text split into {len(docs)} chunks")

    print(" Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print(" Creating vector store...")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local("bangla_faiss_index")
    
    print(" Vector store created and saved successfully!")
    
except FileNotFoundError as e:
    print(f" File Error: {e}")
except ValueError as e:
    print(f" Value Error: {e}")
except Exception as e:
    print(f" Unexpected Error: {e}")