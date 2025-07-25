import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def load_vectorstore():
    """Load the FAISS vector store with error handling"""
    try:
        if not os.path.exists("bangla_faiss_index"):
            raise FileNotFoundError("FAISS index not found. Please run preprocess.py first.")
        
        print(" Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        print(" Loading FAISS index...")
        db = FAISS.load_local("bangla_faiss_index", embedding_model, allow_dangerous_deserialization=True)
        return db.as_retriever()
    
    except Exception as e:
        print(f" Error loading vector store: {e}")
        return None

def initialize_qa_chain():
    """Initialize the QA chain with error handling"""
    try:
     
        if not os.getenv("OPENAI_API_KEY"):
            print("  Warning: OPENAI_API_KEY not found in environment variables.")
            print("Please create a .env file with: OPENAI_API_KEY=your_api_key_here")
            print("Or set environment variable: set OPENAI_API_KEY=your_key_here")
            return None
  
        retriever = load_vectorstore()
        if not retriever:
            return None
        
        print(" Initializing OpenAI model...")
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        print(" QA chain initialized successfully!")
        return qa_chain
    
    except Exception as e:
        print(f" Error initializing QA chain: {e}")
        return None

def chat():
    print(" Initializing Bengali RAG Chatbot...")
    qa_chain = initialize_qa_chain()
    
    if not qa_chain:
        print(" Failed to initialize chatbot. Please check the errors above.")
        return
    
    print("\n" + "="*50)
    print("বাংলা RAG Chatbot — টাইপ করুন 'exit' অথবা 'quit' দিয়ে শেষ করুন।")
    print("="*50 + "\n")
    
    while True:
        try:
            query = input("আপনার প্রশ্ন: ").strip()
            if query.lower() in ("exit", "quit"):
                print("চ্যাট শেষ হল। ধন্যবাদ!")
                break
            
            if not query:
                print("  দয়া করে একটি প্রশ্ন লিখুন।")
                continue
            
            print(" খোঁজা হচ্ছে...")
            
            result = qa_chain.invoke({"query": query})
            
            print(f"\nউত্তর:\n{result['result']}")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nচ্যাট বন্ধ করা হল।")
            break
        except Exception as e:
            print(f" Error processing query: {e}")
            continue

if __name__ == "__main__":
    chat()