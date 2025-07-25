

# load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


input_file = "answered_output.txt"
vectorstore_dir = "vectorstore_index"
chunk_size = 1000
chunk_overlap = 100

def load_rag_chain(filepath: str):
    try:
   
        index_path = "vectorstore_index"
        
       
        if os.path.exists(index_path):
            print(" Loading existing FAISS vector store...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        else:
           
            print(f"Loading document from: {filepath}")
            loader = TextLoader(filepath, encoding="utf-8")
            documents = loader.load()
            
            print(" Splitting...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = splitter.split_documents(documents)

            print(" Creating embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            print(" Creating vector store...")
            vectorstore = FAISS.from_documents(split_docs, embeddings)

            # Save vectorstore
            vectorstore.save_local(index_path)
            print(" Vector store saved to disk.")

        #  retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

 
        print(" Initializing LLM...")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=500)

        #  chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        print(" RAG chain ready (cached or built)")
        return qa_chain

    except Exception as e:
        print(f" Error: {e}")
        return None


# Test function
def test_rag_chain(filepath: str, test_question: str = "What is this document about?"):
    """Test the RAG chain with a sample question"""
    try:
        chain = load_rag_chain(filepath)
        if chain:
            print(f" Testing with question: '{test_question}'")
            result = chain.invoke({"query": test_question})
            print(f" Test Answer: {result['result']}")
            return True
        return False
    except Exception as e:
        print(f" Test failed: {e}")
        return False

if __name__ == "__main__":

    test_filepath = "answered_output.txt"
    if os.path.exists(test_filepath):
        test_rag_chain(test_filepath)
    else:
        print(f"Test file not found: {test_filepath}")