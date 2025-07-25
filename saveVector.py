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

def build_and_save_vectorstore():

    if os.path.exists(os.path.join(vectorstore_dir, "index.faiss")):
        print(" Vector store already exists. Skipping build.")
        return

    if not os.path.exists(input_file):
        raise FileNotFoundError(f" File not found: {input_file}")

    print(" Loading document...")
    loader = TextLoader(input_file, encoding="utf-8")
    documents = loader.load()

    print(" Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(documents)

    print(" Creating multilingual embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(" Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    print(f" Saving vector store to: {vectorstore_dir}")
    vectorstore.save_local(vectorstore_dir)
    print(" Vector store saved successfully.")

if __name__ == "__main__":
    build_and_save_vectorstore()
