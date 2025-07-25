import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()


test_cases = [
    ("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "শুম্ভুনাথ"),
    ("কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "মামাকে"),
    ("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "১৫ বছর")
]

chunk_sizes = [150, 250]
chunk_overlaps = [30, 50]
search_ks = [3, 5]
chain_types = ["stuff", "map_reduce"]

with open("output_text.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

def build_vectorstore(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(docs, embed_model)
    return vectorstore

def evaluate_rag(vectorstore, search_k, chain_type):
    retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type=chain_type
    )
    
    correct = 0
    for q, expected in test_cases:
        result = qa_chain.invoke({"query": q})
        answer = result['result'].strip()
        if expected in answer:
            correct += 1
        print(f"Q: {q}")
        print(f"Expected: {expected}")
        print(f"Got     : {answer}\n")

    return correct

results = []

for chunk_size in chunk_sizes:
    for chunk_overlap in chunk_overlaps:
        for search_k in search_ks:
            for chain_type in chain_types:
                print(f"🔧 Testing: chunk_size={chunk_size}, overlap={chunk_overlap}, k={search_k}, chain={chain_type}")
                vs = build_vectorstore(full_text, chunk_size, chunk_overlap)
                correct = evaluate_rag(vs, search_k, chain_type)
                results.append(((chunk_size, chunk_overlap, search_k, chain_type), correct))
                print("="*60 + "\n")


results.sort(key=lambda x: x[1], reverse=True)
print("🏆 Best Configurations:")
for config, score in results:
    print(f"✅ {config} => {score}/3 correct")
