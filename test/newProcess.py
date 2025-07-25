import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
import pdfplumber


load_dotenv()


def extract_text_from_pdf(file_path):
    """Extract text from PDF with error handling"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        text = ""
        print(f" Opening PDF: {file_path}")
        
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f" Found {total_pages} pages")
            
            for i, page in enumerate(pdf.pages, 1):
                print(f" Processing page {i}/{total_pages}")
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
            
        print(f" Extracted {len(text)} characters from PDF")
        return text
        
    except Exception as e:
        print(f" Error extracting text from PDF: {e}")
        return None

def get_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Use the following document text to answer the question.
If the answer is not in the text, say "Answer not found in the document."

Document:
\"\"\"{context}\"\"\"

Question:
{question}

Answer in the same language as the question (Bengali or English).
"""
    )
def split_text_into_chunks(text, max_tokens=2000):
   
    chunks = []
    current_chunk = ""
    for line in text.split('\n'):
        if len(current_chunk) + len(line) < max_tokens:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def ask_question_with_langchain_llm(context, question, model_name="gpt-4"):
    """Ask question using OpenAI with error handling"""
    try:

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        print(" Initializing ChatOpenAI...")
        llm = ChatOpenAI(
            model=model_name, 
            temperature=0.2, 
            openai_api_key=openai_api_key
        )
        
        print(" Formatting prompt...")
        prompt = get_prompt()
        formatted_prompt = prompt.format(context=context, question=question)
        
        print(" Sending request to OpenAI...")

        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        
        return response.content.strip()
        
    except Exception as e:
        print(f" Error asking question: {e}")
        return None
def ask_question_from_chunks(chunks, question, model_name="gpt-4"):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        openai_api_key=openai_api_key
    )

    prompt = get_prompt()
    best_answer = None

    for i, chunk in enumerate(chunks):
        print(f" Searching in chunk {i+1}/{len(chunks)}...")
        formatted_prompt = prompt.format(context=chunk, question=question)
        try:
            response = llm.invoke([HumanMessage(content=formatted_prompt)])
            answer = response.content.strip()
            if "Answer not found" not in answer:
                best_answer = answer
                break 
        except Exception as e:
            print(f" Chunk {i+1} failed: {e}")
    
    return best_answer or "Answer not found in document."


if __name__ == "__main__":
    try:
       
        pdf_path = "HSC26-Bangla1st-Paper.pdf"
     
        if not os.path.exists(pdf_path):
            print(f" PDF file not found: {pdf_path}")
            print("Please make sure the PDF file is in the same directory as this script.")
            exit(1)
        
        question = input(" Enter your question (Bengali or English): ").strip()
        
        if not question:
            print(" Please enter a valid question.")
            exit(1)

        print(" Reading PDF...")
        document_text = extract_text_from_pdf(pdf_path)
        
        if not document_text:
            print(" Failed to extract text from PDF.")
            exit(1)
        chunks = split_text_into_chunks(document_text, max_tokens=2000)
        print(" Asking question using LangChain + OpenAI...")
        answer = ask_question_from_chunks(chunks, question)
        
        if answer:
            print(" Answer:")
           
            print(answer)
       
        else:
            print(" Failed to get answer from OpenAI.")
            
    except KeyboardInterrupt:
        print(" Process interrupted by user.")
    except Exception as e:
        print(f" Unexpected error: {e}")