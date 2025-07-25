#### SET UP Guide:
After clone git which command you have to run:

open with vs code or pycharm
python -m venv venv

For Windows Vs Code
.\venv\Scripts\activate

N:B: If open with pycharm then use slash

install all dependencies:
pip install -r requirements.txt

## For check in terminal the rag app:
python .\app.py

After loading in terminal you can see the message input area

You: ....input....
Answer: .....

type 'exit' for exit the system



#### USED TOOLS AND PACKAGES:
Libraries are used:
openai
langchain
langchain-community
langchain-openai
langchain-huggingface
faiss-cpu
python-dotenv
pytesseract
pdf2image
Pillow
fastapi==0.111.0
uvicorn==0.29.0
python-dotenv==1.0.0
sentence-transformers

Here used ChatOpenAI from langchain_community and using gpt-4 model for the LLM.
First tries many times with normal pypdf for the pdf extract but it shows bad responosee. Then used  pytesseract and tesseract for OCR extraction.
Creates Chunking and Modify the objectives with answer. First use OpenAI gpt 4 for find out the answer of the question provide and put them beside the question.
With this final txt I used it for augmented data.

Then creates the chain and finally create chat functionality with while loop and short term memory used. Long term memory use FAIS the vector DB.

Using Huggingface embeddings model and sentence transformer for vector embedding and multi lingual functionality.

Run this command:(Put your OPENAI_API_KEY)

(I have provide my open AI subscription key , Don't share it)
docker run -p 8000:8000 --env OPENAI_API_KEY=(you key) multilingual-rag-api

Then go through the api like before


## FOR use Rest API:

uvicorn main:app --reload

Then go to :(Swagger API created)
http://http://127.0.0.1:8000/docs#/

Click on the API /ask then click on "Try out"
then give input in the question:
{
"question": "What is the story about?"
}

Try Bangla or english

*** N:B: MUST Create .env file and like .envExample input your Open AI API key


## With Docker:

Run the command in cmd:
docker push shahriarabir/multilingual-rag-api:tagname


## Sample:
-- বিয়ের সময় কল্যানীর বয়স কত ছিলো?
ans: বিয়ের সময় কল্যাণীর প্রকৃত বয়স ছিল ১৫ বছর।
--এই গল্পের মূল বিষয়বস্তু কী?
ans:এই গল্পের মূল বিষয়বস্তু হল বিরহ এবং মাতৃক্তরেহের প্রভাব। গল্পে অনুপম এবং উদ্দীপকের চরিত্রে
 এই দুটি বিষয় গভীরভাবে প্রকাশ পায়।
--You: Main character of the story?
 Answer: The main character of the story "অপরিচিতা" and Anupom.



##Answer of Question:
**What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**

--Using pytesseract and tesseract for OCR extraction and using pdfplumber and manually create another model only for objectives answer then
append them beside the question. Its really hard to extract properly for bengali language. Actually it full time goes to this thing.


**What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?**
--RecursiveCharacterTextSplitter for chunking. Chunk size=1000,overlap=100, it intelligently breaks by characters, but tries to keep logical structure ( paragraphs, then sentences).

**What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**
--model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
This one is good for multi lingual application embeddings so I choose this .

**How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**
--Using FAIS for,
Fast retrieval even on large datasets

Vector-based similarity matches meaning, not just keywords

k=3 brings top 3 most relevant chunks to the QA chain

This method is ideal for scalable, semantically meaningful retrieval in RAG systems.



**How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**

---Embeddings model converts both query and document chunks into dense vectors in the same space, enabling semantic comparison.

Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?

---Better chunking, Use Bangla LLAMA 3b model offline

