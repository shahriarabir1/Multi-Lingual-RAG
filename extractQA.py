import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()  


def extract_questions(text):
    """Extract questions ending with '?' from the given text."""
    lines = text.splitlines()
    questions = []
    for line in lines:
        line = line.strip()
        if len(line) > 5 and line.endswith('?'):
            questions.append(line)
    return questions

def get_answer(question):
    """Get a concise answer from GPT-4 for a given question."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Answer this Bengali question concisely: {question}"}]
    )
    return response.choices[0].message.content.strip()

def append_answers_to_text(text):
    """
    Find all question lines in the text and append the answer as a new line after the question.
    """
    lines = text.splitlines()
    updated_lines = []

    for line in lines:
        stripped = line.strip()
        updated_lines.append(line)
        if len(stripped) > 5 and stripped.endswith('?'):
            try:
                answer = get_answer(stripped)
                updated_lines.append(f"Answer: {answer}")
            except Exception as e:
                updated_lines.append("Answer: [Error fetching answer]")

    return '\n'.join(updated_lines)



input_file = "output_text.txt"
output_file = "answered_output.txt"

if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ Input file '{input_file}' not found.")

with open(input_file, "r", encoding="utf-8") as f:
    all_text = f.read()



print(" Processing questions and generating answers...")
final_text = append_answers_to_text(all_text)



with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f" Questions with answers saved to {output_file}")
