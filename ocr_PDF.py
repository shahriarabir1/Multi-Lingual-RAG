import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract


pdf_path = "HSC26-Bangla1st-Paper.pdf"
output_text_file = "output_text.txt"
tesseract_lang = "ben"  


print(" Converting PDF to images...")
images = convert_from_path(pdf_path, dpi=300)


print(" Running OCR on each page...")
all_text = ""
for i, image in enumerate(images):
    print(f" Processing page {i+1}...")
    text = pytesseract.image_to_string(image, lang=tesseract_lang)
    all_text += f"\n\n--- Page {i+1} ---\n\n" + text


with open(output_text_file, "w", encoding="utf-8") as f:
    f.write(all_text)

print(f"OCR completed. Output saved to: {output_text_file}")
