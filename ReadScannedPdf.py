import ollama
from pdf2image import convert_from_path
import base64
from io import BytesIO
import os

#Convert PDF to images
def pdf_to_images(pdf_path):
    """This method converts pdf files to images"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    pages = convert_from_path(pdf_path, 300)
    image = pages[0] # Convert the first page to an image
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def read_image(image_str):
    """This method reads the image and returns the text"""
    response = ollama.chat(
        model = "qwen3-vl:32b",
        messages= [
            
            {"role": "system", "content": "You are an helpful assistant that extracts text from images"},
            {
                "role": "user", 
                "content": "Extract all text and table data from the image. Please keep the original formatting. The image file is in base64 format and is provided in the 'images' field. Please return the extracted text in markdown format.",
                "images": [image_str]
            }
        ]
    )
    print(response['message']['content'])


pdf_path = "2025-26-westchester-rgb-explanatory-statement.pdf"
read_image(pdf_to_images(pdf_path)) 
    
