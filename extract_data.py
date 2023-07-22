import os
import unicodedata
import re
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    # Create a PDF document object
    pdf_document = fitz.open(pdf_path)

    # Initialize an empty string to store the extracted text
    text = ""

    # Loop through each page in the PDF
    for page_num in range(pdf_document.page_count):
        # Get the page object
        page = pdf_document[page_num]

        # Extract text from the page and append it to the 'text' variable
        text += page.get_text()

    # Close the PDF document
    pdf_document.close()

    return text

def normalize_text(text):
    # Perform any normalization or preprocessing you need here
    # For example, normalizing accented characters using unicodedata.normalize
    normalized_text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

    return normalized_text



def preprocess_text(text):
    # Perform any preprocessing steps you need here
    text = text.replace("Đ", "D")  # Replace specific characters (e.g., Đ) with appropriate equivalents (e.g., D)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = " ".join(text.lower().split())  # Convert text to lowercase and remove leading/trailing spaces

    # Remove diacritics and special characters using Unicode normalization
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

    # Replace newline characters with spaces
    text = text.replace("\n", " ")

    # Handling Special Characters (excluding Gmail notation)
    # You can create a regular expression pattern to match special characters you want to keep,
    # and then replace the rest with spaces.
    special_chars_to_keep = r"[^a-zA-Z0-9@.]"  # Keep letters, numbers, '@', and '.'
    text = re.sub(special_chars_to_keep, ' ', text)

    return text


if __name__ == "__main__":
    # Provide the directory path containing the PDF files you want to extract text from
    pdf_directory = "/Users/vovankhanh/Documents/testing/data/pdf_file"

    # Get a list of all PDF files in the directory
    pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith(".pdf")]

    for pdf_file in pdf_files:
        # Construct the full path for each PDF file
        pdf_file_path = os.path.join(pdf_directory, pdf_file)

        # Call the function to extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_file_path)

        # Preprocess the extracted text
        preprocessed_text = preprocess_text(extracted_text)

        # Print the preprocessed text
        print(preprocessed_text)

        # Provide the path to the output .txt file
        output_txt_file = os.path.join('/Users/vovankhanh/Documents/testing/data/text_file', f"preprocessed_{pdf_file}.txt")

        # Write the preprocessed text to the .txt file
        with open(output_txt_file, "w", encoding="utf-8") as txt_file:
            txt_file.write(preprocessed_text)

        print("Preprocessed text has been written to", output_txt_file)

