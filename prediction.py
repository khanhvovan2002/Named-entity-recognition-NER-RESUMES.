import spacy
from extract_data import *
# Load the trained NER model from disk
model_path = "final_ner_model_4"
nlp = spacy.load(model_path)
def predict_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

if __name__ == "__main__":
    pdf_directory = "/Users/vovankhanh/Documents/testing/data/pdf_file/cv_35.pdf"
     # Call the function to extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_directory)

        # Preprocess the extracted text
    preprocessed_text = preprocess_text(extracted_text)
    entities = predict_entities(preprocessed_text)
  # Print the entities
    for entity, label in entities:
        print(f"{label}: {entity}")