import os
import json
import spacy
import random
from spacy.training.example import Example
from spacy.util import minibatch, compounding

# Load the English language model for NER
nlp = spacy.blank("en")

# Add the NER component to the pipeline
ner = nlp.add_pipe("ner", name="ner")

# Function to read data from JSON files and convert it to training data
def read_json_files(json_directory):
    training_data = []
    for filename in os.listdir(json_directory):
        with open(os.path.join(json_directory, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            text = data["annotations"][0][0]
            entities = data["annotations"][0][1]["entities"]
            training_data.append((text, {"entities": [tuple(entity) for entity in entities]}))      
    return training_data
# Set up the optimizer
optimizer = nlp.begin_training()

# Provide the path to the directory with JSON files
json_directory_path = "/Users/vovankhanh/Documents/testing/data/json_file_1"

# Read JSON files and convert to training data
TRAIN_DATA = read_json_files(json_directory_path)

# Training loop
best_loss = float("inf")  # Initialize the best loss with a large value
best_model_output_path = ""  # Variable to store the path to the best model
# patience = 30  # Set the early stopping patience
epochs_without_improvement = 0  # Initialize the epochs without improvement counter

for epoch in range(200):  # You can adjust the number of epochs
    losses = {}
    # random.shuffle(TRAIN_DATA)  # Shuffle the training data
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))

    for batch in batches:
        texts, annotations = zip(*batch)
        example_objects = []
        for text, annots in zip(texts, annotations):
            example = Example.from_dict(nlp.make_doc(text), annots)
            example_objects.append(example)
        nlp.update(example_objects, drop=0.05, losses=losses)

    print("Epoch: {} Loss: {}".format(epoch, losses))

    # Check if the current loss is better than the best loss so far
    current_loss = losses["ner"]
    if current_loss < best_loss:
        # Save the model's parameters to a temporary directory
        best_loss = current_loss
        best_model_output_path = "best_ner_model"
        nlp.to_disk(best_model_output_path)
        epochs_without_improvement = 0
    else:
        # Increase the epochs without improvement counter
        epochs_without_improvement += 1

    # # Check if early stopping criteria are met (patience exceeded)
    # if epochs_without_improvement >= patience:
    #     print("Early stopping: No improvement in {} epochs.".format(patience))
    #     break

# Save the best trained NER model to disk
if best_model_output_path:
    nlp.from_disk(best_model_output_path)  # Load the best model back into the pipeline
    final_model_output_path = "final_ner_model_4"
    nlp.to_disk(final_model_output_path)
    print("Best model saved to:", final_model_output_path)
else:
    print("No model was saved as the training didn't improve.")
