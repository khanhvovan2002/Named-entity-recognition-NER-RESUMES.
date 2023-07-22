import json

def convert_to_spacy_format(json_data):
    spacy_data = []
    for entry in json_data["annotations"]:
        text = entry[0]
        entities = entry[1]["entities"]

        # Convert entities to character-based indices
        converted_entities = []
        for entity in entities:
            start, end, label = entity
            converted_entities.append((start, end, label))

        spacy_data.append((text, {"entities": converted_entities}))

    return spacy_data

if __name__ == "__main__":
    # Load the JSON data
    with open("/Users/vovankhanh/Documents/testing/data/annotations.json", "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    # Convert to spaCy NER training format
    spacy_training_data = convert_to_spacy_format(json_data)
    print(spacy_training_data)
    # Save the converted data to a new JSON file
    with open("spacy_training_data.json", "w", encoding="utf-8") as output_file:
        json.dump(spacy_training_data, output_file, ensure_ascii=False, indent=2)
