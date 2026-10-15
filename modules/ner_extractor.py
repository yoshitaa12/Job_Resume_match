import spacy


# Load the pretrained English NER model
nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    """
    Extract named entities from the given text.
    """

    doc = nlp(text)

    entities = []

    for entity in doc.ents:
        entities.append({
            "text": entity.text,
            "label": entity.label_
        })

    return entities
