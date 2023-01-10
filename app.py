import torch
from flair.models import TARSTagger
from flair.data import Sentence

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    global model

    device = 0 if torch.cuda.is_available() else -1
    model = TARSTagger.load("tars-ner").to(device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.


def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    texts = model_inputs["texts"]
    if texts == None:
        return {'message': "No prompt provided"}

    sentences = [Sentence(document) for document in texts]
    try:
        model.predict(sentences, embedding_storage_mode="none", verbose=True)
    except StopIteration:
        print("Error while predicting")

    for sentence in sentences:
        new_spans = sentence.get_spans("ner")
        if len(new_spans) > 0:
            num_docs_with_spans += 1
        spans += new_spans
        for token in sentence:
            token.clear_embeddings()

    # Extract products and brands
    products = [span.text for span in spans if span.get_label(
        "ner").value == "product"]
    brands = [span.text for span in spans if span.get_label(
        "ner").value == "brand"]

    # Return the results as a dictionary
    return {
        "products": products,
        "brands": brands,
    }
