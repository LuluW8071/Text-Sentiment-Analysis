import re
import subprocess
import pandas as pd
import numpy as np
import tqdm
import torch
import spacy

from torch.nn.utils.rnn import pad_sequence

# Precompile regular expressions for faster preprocessing
non_word_chars_pattern = re.compile(r"[^\w\s]|_")
whitespace_pattern = re.compile(r"\s+")
digits_pattern = re.compile(r"\d")
username_pattern = re.compile(r"@\S+")
hashtags_pattern = re.compile(r"#\S+")
html_url_pattern = re.compile(r'<.*?>|http\S+')

# Remove substrings of wrods containing fifa|world|cup|qatar|football
specific_words_pattern = re.compile(r"\b\w*(world|cup|fifa|qatar|football|ecuador|offside)\w*\b", re.IGNORECASE)

def preprocess_text(text):
    # Remove HTML tags and URLs
    text = html_url_pattern.sub('', text)
    # Lowercase text
    text = text.lower()
    # Remove specific words
    text = specific_words_pattern.sub('', text)
    text = username_pattern.sub('', text)
    text = hashtags_pattern.sub('', text)
    # Remove non-word characters
    text = non_word_chars_pattern.sub('', text)
    # Replace whitespaces with a single space
    text = whitespace_pattern.sub(' ', text)
    # Remove digits, usernames, and hashtags
    text = digits_pattern.sub('', text)

    return text.strip()

# Function to check and load the spaCy model
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        print(f"{model_name} not found. Downloading it now...")
        subprocess.run(
            ["python", "-m", "spacy", "download", model_name, "--quiet"], check=True
        )
        nlp = spacy.load(model_name)
        return nlp

def embed(docs):
    # Load spaCy model (download if needed)
    nlp = load_spacy_model("en_core_web_sm")
    docs_tensor = []
    pbar = tqdm.trange(docs.shape[0])
    for t in pbar:
        doc = nlp(docs[t])
        sentence_embeddings = [token.vector for token in doc]
        docs_tensor.append(sentence_embeddings)

    docs_tensor = [torch.tensor(np.array(d)) for d in docs_tensor]
    docs_tensor = pad_sequence(docs_tensor, batch_first=True)
    # print(docs_tensor.shape)
    return docs_tensor


