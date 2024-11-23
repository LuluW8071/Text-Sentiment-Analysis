import streamlit as st
import numpy as np
import onnxruntime as ort
import re
import time
import pickle
import torch
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Precompile regular expressions for faster preprocessing
non_word_chars_pattern = re.compile(r"[^\w\s]")
whitespace_pattern = re.compile(r"\s+")
url_pattern = re.compile(r'http\S+|www\S+')
username_pattern = re.compile(r"@([^\s]+)")
hashtags_pattern = re.compile(r"#\d+")
br_pattern = re.compile(r'<br\s*/?>\s*<br\s*/?>')


# ================================================
@st.cache_resource
def load_nltk_resources():
    try:
        # Check if the required resources are already downloaded
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')

    # Return the required resources
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    return stop_words, lemmatizer


# Define ONNX model loading function
@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession("assets/model.onnx")

    with open("assets/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    return session, vocab

# Load the model, vocab and nltk resources
onnx_session, vocab = load_onnx_model()
stop_words, lemmatizer = load_nltk_resources()

# ==========================================
# Tokenization function
def preprocess_string(s):
    # Lowercase text
    s = s.lower()
    # Remove URLs
    s = url_pattern.sub('', s)
    # Remove usernames and hashtags
    s = username_pattern.sub('', s)
    s = hashtags_pattern.sub('', s)
    # Remove <br /> HTML tags
    s = br_pattern.sub('', s)
    # Remove non-word characters (preserving letters and numbers only)
    s = non_word_chars_pattern.sub(' ', s)
    # Replace multiple spaces with a single space
    s = whitespace_pattern.sub(' ', s)

    # Tokenize, remove stopwords, and lemmatize
    tokens = s.split()
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return filtered_tokens


# Padding function to ensure input sequence length
def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# ====================================================
# Inference function
def predict_sentiment(text):
    # Preprocess and tokenize the input text
    processed_tokens = preprocess_string(text)
    if len(processed_tokens) == 0:
        st.warning("Input text is too short or contains no valid tokens.")
        return None, None

    # Convert tokens to corresponding integer indices (vocab lookup)
    word_seq = np.array([vocab.get(word, 0) for word in processed_tokens])  # Default to 0 if word is not in vocab

    # Pad the sequence to the desired length
    padded_seq = padding_([word_seq], 1000)

    # Perform inference using the ONNX model
    input_tensor = padded_seq.astype(np.int64)
    output, _ = onnx_session.run(None, {"input": input_tensor})
    return np.round(output), output.item()



# ===============================
# Streamlit app UI
# ===============================
st.title("Text Sentiment Analysis")

# User input
user_input = st.text_area("Sentiment Classification", placeholder="Type your text here...", label_visibility="hidden")
if st.button("Analyze Sentiment") and user_input:
    with st.spinner("Analyzing sentiment..."):
        preds, prob = predict_sentiment(user_input)
        neg_prob = 1 - prob

        col1, col2 = st.columns([3, 1])

        # Progress bar
        with col1:
            progress_bars = [st.progress(0) for _ in range(2)]
            for i in range(101):
                progress_bars[0].progress(int(prob * 100))
                progress_bars[1].progress(int(neg_prob * 100))
        
        # Labels with prob
        with col2:
            st.write(f"Positive: {prob:.2%}")
            st.write(f"Negative: {neg_prob:.2%}")
