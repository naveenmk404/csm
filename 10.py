from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Define a function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize by splitting on whitespace
    tokens = text.split()
    return tokens

# Function to read text from a file
def read_text_from_file(filename):
    with open(filename, 'r') as file:
        return file.read()

# Read the content of documents from files
document1 = read_text_from_file("/content/sample_data/document1.txt")
document2 = read_text_from_file('/content/sample_data/document2.txt')

# Tokenize the documents
tokens_doc1 = preprocess_text(document1)
tokens_doc2 = preprocess_text(document2)

# Train Word2Vec model
sentences = [tokens_doc1, tokens_doc2]
word2vec_model = Word2Vec(sentences, min_count=1)

# Cosine similarity
def calculate_cosine_similarity(doc1, doc2, model):
    vec1 = np.mean([model.wv[word] for word in doc1 if word in model.wv], axis=0)
    vec2 = np.mean([model.wv[word] for word in doc2 if word in model.wv], axis=0)
    return cosine_similarity([vec1], [vec2])[0][0]

# Jaccard similarity
def calculate_jaccard_similarity(doc1, doc2):
    intersection = len(set(doc1).intersection(doc2))
    union = len(set(doc1).union(doc2))
    return intersection / union if union != 0 else 0

# Calculate similarities
cosine_sim = calculate_cosine_similarity(tokens_doc1, tokens_doc2, word2vec_model)
jaccard_sim = calculate_jaccard_similarity(tokens_doc1, tokens_doc2)

print("Cosine Similarity:", cosine_sim)
print("Jaccard Similarity:", jaccard_sim)
