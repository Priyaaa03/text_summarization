import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word not in stop_words and word not in string.punctuation]
        processed_sentences.append(" ".join(words))
        
    return sentences, processed_sentences

def build_similarity_matrix(sentences, embeddings):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0,0]
    
    return similarity_matrix

def rank_sentences(similarity_matrix):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    return ranked_sentences

def generate_summary(ranked_sentences, num_sentences=3):
    summary = " ".join([sentence for _, sentence in ranked_sentences[:num_sentences]])
    return summary

# Load and preprocess the text
text = """Artificial Intelligence (AI) is transforming the world in profound ways. From healthcare to finance, 
AI technologies are being integrated into various industries to improve efficiency and outcomes. 
AI's ability to analyze vast amounts of data and make predictions is driving innovation and creating new opportunities. 
However, the rise of AI also raises important ethical and societal questions. 
As AI continues to evolve, it is crucial to consider its impact on employment, privacy, and security. 
Ensuring that AI is developed and used responsibly will be key to harnessing its potential for good."""
original_sentences, processed_sentences = preprocess_text(text)

# Generate sentence embeddings
model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(processed_sentences)

# Build the similarity matrix and rank sentences
similarity_matrix = build_similarity_matrix(processed_sentences, sentence_embeddings)
ranked_sentences = rank_sentences(similarity_matrix)

# Generate the summary
summary = generate_summary(ranked_sentences)
print("Summary:")
print(summary)
