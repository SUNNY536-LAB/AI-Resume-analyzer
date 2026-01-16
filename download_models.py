from sentence_transformers import SentenceTransformer
import spacy
from spacy.cli import download

print("Downloading Sentence Transformer Model...")
SentenceTransformer('all-mpnet-base-v2')
print("✅ Sentence Transformer Downloaded.")

print("Downloading Spacy Model...")
try:
    spacy.load("en_core_web_sm")
except:
    download("en_core_web_sm")
print("✅ Spacy Model Downloaded.")
