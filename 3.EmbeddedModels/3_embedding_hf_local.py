import os
import logging
# Suppress the framework warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_huggingface import HuggingFaceEmbeddings

# Use a dedicated embedding model (this is small and fast)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is capital of India"

# Generate the vector
vector = embedding.embed_query(text)

# Print the first 5 numbers of the vector and its total length
print(f"Vector Length: {len(vector)}")
print(f"First 5 values: {vector[:5]}")