# Import the OpenAI embedding class from LangChain
from langchain_openai import OpenAIEmbeddings
# Import dotenv to read your API key from the .env file
from dotenv import load_dotenv
# Import the math function to calculate how 'similar' two vectors are
from sklearn.metrics.pairwise import cosine_similarity
# Import numpy for numerical array handling
import numpy as np

# Load the OPENAI_API_KEY from your .env file into the script
load_dotenv() 

# Initialize the embedding model. 
# text-embedding-3-large is OpenAI's most powerful embedding model.
# dimensions=300 shrinks the output vector size (default is 3072).
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

# Our "database" of sentences we want to search through
documents = [
    "Delhi is capital of India",
    "kolkata is capital of west bengal",
    "mumbai is capital of maharashtra"
] 

# The user's question
query = "What is the capital of India?"

# Step 1: Convert the list of documents into a list of 300-dimension numbers (vectors)
doc_embeddings = embedding.embed_documents(documents)

# Step 2: Convert the query into a single 300-dimension vector
query_embedding = embedding.embed_query(query)

# Step 3: Use Cosine Similarity to compare the query vector against all document vectors
# It returns a list of scores between 0 and 1 (1 being a perfect match)
score = cosine_similarity([query_embedding], doc_embeddings)[0]

# Step 4: Find the best match. 
# enumerate pairs the score with its index. 
# sorted() organizes them. key=lambda x:x[1] sorts by the score.
index, best_score = sorted(list(enumerate(score)), key=lambda x: x[1], reverse=True)[0]

# Print results
print(f"Query: {query}")
print(f"Top Result: {documents[index]}")
print(f"Similarity Score: {best_score}")