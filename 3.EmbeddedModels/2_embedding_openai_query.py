from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file  

embedding= OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents= ["Delhi is capital of India","kolkata is capital of west bengal","mumbai is capital of maharashtra"]