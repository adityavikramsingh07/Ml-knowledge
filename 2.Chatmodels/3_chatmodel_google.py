from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()#loading google key from .env file

model= ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0.9,max_output_tokens=1024)#object of ChatGoogleGenerativeAI class
#important to use generativeAI instead of genai as the former is the latest version of google's language model
#temperature is a parameter that controls the randomness of the output. A higher temperature will result in
result=model.invoke("What is the capital of France?")

print(result.content)