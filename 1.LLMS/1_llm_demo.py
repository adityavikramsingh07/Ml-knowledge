from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()#loading openai key from .env file

llm = OpenAI(model="gpt-3.5-turbo-instruct")#object of OpenAI class

result=llm.invoke("What is the capital of France?")#calling the invoke method to get the answer of the question

print(result)