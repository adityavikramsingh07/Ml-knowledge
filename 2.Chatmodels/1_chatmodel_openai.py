from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()#loading openai key from .env file

model= ChatOpenAI(model='gpt-4',temperature=0.9,max_completion_tokens=1024)#object of ChatOpenAI class
#temperature is a parameter that controls the randomness of the output. A higher temperature will result in more random output, while a lower temperature will result in more deterministic output.
#restricting the output to 1024 tokens
result= model.invoke("What is the capital of France?")

print(result)