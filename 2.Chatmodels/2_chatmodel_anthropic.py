from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()#loading anthropic key from .env file

model= ChatAnthropic(model='claude-3-5-sonnet-20241022',temperature=0.9,max_completion_tokens=1024)#object of ChatAnthropic class
#temperature is a parameter that controls the randomness of the output. A higher temperature will result in more random output, while a lower temperature will result in more deterministic output.
#restricting the output to 1024 tokens
result= model.invoke("What is the capital of France?")
print(result.content)
