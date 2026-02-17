from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = [
    SystemMessage(content="You are a helpful assistant that provides concise and accurate answers to user questions."),
    HumanMessage(content="What is the capital of France?")
]

messages.append(AIMessage(content=model.invoke(messages).content))

print(messages)
