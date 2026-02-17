from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful assistant that provides {domain} concise and accurate answers to user questions."),
    #SystemMessage(content="You are a helpful assistant that provides {domain} concise and accurate answers to user questions."),
    #HumanMessage(content="explain the concept of recursion in programming{topic}.")
])

prompt = chat_template.invoke({"domain": "computer science", "topic": " in the context of Python"})

print(prompt)