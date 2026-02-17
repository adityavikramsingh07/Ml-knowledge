from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
#chat_template
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful assistant that provides concise and accurate answers to user questions."),
    MessagesPlaceholder(variable_name="history"),
    ('human', "explain the concept of recursion in programming.")
])

chat_hitory = []
#template
with open("chat_history.txt") as f:
    chat_hitory.extend(f.readlines())

print(chat_hitory)

#chat_prompt

prompt = chat_template.invoke({"history": chat_hitory,"query": "explain the concept of recursion in programming."})

print(prompt)