import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

# 1. Load the key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key exists
if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in .env file!")
    print("Please create a .env file with: GOOGLE_API_KEY=your_key_here")
    exit(1)

# 2. Setup AI with the CORRECT model name
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # <--- Fixed this name
    google_api_key=api_key,
    temperature=0
)

# 3. Define the logic
class FeedbackType(BaseModel):
    sentiment: Literal["good", "bad"] = Field(description="Is the feedback good or bad?")

prompt = ChatPromptTemplate.from_template("Classify this feedback as 'good' or 'bad': {text}")
agent = prompt | llm.with_structured_output(FeedbackType)

# 4. Run the program
try:
    print("\n--- AGENT READY ---")
    user_text = input("Enter feedback: ")
    
    # This sends the text to Google's Brain
    result = agent.invoke({"text": user_text})
    
    # 5. Check the result and print the response
    if result.sentiment == "good":
        print("\n[RESULT]: Thank you! We have received your positive feedback.")
    else:
        print(f"\n[RESULT]: ALERT! This was bad feedback. Emailing support now about: '{user_text}'")

except Exception as e:
    print(f"\nAn error occurred: {e}")
