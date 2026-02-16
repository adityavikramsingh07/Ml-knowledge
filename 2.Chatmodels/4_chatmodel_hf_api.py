import os
# Suppress the framework warning (set to 3 to hide everything but Errors)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging
# Optional: also suppress standard python logging warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Update the task to 'conversational'
llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.1",
    task="conversational",  # <--- Change this from 'text-generation'
    max_new_tokens=512,
)

# 2. Wrap it in ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# 3. Invoke
try:
    result = model.invoke("What is the capital of France?")
    print(result.content)
except Exception as e:
    print(f"Error: {e}")