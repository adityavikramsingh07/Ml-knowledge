import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load the environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
else:
    print(f"✅ Key found: {api_key[:5]}... (hidden)")
    
    # 2. Configure Google AI
    genai.configure(api_key=api_key)

    print("\n--- ASKING GOOGLE FOR AVAILABLE MODELS ---")
    try:
        # 3. List all models that support generating text
        found_any = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Model Name: {m.name}")
                found_any = True
        
        if not found_any:
            print("⚠️ No text-generation models found. Check your API key permissions.")
            
    except Exception as e:
        print(f"❌ Error connecting to Google: {e}")