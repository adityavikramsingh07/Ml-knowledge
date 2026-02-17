import os
import google.generativeai as genai

# ------------------------------------------------------------------
# PASTE YOUR GOOGLE KEY HERE (Remove the 'sk-' prefix!)
# ------------------------------------------------------------------
GOOGLE_API_KEY = "AIzaSyD8StMk4Bq0WGQHVKtDdCiK4FNywcFscmE"
api_key = GOOGLE_API_KEY

try:
    print("Authenticating with Google...")
    
    # Configure the library with your key
    genai.configure(api_key=GOOGLE_API_KEY)

    print("\n✅ API Key is valid! Here are the Gemini models you can use:\n")
    
    # Google's method to list models
    for m in genai.list_models():
        # We only want models that can generate text (chat models), not embeddings
        if 'generateContent' in m.supported_generation_methods:
            # The name usually comes as "models/gemini-pro", we print it simply
            print(f" - {m.name}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure you removed 'sk-' from the start of the key.")
    print("2. Make sure you are connected to the internet.")