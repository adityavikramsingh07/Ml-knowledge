import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.header("Research Tool")

paper_input=st.selectbox( "Select reseacrh paper name", ["select...", "attention is all you need", "BERT: pretraning of the deep bidirectional transformer", "GPT-3: language models are few-shot learners","Diffusion models beat gans on image synthesis"])

style_input=st.selectbox( "Select EXPPLANATION STYLE", ["begineer-friendly", "technical", "core-mathematical"])

length_input=st.selectbox( "Select EXPPLANATION LENGTH", ["short(1-2 paragraphs)", "medium(3-5 paragraphs)", "long(detailed explanation)"])
 
#template
template = load_prompt('template.json')
#fill the placeholders in the template

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({"paper_input": paper_input, "style_input": style_input, "length_input": length_input})
    st.write(result.content)