import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

prompt = ChatPromptTemplate(
    [
        ("system", "you are a helpful assistant. Please respond to the question asked."),
        ("user", "Question:{question}")
    ]
)

st.title("Langchain Demo with Google Gemma 2 model")
input_text = st.text_input("Something you have in mind?")


llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question" : input_text}))