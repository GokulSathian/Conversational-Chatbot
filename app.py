from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI assistant help me with the following queries"),
        ("user","Question:{question}" )
    ]
)

st.title("Demo")
input_text = st.text_input("Type here")



llm = ChatOpenAI(model= "gpt-4o-mini")
output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))