import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate

# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
# Make sure the LLM is running locally with the command: ollama serve, 'oll
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
# For open source LLM model you need not pass the API_KEY

prompt = PromptTemplate(
    input_variables=["City", "month","budget","language"]
    , template=""" Welcome to the city {City} travel guide. If you are visitng in the month of {month}, here's whatr you can explore.
        1- Must visit attractions and landmarks in {City} during {month}.
        2- Recommended local cuisine and dining experiences.
        3- Useful phrases in the {language}.
        5- {budget}-friendly options for accommodation
        """)

