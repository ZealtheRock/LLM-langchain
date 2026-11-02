import os

import dotenv
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
llm1 = ChatOpenAI(model="gpt-4o", api_key=OPENAI_KEY)
#llm2 = ChatOllama(model="mistral")
llm2 = ChatOllama(model="mistral")

outline_prompt = PromptTemplate(
     input_variables = ["topic"],
     # polutin, brain issues, desease, health, environment, climate change, global warming, pollution, sustainability, renewable energy, conservation, biodiversity, ecosystem, deforestation, carbon footprint, waste management, recycling, air quality, water pollution, soil contamination, noise pollution, plastic pollution
     template = """You are a professional blogger. Create a outline for a blog post on the following topis: {topic}.
                The outline shoult include : 
                - Introduction
                - 3 main points with sub points
                -  Conclusion """

)

intrduction_prompt = PromptTemplate(
     input_variables=["outline"],
    template="""You are a professional blogger. Write engaging introduction paragraph based on the following blog post outline: {outline}
             the introdunction should hook the user and provide a brief overview of the topic.
             The introduction should be concise, informative, and set the tone for the rest of the blog post. """
)

first_chain = outline_prompt | llm1| StrOutputParser() | (lambda title:(st.write(title),title)[1])
second_chain = intrduction_prompt | llm2
overall_chain = first_chain | second_chain

st.title("Blog post generator...")

topic = st.text_input("Enter the topic for blog post")
if topic:
    response = overall_chain.invoke({"topic":topic})
    st.write(response.content)

