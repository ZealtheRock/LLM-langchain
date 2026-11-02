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

title_prompt = PromptTemplate(
     input_variables = ["topic"],
     # polutin, brain issues, desease, health, environment, climate change, global warming, pollution, sustainability, renewable energy, conservation, biodiversity, ecosystem, deforestation, carbon footprint, waste management, recycling, air quality, water pollution, soil contamination, noise pollution, plastic pollution
     template = """You are experienced speech writer.You need craft an impactful title for the given {topic}."
                    "Answer exactly only one title. """

)

speech_prompt = PromptTemplate(
     input_variables=["title"],
    template="""You need to write a impressive speech of 50 words for the following title :{title} """
)

first_chain = title_prompt | llm1| StrOutputParser() | (lambda title:(st.write(title),title)[1])
second_chain = speech_prompt | llm2
final_chain = first_chain | second_chain

st.title("Speech generator for Chat")

topic = st.text_input("Enter the topic for speech")
if topic:
    response = final_chain.invoke({"topic":topic})
    st.write(response.content)

