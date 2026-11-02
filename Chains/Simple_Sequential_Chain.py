from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate
#ollama pull llama3.2    has to be pulled before running the code, if not already pulled
llm = ChatOllama(model="llama3.2")
title_prompt = PromptTemplate(
     input_variables = ["topic"],
    template = """You are experienced speech writer.You need craft an impactful title for the given {topic}."
                    "Answer exactly only one title. """

)

speech_prompt = PromptTemplate(
     input_variables=["title"],
    template="""You need to write a impressive speech of 350 words for the following title :{title} """
)

first_chain = title_prompt | llm| StrOutputParser() | (lambda title:(st.write(title),title)[1])
second_chain = speech_prompt | llm
final_chain = first_chain | second_chain

st.title("Speech generator for Chat")

topic = st.text_input("Enter the topic for speech")
if topic:
    response = final_chain.invoke({"topic":topic})
    st.write(response.content)

