from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2")
json_parser = JsonOutputParser()
#json_llm = ChatOllama(model="llama3.2", format="json", temperature=0)
title_prompt = PromptTemplate(
     input_variables = ["topic"],
    template = """You are experienced speech writer.You need craft an impactful title for the given {topic}."
                    "Answer exactly only one title. """

)

speech_prompt = PromptTemplate(
     input_variables=["title", "emotion"],
    template="""Write an impressive {emotion} speech of about 350 words for this title: {title}.
    Return only a valid JSON object with exactly two string keys: "Title" and "Speech", with their respective values."""
)

first_chain = title_prompt | llm| StrOutputParser() | (lambda title:(st.write(title),title)[1])
# hoe to pass the an object of the jsonParser to the second chain, so that it can parse the output of the llm and return a valid json object
second_chain = speech_prompt | llm | json_parser
final_chain = first_chain | (lambda title:{"title":title, "emotion": emotion}) |second_chain

st.title("Speech generator for Chat")

topic = st.text_input("Enter the topic for speech")
emotion = st.text_input("Enter the Emotion related to the speech")

if topic and emotion:
    response = final_chain.invoke({"topic":topic, "emotion": emotion})
    st.write(response)