import os
from http.client import responses
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)

# This prompt template is designed to provide a system message that sets the context for the LLM,
#  instructing it to act as an Agile coach. The user input will be inserted into the prompt where "{input}" is specified, 
# allowing the model to respond appropriately to user queries related to Agile and Scrum.
# without maintaining the context of the conversation, the model may not provide accurate or relevant answers. 
# The system message helps to establish the role and expertise of the model, guiding its responses to be more aligned with 
# Agile coaching.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a Agile coach. Answer any question related to Agile"),
    ("user", "{input}") 
    #  this is where the user input will be inserted into the prompt
])

st.title("Agile Coach")

input_txt = st.text_input("Enter your question related to Agile and Scrum:")
chain = prompt_template | llm
if input_txt:
    response = chain.invoke({"input": input_txt})
    st.write(response.content)
