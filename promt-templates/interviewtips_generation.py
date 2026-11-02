from gc import set_debug

from langchain_community.chat_models import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate

#from openai_demo import response

# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
# Make sure the LLM is running locally with the command: ollama serve, 'ollama pull gemma:2b'
llm = ChatOllama(model="mistral")
# For open source LLM model you need not pass the API_KEY, but for the OpenAI model you need to pass the API_KEY.
prompt = PromptTemplate(
    input_variables=["Country","no_of_paras","language"],
    template=""" You are an expert in traditional cuisine. I want you to provide a detailed description of the traditional dishes and culinary practices of {Country}. 
    Please include information about the ingredients, cooking methods, cultural significance, and any unique flavors or techniques associated with the cuisine of {Country}.
     Your response should be informative, engaging, and suitable for someone looking to learn more about the culinary traditions of {Country}.
     Refrain from giving explanation if user provides a country name which doesn't exists, just say "I dont know about the traditional cuisine of {Country}".
     Answer in a well-structured format.
      """,
)
# Setting debug mode to True to see detailed logs and debug information

st.title("Cuisine Explorer with Mistral LLM...")
Country = st.text_input("Enter the country name to explore its traditional cuisine:")
no_of_paras = st.number_input("Enter the number of paras to include in your response", min_value=1, max_value=5)
language = st.text_input("Select the language for the response:")
if Country and no_of_paras and language:
    response = llm.invoke(prompt.format(Country=Country, no_of_paras=no_of_paras, language=language))
    st.write(response.content) # to get the response only not the metadata
