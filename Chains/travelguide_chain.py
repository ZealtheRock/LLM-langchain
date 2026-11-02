
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate

# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
# Make sure the LLM is running locally with the command: ollama serve, 'oll
llm = ChatOllama(model="llama3.2", temperature=0.7)
# For open source LLM model you need not pass the API_KEY

prompt = PromptTemplate(
    input_variables=["City", "month","budget","language"]
    , template=""" Welcome to the city {City} travel guide. If you are visiting in the month of {month}, here's whatr you can explore.
        1- Must visit attractions and landmarks in {City} during {month}.
        2- Recommended local cuisine and dining experiences.
        3- Useful phrases in the {language}.
        5- {budget}-friendly options for accommodation
        """)
#------tool, google,

if st.title("Generate Travel Guide"):
    City = st.text_input("Enter the city name to explore its travel guide:")
    month = st.text_input("Enter the month of visit:")
    budget = st.selectbox("Enter your budget", ['Low', 'Medium','High'])
    language = st.text_input("Select the language for the response:")

    chain = prompt |llm

    if City and month and budget and language:
        # Instead of llm.invoke,
        # for chain we need to chain.invoke, which will accept the dictionary rather than prompt, as below
        response = llm.invoke({"City":City, "month":month, "budget":budget, "language":language})
        st.write(response.content) # to get the response only not the metadata