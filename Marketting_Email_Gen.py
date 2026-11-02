import os
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_KEY)
json_parser = JsonOutputParser()
#json_llm = ChatOllama(model="llama3.2", format="json", temperature=0)
product_prompt = PromptTemplate(
     input_variables = ["product_name", "feature"],
    template = """You are an experienced marketting specialist.
    create a achachy subject line  for a. marketting email promoting the following products: {product_name}.
    highlight the features {feature}."
                    "Respond exactly only one subject line. """

)

email_prompt = PromptTemplate(
     input_variables=["subject_line", "product_name", "target_audience"],
    template="""Write marketting email of 300 words about the product: {product_name}.
     use the subject line : {subject_line} and tailor the  email to the following audience: {target_audience}.
    format the outpu as JSON object with exactly three string keys: "Subject" , "Audience", "email"
     and fill them with resopective values."""
)

first_chain = product_prompt | llm| StrOutputParser()
# hoe to pass the an object of the jsonParser to the second chain, so that it can parse the output of the llm and return a valid json object
second_chain = email_prompt | llm | json_parser
final_chain = first_chain | (lambda subject_line:{"subject_line":subject_line, "product_name": product_name, "target_audience": target_audience}) |second_chain

st.title("Marketting email generator")

product_name = st.text_input("Enter product name")
feature = st.text_input("Provide the key features of the product (comma-seperated)")
target_audience = st.text_input("Input target audience for the email")

if product_name and feature and target_audience:
    response = final_chain.invoke({"product_name": product_name, "feature": feature})
    st.write(response)