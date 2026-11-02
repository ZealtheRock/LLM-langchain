import os
from http.client import responses

from langchain import CHATOpenAI
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

llm = CHATOpenAI(model="gpt-4o",api_ky=OPENAI_API_KEY)

prompt = input("Enter a sentence: ")
response = llm.prompt(prompt)
print(response)
