from gc import set_debug

from langchain_community.chat_models import ChatOllama

# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
# Make sure the LLM is running locally with the command: ollama serve, 'ollama pull gemma:2b'
llm = ChatOllama(model="mistral")
# For open source LLM model you need not pass the API_KEY, but for the OpenAI model you need to pass the API_KEY.

# Setting debug mode to True to see detailed logs and debug information
set_debug(True)

prompt = input("Enter a question: ")
response = llm.invoke(prompt)
print(response.content) # to get the response only not the metadata
