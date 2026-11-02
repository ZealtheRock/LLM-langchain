from langchain_community.chat_models import ChatOllama

# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
# Make sure the LLM is running locally with the command: ollama serve, 'ollama pull gemma:2b'
llm = ChatOllama(model="gemma:2b")
# For open source LLM model you need not pass the API_KEY, but for the OpenAI model you need to pass the API_KEY.


prompt = input("Enter a question: ")
response = llm.invoke(prompt)
print(response.content)
