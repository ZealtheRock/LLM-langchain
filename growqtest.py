import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client(provider):
    if provider == "huggingface":
        api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        base_url = os.getenv("HF_BASE_URL") or os.getenv("H_base_url")
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROWQ_KEY")
        base_url = os.getenv("GROQ_BASE_URL") or os.getenv("G_base_url") or "https://api.groq.com/openai/v1"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{provider}'. "
            "Set GROQ_API_KEY or G_TOKEN in your .env file."
        )

    return OpenAI(api_key=api_key, base_url=base_url)


def ask_llm(provider, model, question):
    client = get_client(provider)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    answer = ask_llm(
        provider="groq",
        model="openai/gpt-oss-20b",
        question="Where is Chicago?",
    )
    print(answer)