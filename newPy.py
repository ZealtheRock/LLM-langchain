import anthropic
import os
client = anthropic.Anthropic(
    api_key="sk-SEyBSU_bhp_FIUWJM4UImQ",
    base_url="https://llmgw-wp.tekstac.com"
)

message = client.messages.create(
    model="global.anthropic.claude-sonnet-4-6",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "Explain Docker networking, only in five pointers"
        }
    ]
)

print(message.content[0].text)