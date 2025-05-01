import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Read text from file
text = ""
with open('data.txt', 'r') as file:
    text = file.read()

# Connect to the OpenAI API
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "assistant", "content": "You are a helpful assistant."},
        {"role": "user", "content": text}
    ]
)

# Print the token usage
usage = response.usage
print(usage)
# example response 1st time: CompletionUsage(completion_tokens=451, prompt_tokens=1156, total_tokens=1607, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0))
# example response 2nd time: CompletionUsage(completion_tokens=475, prompt_tokens=1156, total_tokens=1631, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=1024))