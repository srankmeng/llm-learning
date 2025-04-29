from helper import get_openai_api_key, get_llama_cloud_api_key
from IPython.display import display, HTML
from helper import extract_html_content
from llama_index.utils.workflow import draw_all_possible_flows
from llama_parse import LlamaParse
import os

import nest_asyncio
nest_asyncio.apply()

llama_cloud_api_key = get_llama_cloud_api_key()
openai_api_key = get_openai_api_key()

documents = LlamaParse(
    api_key=llama_cloud_api_key,
    base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
    result_type="markdown",
    content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
).load_data(
    "fake_resume.pdf",
)

print("===== PAGE 1 =====")
print(documents[0].text)
print("===== PAGE 2 =====")
print(documents[1].text)
print("===== PAGE 3 =====")
print(documents[2].text)