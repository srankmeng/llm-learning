import os, json
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)
from helper import get_openai_api_key, get_llama_cloud_api_key
from IPython.display import display, HTML
from helper import extract_html_content
from llama_index.utils.workflow import draw_all_possible_flows

import nest_asyncio
nest_asyncio.apply()

llama_cloud_api_key = get_llama_cloud_api_key()
openai_api_key = get_openai_api_key()

parser = LlamaParse(
    api_key=llama_cloud_api_key,
    base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
    result_type="markdown",
    content_guideline_instruction="This is a job application form. Create a list of all the fields that need to be filled in.",
    formatting_instruction="Return a bulleted list of the fields ONLY."
)

bullet_result = parser.load_data("fake_application_form.pdf")[0]
print("===== Bulltet result =====")
print(bullet_result.text)

llm = OpenAI(model="gpt-4o-mini")
raw_json = llm.complete(
    f"""
    This is a parsed form.
    Convert it into a JSON object containing only the list 
    of fields to be filled in, in the form {{ fields: [...] }}. 
    <form>{bullet_result.text}</form>. 
    Return JSON ONLY, no markdown."""
)

# raw_json.text เป็นหน้าตาแบบนี้ '{"fields":["First Name","Last Name","Email","Phone","Linkedin","Project Portfolio","Degree","Graduation Date","Current Job Title","Current Employer","Technical Skills","Describe why you’re a good fit for this position","Do you have 5 years of experience in React?"]}'
fields = json.loads(raw_json.text)["fields"]

print("===== Without bullet result =====")
for field in fields:
    print(field)