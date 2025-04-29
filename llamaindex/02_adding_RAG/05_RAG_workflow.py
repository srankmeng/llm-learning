from helper import get_openai_api_key, get_llama_cloud_api_key
from IPython.display import display, HTML
from helper import extract_html_content
from llama_index.utils.workflow import draw_all_possible_flows
from llama_parse import LlamaParse
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)
import os

import nest_asyncio
nest_asyncio.apply()

llama_cloud_api_key = get_llama_cloud_api_key()
openai_api_key = get_openai_api_key()

# Parse document
documents = LlamaParse(
    api_key=llama_cloud_api_key,
    base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
    result_type="markdown",
    content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
).load_data(
    "fake_resume.pdf",
)

# Create vector store
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small", api_key=openai_api_key)
)

# Query with the Index
llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key)
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
query_response = query_engine.query("What is this person's name and what was their most recent job?")

print("===== Query Response =====")
print(query_response)

# Persist index to disk
storage_dir = "./storage"
index.storage_context.persist(persist_dir=storage_dir)

if os.path.exists(storage_dir):
    # Load the index from disk
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    restored_index = load_index_from_storage(storage_context)
else:
    print("Index not found on disk.")

# Query with the restored index
restored_response = restored_index.as_query_engine().query("What is this person's name and what was their most recent job?")
print("===== Restored Index Query Response =====")
print(restored_response)

def query_resume(q: str) -> str:
    response = query_engine.query(f"This is a question about the specific resume we have in our database: {q}")
    return response.response

resume_tool = FunctionTool.from_defaults(fn=query_resume)
agent = FunctionCallingAgent.from_tools(
    tools=[resume_tool],
    llm=llm,
    verbose=True
)
agent_response = agent.chat("How many years of experience does the applicant have?")
print("===== Agent Response =====")
print(agent_response)

class QueryEvent(Event):
    query: str

class RAGWorkflow(Workflow):
    storage_dir = "./storage"
    llm: OpenAI
    query_engine: VectorStoreIndex

    # the first step will be setup
    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> QueryEvent:

        if not ev.resume_file:
            raise ValueError("No resume file provided")

        # define an LLM to work with
        self.llm = OpenAI(model="gpt-4o-mini")

        # ingest the data and set up the query engine
        if os.path.exists(self.storage_dir):
            # you've already ingested your documents
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            # parse and load your documents
            documents = LlamaParse(
                result_type="markdown",
                content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
            ).load_data(ev.resume_file)
            # embed and index the documents
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=OpenAIEmbedding(model_name="text-embedding-3-small")
            )
            index.storage_context.persist(persist_dir=self.storage_dir)

        # either way, create a query engine
        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

        # now fire off a query event to trigger the next step
        return QueryEvent(query=ev.query)

    # the second step will be to ask a question and return a result immediately
    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        response = self.query_engine.query(f"This is a question about the specific resume we have in our database: {ev.query}")
        return StopEvent(result=response.response)

async def main():
    w = RAGWorkflow(timeout=120, verbose=False)
    workflow_result = await w.run(
        resume_file="fake_resume.pdf",
        query="Where is the first place the applicant worked?"
    )
    print("===== Workflow Result =====")
    print(workflow_result)

    WORKFLOW_FILE = "workflows/rag_workflow.html"
    draw_all_possible_flows(w, filename=WORKFLOW_FILE)
    html_content = extract_html_content(WORKFLOW_FILE)
    display(HTML(html_content), metadata=dict(isolated=True))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

