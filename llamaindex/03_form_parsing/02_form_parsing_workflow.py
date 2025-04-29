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

class ParseFormEvent(Event):
    application_form: str

class QueryEvent(Event):
    query: str
    field: str

# new!
class ResponseEvent(Event):
    response: str

class RAGWorkflow(Workflow):
    
    storage_dir = "./storage"
    llm: OpenAI
    query_engine: VectorStoreIndex

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> ParseFormEvent:

        if not ev.resume_file:
            raise ValueError("No resume file provided")

        if not ev.application_form:
            raise ValueError("No application form provided")

        # define the LLM to work with
        self.llm = OpenAI(model="gpt-4o-mini")

        # ingest the data and set up the query engine
        if os.path.exists(self.storage_dir):
            # you've already ingested the resume document
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            # parse and load the resume document
            documents = LlamaParse(
                api_key=llama_cloud_api_key,
                base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
                result_type="markdown",
                content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
            ).load_data(ev.resume_file)
            # embed and index the documents
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=OpenAIEmbedding(model_name="text-embedding-3-small")
            )
            index.storage_context.persist(persist_dir=self.storage_dir)

        # create a query engine
        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

        # you no longer need a query to be passed in, 
        # you'll be generating the queries instead 
        # let's pass the application form to a new step to parse it
        return ParseFormEvent(application_form=ev.application_form)

    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent) -> QueryEvent:
        parser = LlamaParse(
            api_key=llama_cloud_api_key,
            base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
            result_type="markdown",
            content_guideline_instruction="This is a job application form. Create a list of all the fields that need to be filled in.",
            formatting_instruction="Return a bulleted list of the fields ONLY."
        )

        # get the LLM to convert the parsed form into JSON
        result = parser.load_data(ev.application_form)[0]
        raw_json = self.llm.complete(
            f"""
            This is a parsed form. 
            Convert it into a JSON object containing only the list 
            of fields to be filled in, in the form {{ fields: [...] }}. 
            <form>{result.text}</form>. 
            Return JSON ONLY, no markdown.
            """)
        fields = json.loads(raw_json.text)["fields"]

        # new!
        # generate one query for each of the fields, and fire them off
        for field in fields:
            ctx.send_event(QueryEvent(
                field=field,
                query=f"How would you answer this question about the candidate? {field}"
            ))

        # store the number of fields so we know how many to wait for later
        await ctx.set("total_fields", len(fields))
        return

    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> ResponseEvent:
        response = self.query_engine.query(f"This is a question about the specific resume we have in our database: {ev.query}")
        return ResponseEvent(field=ev.field, response=response.response)

    # new!
    @step
    async def fill_in_application(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
        # get the total number of fields to wait for
        total_fields = await ctx.get("total_fields")

        responses = ctx.collect_events(ev, [ResponseEvent] * total_fields)
        if responses is None:
            return None # do nothing if there's nothing to do yet

        # we've got all the responses!
        responseList = "\n".join("Field: " + r.field + "\n" + "Response: " + r.response for r in responses)

        result = self.llm.complete(f"""
            You are given a list of fields in an application form and responses to
            questions about those fields from a resume. Combine the two into a list of
            fields and succinct, factual answers to fill in those fields.

            <responses>
            {responseList}
            </responses>
        """)
        return StopEvent(result=result)

async def main():
    w = RAGWorkflow(timeout=120, verbose=False)
    result = await w.run(
        resume_file="fake_resume.pdf",
        application_form="fake_application_form.pdf"
    )
    print(result)

    WORKFLOW_FILE = "workflows/form_parsing_workflow.html"
    draw_all_possible_flows(w, filename=WORKFLOW_FILE)
    html_content = extract_html_content(WORKFLOW_FILE)
    display(HTML(html_content), metadata=dict(isolated=True))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())