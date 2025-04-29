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
    Context,
    InputRequiredEvent,
    HumanResponseEvent
)
from IPython.display import display, HTML, DisplayHandle
from helper import extract_html_content
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.readers.whisper import WhisperReader
import gradio as gr
import asyncio
from queue import Queue

import nest_asyncio
nest_asyncio.apply()

import warnings
warnings.filterwarnings('ignore')

from helper import get_openai_api_key, get_llama_cloud_api_key

llama_cloud_api_key = get_llama_cloud_api_key()
openai_api_key = get_openai_api_key()

class ParseFormEvent(Event):
    application_form: str

class QueryEvent(Event):
    query: str

class ResponseEvent(Event):
    response: str

class FeedbackEvent(Event):
    feedback: str

class GenerateQuestionsEvent(Event):
    pass

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

        # give ourselves an LLM to work with
        self.llm = OpenAI(model="gpt-4o-mini")

        # ingest our data and set up the query engine
        if os.path.exists(self.storage_dir):
            # we've already ingested our documents
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            # we need to parse and load our documents
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

        # either way, create a query engine
        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

        # let's pass our application form to a new step where we parse it
        return ParseFormEvent(application_form=ev.application_form)

    # we've separated the form parsing from the question generation
    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent) -> GenerateQuestionsEvent:
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
            f"This is a parsed form. Convert it into a JSON object containing only the list of fields to be filled in, in the form {{ fields: [...] }}. <form>{result.text}</form>. Return JSON ONLY, no markdown.")
        fields = json.loads(raw_json.text)["fields"]

        await ctx.set("fields_to_fill", fields)

        return GenerateQuestionsEvent()

    # this step can get triggered either by GenerateQuestionsEvent or a FeedbackEvent
    @step
    async def generate_questions(self, ctx: Context, ev: GenerateQuestionsEvent | FeedbackEvent) -> QueryEvent:

        # get the list of fields to fill in
        fields = await ctx.get("fields_to_fill")

        # generate one query for each of the fields, and fire them off
        for field in fields:
            question = f"How would you answer this question about the candidate? <field>{field}</field>"

            if hasattr(ev,"feedback"):
                question += f"""
                    \nWe previously got feedback about how we answered the questions.
                    It might not be relevant to this particular field, but here it is:
                    <feedback>{ev.feedback}</feedback>
                """

            ctx.send_event(QueryEvent(
                field=field,
                query=question
            ))

        # store the number of fields so we know how many to wait for later
        await ctx.set("total_fields", len(fields))
        return

    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> ResponseEvent:
        print(f"Asking question: {ev.query}")

        response = self.query_engine.query(f"This is a question about the specific resume we have in our database: {ev.query}")

        print(f"Answer was: {str(response)}")

        return ResponseEvent(field=ev.field, response=response.response)

    # we now emit an InputRequiredEvent
    @step
    async def fill_in_application(self, ctx: Context, ev: ResponseEvent) -> InputRequiredEvent:
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

        # save the result for later
        await ctx.set("filled_form", str(result))

        # Let's get a human in the loop
        return InputRequiredEvent(
            prefix="How does this look? Give me any feedback you have on any of the answers.",
            result=result
        )

    # Accept the feedback.
    @step
    async def get_feedback(self, ctx: Context, ev: HumanResponseEvent) -> FeedbackEvent | StopEvent:

        result = self.llm.complete(f"""
            You have received some human feedback on the form-filling task you've done.
            Does everything look good, or is there more work to be done?
            <feedback>
            {ev.response}
            </feedback>
            If everything is fine, respond with just the word 'OKAY'.
            If there's any other feedback, respond with just the word 'FEEDBACK'.
        """)

        verdict = result.text.strip()

        print(f"LLM says the verdict was {verdict}")
        if (verdict == "OKAY"):
            return StopEvent(result=await ctx.get("filled_form"))
        else:
            return FeedbackEvent(feedback=ev.response)

async def main():

    WORKFLOW_FILE = "workflows/workflow.html"
    draw_all_possible_flows(RAGWorkflow, filename=WORKFLOW_FILE)
    html_content = extract_html_content(WORKFLOW_FILE)
    display(HTML(html_content), metadata=dict(isolated=True))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
