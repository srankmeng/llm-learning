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
from helper import get_openai_api_key, get_llama_cloud_api_key
from IPython.display import display, HTML
from helper import extract_html_content
from llama_index.utils.workflow import draw_all_possible_flows

import nest_asyncio
nest_asyncio.apply()

llama_cloud_api_key = get_llama_cloud_api_key()
openai_api_key = get_openai_api_key()

class ParseFormEvent(Event):
    application_form: str

class QueryEvent(Event):
    query: str
    field: str
    
class ResponseEvent(Event):
    response: str

# new!
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

        # define the LLM to work with
        self.llm = OpenAI(model="gpt-4o-mini")

        # ingest the data and set up the query engine
        if os.path.exists(self.storage_dir):
            # you've already ingested the resume document
            storage_context = StorageContext.from_defaults(persist_dir=
                                                           self.storage_dir)
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

        # let's pass the application form to a new step to parse it
        return ParseFormEvent(application_form=ev.application_form)

    # form parsing
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

    # generate questions
    @step
    async def generate_questions(self, ctx: Context, ev: GenerateQuestionsEvent | FeedbackEvent) -> QueryEvent:

        # get the list of fields to fill in
        fields = await ctx.get("fields_to_fill")

        # generate one query for each of the fields, and fire them off
        for field in fields:
            question = f"How would you answer this question about the candidate? <field>{field}</field>"

            # new! Is there feedback? If so, add it to the query:
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
        response = self.query_engine.query(f"This is a question about the specific resume we have in our database: {ev.query}")
        return ResponseEvent(field=ev.field, response=response.response)

  
    # Get feedback from the human
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

        # Fire off the feedback request
        return InputRequiredEvent(
            prefix="How does this look? Give me any feedback you have on any of the answers.",
            result=result
        )

    # Accept the feedback when a HumanResponseEvent fires
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
    w = RAGWorkflow(timeout=600, verbose=False)
    handler = w.run(
        resume_file="fake_resume.pdf",
        application_form="fake_application_form.pdf"
    )

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            print("We've filled in your form! Here are the results:\n")
            print(event.result)
            # now ask for input from the keyboard
            response = input(event.prefix)
            handler.ctx.send_event(
                HumanResponseEvent(
                    response=response
                )
            )

    response = await handler
    print("Agent complete! Here's your final result:")
    print(str(response))

    WORKFLOW_FILE = "workflows/feedback_workflow.html"
    draw_all_possible_flows(w, filename=WORKFLOW_FILE)
    html_content = extract_html_content(WORKFLOW_FILE)
    display(HTML(html_content), metadata=dict(isolated=True))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

