from IPython.display import display, HTML
from helper import extract_html_content
import random
from helper import get_openai_api_key

api_key = get_openai_api_key()

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.workflow import Event
from llama_index.llms.openai import OpenAI

class FirstEvent(Event):
    first_output: str

class SecondEvent(Event):
    second_output: str
    response: str

class TextEvent(Event):
    delta: str

class ProgressEvent(Event):
    msg: str


class MyWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening"))
        return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        llm = OpenAI(model="gpt-4o-mini", api_key=api_key) 
        generator = await llm.astream_complete(
            "Please give me the first 50 words of Moby Dick, a book in the public domain."
        )
        async for response in generator:
            # Allow the workflow to stream this piece of response
            ctx.write_event_to_stream(TextEvent(delta=response.delta))
        return SecondEvent(
            second_output="Second step complete, full response attached",
            response=str(response),
        )

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step three is happening"))
        return StopEvent(result="Workflow complete.")

async def main():
    workflow = MyWorkflow(timeout=30, verbose=False)
    handler = workflow.run(first_input="Start the workflow.")

    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.msg)
        if isinstance(ev, TextEvent):
            print(ev.delta, end="")

    final_result = await handler
    print("Final result = ", final_result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
