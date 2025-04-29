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

class BranchA1Event(Event):
    payload: str

class BranchA2Event(Event):
    payload: str

class BranchB1Event(Event):
    payload: str

class BranchB2Event(Event):
    payload: str

class BranchWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> BranchA1Event | BranchB1Event:
        if random.randint(0, 1) == 0:
            print("Go to branch A")
            return BranchA1Event(payload="Branch A")
        else:
            print("Go to branch B")
            return BranchB1Event(payload="Branch B")

    @step
    async def step_a1(self, ev: BranchA1Event) -> BranchA2Event:
        print(ev.payload)
        return BranchA2Event(payload=ev.payload)

    @step
    async def step_b1(self, ev: BranchB1Event) -> BranchB2Event:
        print(ev.payload)
        return BranchB2Event(payload=ev.payload)

    @step
    async def step_a2(self, ev: BranchA2Event) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Branch A complete.")

    @step
    async def step_b2(self, ev: BranchB2Event) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Branch B complete.")

async def main():
    WORKFLOW_FILE = "workflows/branching.html"
    draw_all_possible_flows(BranchWorkflow, filename=WORKFLOW_FILE)
    html_content = extract_html_content(WORKFLOW_FILE)
    display(HTML(html_content), metadata=dict(isolated=True))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
