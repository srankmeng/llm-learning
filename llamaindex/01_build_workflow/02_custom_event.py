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

class FirstEvent(Event):
    first_output: str

class SecondEvent(Event):
    second_output: str

class LoopEvent(Event):
    loop_output: str

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> FirstEvent:
        print(ev.first_input)
        return FirstEvent(first_output="First step complete.")

    async def step_one(self, ev: StartEvent | LoopEvent) -> FirstEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="Second step complete.")

    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result="Workflow complete.")

async def main():
    workflow = MyWorkflow(timeout=10, verbose=False)
    result = await workflow.run(first_input="Start the workflow.")
    print(result)

    WORKFLOW_FILE = "workflows/custom_events.html"
    draw_all_possible_flows(workflow, filename=WORKFLOW_FILE)
    html_content = extract_html_content(WORKFLOW_FILE)
    display(HTML(html_content), metadata=dict(isolated=True))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
