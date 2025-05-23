import os
from dotenv import load_dotenv, find_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI


def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=get_openai_api_key()
)

agent = create_csv_agent(
    model, 
    path='data.csv', 
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True)

agent.invoke("how many rows are there in the dataset")
# agent.invoke("count number of row by Sex")
# agent.invoke("List down the numerical and categorical columns")
# agent.invoke("Top 3 of oldest people")

# while True:
#     print("\nQuestion:")
#     question = input()
#     result = agent.invoke(question)
#     print("Answer:")
#     print(result)
    


# result = agent.run("List down the numerical and categorical columns")
# print(result)