import os 
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv(find_dotenv())

def format_docs(docs):
    if not docs:
        return "I don't know."
    return "\n\n".join(doc.page_content for doc in docs)


# 1. Load vectorstore from chromadb from previous step
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="demo.db",
    collection_name="demo_web"
)

query = "RAG คืออะไร"

print("1. Retrieving the most similar documents...")
docs = vectorstore.similarity_search(query, k=4)
print(f"Query: {query}")
print(f"Retrieved documents: {len(docs)}")
for doc in docs:
    doc_details = doc.to_json()['kwargs']
    print("ID: ", doc_details['metadata']['id'])
    print("Source: ", doc_details['metadata']['source'])
    formatted_text = format_docs([doc])
    print("Text: ", formatted_text)
    print("================\n")