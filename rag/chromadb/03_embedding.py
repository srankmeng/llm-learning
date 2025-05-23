import os 
from dotenv import load_dotenv, find_dotenv
os.environ['USER_AGENT'] = 'demoagent'

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv(find_dotenv())

# 1. Load data from web
loader = WebBaseLoader(
    web_paths=(
        "https://aws.amazon.com/th/what-is/retrieval-augmented-generation/",
    ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_="eb-faq-content")
    ),
)
docs = loader.load()

# 2. Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print("Size of chunk:", splits.__len__())
for idx, text in enumerate(splits):
    text.metadata["id"] = idx

# 3. Save vectorstore to disk (embeding)
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings(),
    persist_directory="demo.db",
    collection_name="demo_web"
)

print("Vectorstore saved to disk")