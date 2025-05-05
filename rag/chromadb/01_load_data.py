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
print(docs)