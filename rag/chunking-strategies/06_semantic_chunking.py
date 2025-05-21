from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import os 
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()
splitter = SemanticChunker(embeddings)
text = "Galaxies form part of the universe. Black holes are regions of space-time. It very fantastic."
chunks = splitter.split_text(text)
print(chunks)
# ====== PRINT CHUNK ======
# [
#     'Galaxies form part of the universe. Black holes are regions of space-time.',
#     'It very fantastic.'
# ]