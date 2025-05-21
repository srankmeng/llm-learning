from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """LangChain supports modular pipelines for AI workflows.  
These workflows include document loading, chunking, retrieval, and LLM integration.  
LangChain simplifies AI model deployment."""
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_text(text)
print(chunks)
# ====== PRINT CHUNK ======
# [
#     'LangChain supports modular pipelines for AI',
#     'for AI workflows.',
#     'These workflows include document loading,',
#     'loading, chunking, retrieval, and LLM',
#     'and LLM integration.',
#     'LangChain simplifies AI model deployment.'
# ]