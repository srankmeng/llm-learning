from langchain.text_splitter import CharacterTextSplitter

text = "LangChain simplifies AI workflows. It enables advanced retrieval-augmented generation systems for NLP tasks."
splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator="")
chunks = splitter.split_text(text)
print(chunks)
# ====== PRINT CHUNK ======
# [
#     'LangChain simplifies AI workflows. It enables adva',
#     'ables advanced retrieval-augmented generation syst',
#     'ation systems for NLP tasks.'
# ]