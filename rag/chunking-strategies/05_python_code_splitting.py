from langchain.text_splitter import PythonCodeTextSplitter

text = """def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b"""
splitter = PythonCodeTextSplitter(chunk_size=30, chunk_overlap=10)
chunks = splitter.split_text(text)
print(chunks)
# ====== PRINT CHUNK ======
# [
#     'def add(a, b):',
#     'return a + b',
#     'def subtract(a, b):',
#     'return a - b'
# ]