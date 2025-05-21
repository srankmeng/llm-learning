from langchain.text_splitter import MarkdownTextSplitter

text = "# Header 1\nContent under header.\n\n## Header 2\nMore content here."
splitter = MarkdownTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_text(text)
print(chunks)
# ====== PRINT CHUNK ======
# [
#     '# Header 1\nContent under header.',
#     '## Header 2\nMore content here. '
# ]