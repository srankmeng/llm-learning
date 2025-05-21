# Import necessary classes
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document

# Define a sample text string
sample_text = (
    "This is a test string to demonstrate token-based splitting. "
    "It has several words and characters, allowing us to see how "
    "TokenTextSplitter works with chunk_size and chunk_overlap."
)

# Instantiate TokenTextSplitter
# chunk_size is the maximum number of tokens in a chunk.
# chunk_overlap is the number of tokens to overlap between chunks.
# The tokenizer by default is tiktoken (used by OpenAI models).
splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=2)

# Create a LlamaIndex Document object from the sample text
document = Document(text=sample_text)

# Call the get_nodes_from_documents method
# This splits the document into nodes (chunks) based on token counts.
nodes = splitter.get_nodes_from_documents([document])

# Iterate through the returned nodes and print their text content
print(f"Original text: '{sample_text}'")
print(f"\nUsing TokenTextSplitter with chunk_size=10 and chunk_overlap=2:")
print(f"\nNumber of nodes created: {len(nodes)}")
for i, node in enumerate(nodes):
    print(f"\nNode {i+1}:")
    # node.get_content() retrieves the text of the chunk.
    # Note: The exact tokenization depends on the underlying tokenizer (default is tiktoken).
    # So, the output text might not always align perfectly with word boundaries if a word's tokens
    # cause it to exceed the chunk_size.
    print(f"Text: '{node.get_content()}'")
    # print(f"Metadata: {node.metadata}") # Metadata might contain information like window, etc.

# Further explanation of token-based splitting:
# - TokenTextSplitter divides text based on a specified number of tokens.
# - Unlike SentenceSplitter, it does not necessarily respect sentence or word boundaries.
# - 'chunk_size' defines the target size for each text chunk in tokens.
# - 'chunk_overlap' specifies how many tokens from the end of the previous chunk
#   should be repeated at the beginning of the current chunk. This helps maintain
#   context between chunks.
# - The default tokenizer is `tiktoken.get_encoding("gpt-3.5-turbo")`.
#   Different models might use different tokenizers, which can affect how text is split.

# Example: If sample_text is "abcdefghijklmnopqrstuvwxyz" and chunk_size=5, overlap=1
# Chunk 1 might be "abcde"
# Chunk 2 might be "efghi" (starting with 'e' due to overlap)
# (Actual tokenization is more complex than character counting)

print(
    "\nNote: TokenTextSplitter splits text based on token counts, "
    "not necessarily word or sentence boundaries."
)
print("The overlap ensures some continuity between chunks.")
