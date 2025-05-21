# Import necessary classes
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

# Define a sample multi-sentence text string
sample_text = (
    "This is the first sentence. "
    "This is the second sentence, which is a bit longer. "
    "The third sentence is short."
)

# Instantiate SentenceSplitter
# chunk_size is the maximum size of a chunk (in tokens, but SentenceSplitter respects sentence boundaries)
# chunk_overlap is the number of tokens to overlap between chunks
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

# Create a LlamaIndex Document object from the sample text
document = Document(text=sample_text)

# Call the get_nodes_from_documents method
# This splits the document into nodes (chunks) based on sentence boundaries
nodes = splitter.get_nodes_from_documents([document])

# Iterate through the returned nodes and print their text content
print(f"Original text: '{sample_text}'")
print(f"\nNumber of nodes created: {len(nodes)}")
for i, node in enumerate(nodes):
    print(f"\nNode {i+1}:")
    print(f"Text: '{node.get_content()}'")
    # You can also access metadata like node.metadata
    # print(f"Metadata: {node.metadata}")

# Example of how chunk_size and overlap work conceptually with sentence splitting:
# If chunk_size is small, say 20 tokens, and a sentence is 30 tokens long,
# the splitter will still keep the sentence as one chunk because it prioritizes sentence integrity.
# If sentences are very short, multiple sentences might be grouped into one chunk until chunk_size is approached.
# Overlap is more straightforward, it will repeat the specified number of tokens from the end of
# the previous chunk at the beginning of the current chunk, if applicable (i.e., if it's not the first chunk).
# With the given sample text and chunk_size 1024, each sentence will likely be its own node
# as the entire text is much smaller than 1024 tokens.
# The overlap will also likely not be visible unless sentences are split across chunks,
# which won't happen here. Let's try a smaller chunk_size for demonstration of splitting.

print("\n--- Demonstrating with smaller chunk_size to see splitting (if sentences allow) ---")
splitter_small_chunk = SentenceSplitter(chunk_size=15, chunk_overlap=5) # Using token counts approximately
nodes_small_chunk = splitter_small_chunk.get_nodes_from_documents([document])

print(f"\nOriginal text: '{sample_text}'")
print(f"Number of nodes created (chunk_size=15): {len(nodes_small_chunk)}")
for i, node in enumerate(nodes_small_chunk):
    print(f"\nNode {i+1}:")
    print(f"Text: '{node.get_content()}'")
    # Expected behavior: Since SentenceSplitter respects sentence boundaries,
    # even with a small chunk_size, it will not split a sentence.
    # It will create chunks containing one or more full sentences.
    # If a single sentence exceeds chunk_size, it will be a chunk by itself.
    # The overlap will apply if a chunk boundary occurs between sentences.

# To actually see overlap, the text needs to be longer and sentences structured
# such that splitting occurs.
# For this specific example, the primary outcome is to show sentences being treated as nodes.
print("\nNote: The SentenceSplitter prioritizes keeping sentences whole.")
print("For this short example, each sentence becomes a node.")
