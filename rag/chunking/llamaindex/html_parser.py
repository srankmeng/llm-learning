# Import necessary classes
from llama_index.core.node_parser import HTMLNodeParser
from llama_index.core import Document

# Define a sample HTML string
sample_html = """
<html>
  <head>
    <title>Test Page</title>
  </head>
  <body>
    <h1>Main Title of The Page</h1>
    <p>This is the first paragraph. It contains some <b>bold text</b> and <i>italic text</i>.</p>
    <div class="content">
      <h2>Subtitle Here</h2>
      <p>This paragraph is nested inside a div. It's good for testing structure.</p>
      <ul>
        <li>First list item.</li>
        <li>Second list item.</li>
        <li>Third list item with a <a href="#">link</a>.</li>
      </ul>
    </div>
    <p>This is a final standalone paragraph.</p>
    <span>This is a span, which we are not explicitly parsing.</span>
  </body>
</html>
"""

# Instantiate HTMLNodeParser
# We can specify which HTML tags we want to extract content from.
# If tags are not specified, HTMLNodeParser uses a default set of common tags.
# Common tags include 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'th', 'td'
# For this example, let's explicitly define the tags we're interested in.
parser = HTMLNodeParser(tags=["p", "h1", "h2", "li"])

# Create a LlamaIndex Document object from the sample HTML string
# The HTML content is passed as the 'text' argument.
document = Document(text=sample_html)

# Call the get_nodes_from_documents method
# This method processes the HTML document and extracts content from the specified tags
# into separate nodes. Each node typically corresponds to the content within one of the specified HTML tags.
nodes = parser.get_nodes_from_documents([document])

# Iterate through the returned nodes and print their text content
print(f"Original HTML:\n{sample_html}\n")
print("--- Parsed Nodes ---")
print(f"Number of nodes created: {len(nodes)}\n")

for i, node in enumerate(nodes):
    print(f"Node {i+1}:")
    # node.get_content() retrieves the cleaned text content of the HTML element.
    print(f"  Text: '{node.get_content()}'")
    # Nodes from HTMLNodeParser also store metadata, like the tag name.
    if 'tag' in node.metadata:
        print(f"  HTML Tag: '{node.metadata['tag']}'")
    # print(f"  Metadata: {node.metadata}") # Full metadata
    print("-" * 20)

# Explanation:
# - HTMLNodeParser is designed to extract meaningful text segments from HTML documents.
# - It parses the HTML structure and creates separate nodes for content within specified (or default) HTML tags.
# - This is useful for cleaning web-scraped data, focusing on relevant textual content while discarding HTML markup.
# - Each node's metadata can include the original HTML tag from which the content was extracted.
# - Content from tags not specified (e.g., 'span' in this example, or 'title' if not in defaults/specified)
#   will be ignored unless they are children of a parsed tag and their text content is part of it.

print("\nNote: The HTMLNodeParser creates nodes based on specified or default HTML tags.")
print("Content from other tags (like 'span' or 'title' in this example, unless specified) is generally not created as separate nodes.")
print("The parser extracts the textual content within these tags.")
