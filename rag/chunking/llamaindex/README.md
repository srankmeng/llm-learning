# LlamaIndex Chunking Examples

This directory contains examples demonstrating various chunking (node parsing) strategies available in LlamaIndex.

## Examples

-   `sentence_splitter.py`: Demonstrates sentence-based chunking using LlamaIndex's `SentenceSplitter`. This parser attempts to split text while respecting sentence boundaries, making it ideal for prose and general text.
-   `token_splitter.py`: Shows token-based chunking using LlamaIndex's `TokenTextSplitter`. This parser splits text into chunks of a specified number of tokens, offering fine-grained control over chunk size.
-   `html_parser.py`: Illustrates parsing HTML content into nodes using LlamaIndex's `HTMLNodeParser`. This is useful for extracting structured content from web pages, where chunks can be created based on HTML tags.

## Setup

To run these examples, ensure you have Python installed and the necessary dependencies.

1.  **Navigate to this directory**:
    ```sh
    cd rag/chunking/llamaindex
    ```

2.  **Create a virtual environment (optional but recommended)**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    Make sure you have a `requirements.txt` file in this directory with at least:
    ```
    llama-index-core
    beautifulsoup4 
    ```
    Then run:
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: The `requirements.txt` will be created in a subsequent step in the main plan).*

4.  **Run an example script**:
    ```sh
    python3 sentence_splitter.py
    ```
    or
    ```sh
    python3 token_splitter.py
    ```
    or
    ```sh
    python3 html_parser.py
    ```
