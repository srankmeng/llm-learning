# Chunking strategies

- <https://medium.com/@anixlynch/7-chunking-strategies-for-langchain-b50dac194813#c3fa>

### LlamaIndex Examples
- `07_llama_sentence_splitter.py`: Demonstrates sentence-based chunking using LlamaIndex's `SentenceSplitter`.
- `08_llama_token_splitter.py`: Shows token-based chunking using LlamaIndex's `TokenTextSplitter`.
- `09_llama_html_parser.py`: Illustrates parsing HTML content into nodes using LlamaIndex's `HTMLNodeParser`.

## Setup

Create virtual environment:

```sh
python3 -m venv ./demo/venv
source ./demo/venv/bin/activate
```

Install dependencies:

```sh
pip install -r requirements.txt
```

Run:

```sh
python3 demo.py
```
