# RAG with ChromaDB

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

## Step Run

1. Load data from web

    ```sh
    python3 01_load_data.py
    ```

2. Chunking data

    ```sh
    python3 02_chunking.py
    ```

3. Store data to chromadb (embedding)

    ```sh
    python3 03_embedding.py
    ```

4. Retrieve and generate response

    ```sh
    python3 04_retriever_normal.py
    ```

5. ลอง search ข้อมูลใน chromadb

    ```sh
    python3 05_retrieve_search.py
    ```

6. **Improvement**: re-ranking

    - with cohere (create API key before: <https://dashboard.cohere.com/api-keys>)

        ```sh
        python3 06_reranking_cohere.py
        ```

    - **OR** with flashrank

        ```sh
        python3 06_reranking_flashrank.py
        ```

7. **Improvement**: Retrieve and generate response with re-ranking data

    ```sh
    python3 07_retriever_reranking.py
    ```

## Run test (evaluation)

```sh
deepeval test run test_correctness.py
```
