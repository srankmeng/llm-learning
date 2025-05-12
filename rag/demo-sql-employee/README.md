# Chunking RAG

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

1. Initialize data in sqlite

    ```sh
    python3 01_init_data.py
    ```

2. Convert data to vector and store to chromadb (embedding)

    ```sh
    python3 02_embeding.py
    ```

3. ลอง search ข้อมูลใน chromadb

    ```sh
    python3 03_search.py
    ```

## Run test (evaluation)

```sh
deepeval test run test_correctness.py
```
