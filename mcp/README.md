# MCP

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

Make sure FastMCP is installed:

```sh
fastmcp version
```

## Run MCP server

```sh
python3 demo_server.py
```

or

```sh
fastmcp run demo_server.py:mcp
```

## Run client

```sh
python3 demo_client.py
```

## Run with MCP client + LLM

- Download Claude desktop app from <https://claude.ai/download>
- Go to menu settings > Developer > Edit config
- Add the example following configuration:

    ```json
    {
        "mcpServers": {
            "demo-server": {
                "command": "/Users/jaturong/Works/llm-learning/demo/venv/bin/python3",
                "args": [
                    "/Users/jaturong/Works/llm-learning/mcp/demo_server.py"
                ],
                "host": "127.0.0.1",
                "port": 8080,
                "timeout": 10000
            }
        }
    }
    ```

    > Find python3 directory with `which python3`
- Restart Claude
- Input prompt: `How about Bangkok weather?`

## MCP Server Example

- <https://github.com/modelcontextprotocol/servers>
