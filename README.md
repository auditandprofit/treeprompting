# Tree Prompting

A command-line utility for turning a base prompt into a directed acyclic graph (DAG) of follow-up ideas using the OpenAI Responses API. The tool repeatedly asks the model to expand each node by calling a `create_children` function, incrementally building a JSON tree on disk and printing the final structure to standard output.

## Features

- Expand a plain-text prompt into a multi-level idea tree.
- Enforce structured responses via OpenAI function calling.
- Resume expansion from an existing tree JSON file.
- Limit depth and concurrency to control cost and complexity.
- Persist intermediate results safely while the tree grows.

## Installation

1. Create and activate a Python 3.11+ environment.
2. Install dependencies:

   ```bash
   pip install openai
   ```

   The project relies on the official [`openai`](https://pypi.org/project/openai/) package and requires the `OPENAI_API_KEY` environment variable to be set before use.

## Usage

```bash
python -m dag_tree PROMPT_FILE [--model MODEL] [--max-depth N] [--concurrency N]
                              [--reasoning-effort {low,medium,high}] [--service-tier TIER]
                              [--output PATH] [--resume-from TREE_JSON]
```

### Basic Example

```bash
export OPENAI_API_KEY="sk-..."
python -m dag_tree prompt.txt --max-depth 3 --concurrency 2
```

By default, the tool writes incremental results beside the prompt file (e.g., `prompt.txt.json`) and prints the completed tree to stdout.

### Resuming an Existing Tree

Use `--resume-from` with a JSON file previously produced by the tool. Expansion continues from any leaf nodes that are still shallower than `--max-depth`. When resuming, the output file defaults to the resume path.

### Controlling Parallelism and Reasoning

- `--concurrency` limits simultaneous API calls when expanding nodes.
- `--reasoning-effort` and `--service-tier` are forwarded to the OpenAI Responses API if provided.
- `--model` defaults to `gpt-4o-mini` but can be overridden.

## Output Format

Trees are stored as JSON documents with the following schema:

```json
{
  "title": "root",
  "description": "...",
  "children": [
    {
      "title": "Child node title",
      "description": "Summary for the child node",
      "children": [
        "... nested children ..."
      ]
    }
  ]
}
```

Each node contains a short `title`, a `description`, and an array of recursive `children`. The program validates data returned by the model to ensure titles and descriptions are non-empty strings.

## Development Notes

- The entry point is [`dag_tree.py`](dag_tree.py), which exposes a `main()` function and supporting helpers for parsing arguments, calling the OpenAI API, and managing the expansion queue.
- Results are saved atomically by writing to a temporary file before replacing the target path, minimizing the risk of corruption.
- Tests are not currently bundled; add your own to validate custom behavior.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) if present.
