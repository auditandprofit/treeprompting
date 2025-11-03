# informationspace

informationspace is a command-line utility for enumerating latent knowledge held by large language models. It grows a branching representation of "information space" by prompting an LLM to surface breadth-first and depth-first leads, persisting the resulting structure as JSON for downstream reasoning tools. Rather than relying solely on a linear chain-of-thought, informationspace acts as an oracle that maps existing human information in a branching manner so you can explore, prioritize, and analyze expansive knowledge domains.

## Features

- Enumerate an information space from an initial prompt into a branching DAG of tacit insights.
- Persist snapshots of the information space as JSON for separate reasoning workflows and analytic tooling.
- Resume exploration from a previously saved information space JSON file to deepen specific branches.
- Tune depth, concurrency, and reasoning controls to balance exploration breadth, depth, cost, and fidelity.
- Safely write intermediate states while the information space expands.

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

By default, the tool writes incremental results beside the prompt file (e.g., `prompt.txt.json`) and prints the completed information space map to stdout.

### Resuming an Existing Information Space

Use `--resume-from` with a JSON file previously produced by the tool. Expansion continues from any leaf nodes that are still shallower than `--max-depth`. When resuming, the output file defaults to the resume path.

### Controlling Parallelism and Reasoning

- `--concurrency` limits simultaneous API calls when expanding nodes.
- `--reasoning-effort` and `--service-tier` are forwarded to the OpenAI Responses API if provided.
- `--model` defaults to `gpt-4o-mini` but can be overridden.

## Output Format

Information spaces are stored as JSON documents with the following schema:

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

Each node contains a short `title`, a `description`, and an array of recursive `children`. The program validates data returned by the model to ensure titles and descriptions are non-empty strings, making it easy to post-process and derive cross-branch insights.

## Development Notes

- The entry point is [`dag_tree.py`](dag_tree.py), which exposes a `main()` function and supporting helpers for parsing arguments, calling the OpenAI API, and managing the expansion queue.
- Results are saved atomically by writing to a temporary file before replacing the target path, minimizing the risk of corruption.
- Tests are not currently bundled; add your own to validate custom behavior.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) if present.
