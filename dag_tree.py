"""Command-line tool for expanding a prompt into a DAG-like tree using the OpenAI API."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

from openai import OpenAI
from openai.types.responses import ResponseFunctionToolCall


SYSTEM_PROMPT = (
    "You expand ideas into a directed acyclic graph (tree). "
    "Always respond by calling the `create_children` function with a list of child nodes. "
    "If no children are appropriate, return an empty list."
)


@dataclass
class Node:
    """Represents a single node in the generated tree."""

    title: str
    description: str
    children: List["Node"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "children": [child.to_dict() for child in self.children],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a prompt file into a recursive DAG-like tree using OpenAI function calling."
        )
    )
    parser.add_argument(
        "prompt_file",
        type=Path,
        help="Path to the file containing the base prompt text.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum depth to expand (root depth is 0).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Optional reasoning effort to request from the model.",
    )
    parser.add_argument(
        "--service-tier",
        help="Optional service tier to request when calling the model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Path to write the tree JSON incrementally while expanding. "
            "Defaults to the prompt file name with a .json suffix."
        ),
    )
    return parser


def call_model(
    client: OpenAI,
    model: str,
    base_prompt: str,
    lineage: list[tuple[list[str], str]],
    reasoning_effort: str | None,
    service_tier: str | None,
) -> List[dict]:
    """Call the model and return the list of child node dicts."""

    prompt_message = base_prompt.strip() or "(empty prompt)"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_message},
    ]

    for idx, (node_path, description) in enumerate(lineage):
        path_str = " > ".join(node_path) if node_path else "root"
        content_lines = [
            f"Node path: {path_str}",
            f"Content: {description.strip() or '(empty description)'}",
        ]
        if idx == len(lineage) - 1:
            content_lines.append(
                "Respond by calling the `create_children` function with the children for this node."
            )
        messages.append({"role": "user", "content": "\n".join(content_lines)})

    input_messages: list[dict[str, object]] = []

    for message in messages:
        input_messages.append(
            {
                "role": message["role"],
                "type": "message",
                "content": [
                    {
                        "type": "input_text",
                        "text": message["content"],
                    }
                ],
            }
        )

    request_kwargs = {
        "model": model,
        "input": input_messages,
        "tools": [
            {
                "type": "function",
                "name": "create_children",
                "description": "Create child nodes for the current DAG tree node.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "children": {
                            "type": "array",
                            "description": "List of child nodes to attach to the parent node.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "description": "Short name for the child node.",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "One or two sentences summarizing the child node.",
                                    },
                                },
                                "required": ["title", "description"],
                            },
                        }
                    },
                    "required": ["children"],
                },
            }
        ],
        # The Responses API expects the chosen tool referenced by `name` and `type`;
        # use the `function` type (not an arbitrary `tool` type).
        "tool_choice": {"type": "function", "name": "create_children"},
    }

    if reasoning_effort is not None:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    if service_tier is not None:
        request_kwargs["service_tier"] = service_tier

    response = client.responses.create(**request_kwargs)

    tool_call: ResponseFunctionToolCall | None = None
    for item in response.output or []:
        if isinstance(item, ResponseFunctionToolCall) and item.name == "create_children":
            tool_call = item
            break

    if tool_call is None:
        raise RuntimeError("Model did not call the create_children function.")

    arguments = tool_call.arguments
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON returned from model.") from exc

    children = payload.get("children", [])
    if not isinstance(children, list):
        raise RuntimeError("Model returned a non-list `children` field.")

    validated_children = []
    for idx, child in enumerate(children, start=1):
        if not isinstance(child, dict):
            raise RuntimeError(f"Child #{idx} is not an object.")
        title = child.get("title")
        description = child.get("description")
        if not isinstance(title, str) or not title.strip():
            raise RuntimeError(f"Child #{idx} has an invalid title.")
        if not isinstance(description, str) or not description.strip():
            raise RuntimeError(f"Child #{idx} has an invalid description.")
        validated_children.append({
            "title": title.strip(),
            "description": description.strip(),
        })
    return validated_children


def expand_node(
    client: OpenAI,
    model: str,
    base_prompt: str,
    node: Node,
    path: List[str],
    lineage: list[tuple[list[str], str]],
    depth: int,
    max_depth: int,
    reasoning_effort: str | None,
    service_tier: str | None,
    save_tree: Callable[[], None],
) -> None:
    """Recursively expand a node up to the maximum depth.

    The ``save_tree`` callback is invoked after each new child is attached so the
    on-disk representation stays in sync with the in-memory tree.
    """

    if depth >= max_depth:
        return

    try:
        child_dicts = call_model(
            client,
            model,
            base_prompt,
            lineage,
            reasoning_effort,
            service_tier,
        )
    except Exception as exc:  # noqa: BLE001 - propagate clear message
        raise RuntimeError(f"Failed to expand node {' > '.join(path) or 'root'}: {exc}") from exc

    for child in child_dicts:
        child_node = Node(title=child["title"], description=child["description"])
        node.children.append(child_node)
        save_tree()
        child_path = path + [child_node.title]
        child_lineage = lineage + [(child_path, child_node.description)]
        expand_node(
            client,
            model,
            base_prompt,
            child_node,
            child_path,
            child_lineage,
            depth + 1,
            max_depth,
            reasoning_effort,
            service_tier,
            save_tree,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        prompt_text = args.prompt_file.read_text(encoding="utf-8")
    except OSError as exc:
        parser.error(f"Unable to read prompt file: {exc}")

    if args.output is not None:
        output_path = args.output
    else:
        if args.prompt_file.suffix:
            output_path = args.prompt_file.with_suffix(
                args.prompt_file.suffix + ".json"
            )
        else:
            output_path = args.prompt_file.with_name(args.prompt_file.name + ".json")

    client = OpenAI()

    root = Node(title="root", description=prompt_text.strip() or "(empty prompt)")

    def save_tree() -> None:
        payload = json.dumps(root.to_dict(), indent=2, ensure_ascii=False)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix:
                temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            else:
                temp_path = output_path.with_name(output_path.name + ".tmp")
            temp_path.write_text(payload, encoding="utf-8")
            temp_path.replace(output_path)
        except OSError as exc:
            raise RuntimeError(f"Unable to write tree to {output_path}: {exc}") from exc

    save_tree()

    expand_node(
        client=client,
        model=args.model,
        base_prompt=prompt_text,
        node=root,
        path=[root.title],
        lineage=[([root.title], root.description)],
        depth=0,
        max_depth=args.max_depth,
        reasoning_effort=args.reasoning_effort,
        service_tier=args.service_tier,
        save_tree=save_tree,
    )

    print(json.dumps(root.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
