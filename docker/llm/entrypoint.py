"""LLM container entrypoint.

Reads system prompt from /prompts/system.txt (mounted).
Reads request data from /work/input.json.
Makes sequential Anthropic API calls.
Writes results to /work/output.json.
"""

import json
import os
import sys
import time

import anthropic


def load_system_prompt() -> str:
    path = "/prompts/system.txt"
    if not os.path.exists(path):
        print(f"ERROR: System prompt not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return f.read()


def load_input() -> dict:
    path = "/work/input.json"
    if not os.path.exists(path):
        print(f"ERROR: Input not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def make_request(client: anthropic.Anthropic, system_prompt: str, req: dict) -> dict:
    """Make a single API call with one retry on rate limit."""
    model = req.get("model", "claude-haiku-4-5-20251001")
    max_tokens = req.get("max_tokens", 4096)
    messages = req["messages"]

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            )
            return {
                "id": req["id"],
                "content": response.content[0].text if response.content else "",
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "error": None,
            }
        except anthropic.RateLimitError:
            if attempt == 0:
                print("Rate limited, waiting 60s...", file=sys.stderr)
                time.sleep(60)
            else:
                return {
                    "id": req["id"],
                    "content": "",
                    "usage": None,
                    "error": "Rate limited after retry",
                }
        except Exception as e:
            return {
                "id": req["id"],
                "content": "",
                "usage": None,
                "error": str(e),
            }


def main():
    system_prompt = load_system_prompt()
    input_data = load_input()
    requests = input_data["requests"]

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    responses = []
    for i, req in enumerate(requests):
        print(f"Processing request {i + 1}/{len(requests)}: {req['id']}", file=sys.stderr)
        result = make_request(client, system_prompt, req)
        responses.append(result)

    output = {"responses": responses}
    with open("/work/output.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Completed {len(responses)} requests", file=sys.stderr)


if __name__ == "__main__":
    main()
