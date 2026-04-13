"""Actor container entrypoint: runs smolagents CodeAgent on a GAIA question.

Reads:
  /work/task_prompt.txt  -- full task prompt (question + skills + instructions)
  /work/config.json      -- {"model_id": "...", "max_steps": 25, "api_base": "..."}
  /work/*.{pdf,xlsx,...}  -- optional attached files

Writes:
  /work/output.json      -- {"final_answer": "...", "transcript": "...", "error": null}
"""

import json
import os
import traceback

from exa_py import Exa
from smolagents import CodeAgent, VisitWebpageTool, LiteLLMModel, Tool


class ExaSearchTool(Tool):
    """Web search via Exa.ai API — returns clean extracted text."""

    name = "web_search"
    description = "Search the web using Exa. Returns search results with extracted page content."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def __init__(self, num_results=5, max_characters=10000):
        super().__init__()
        self._client = Exa(api_key=os.environ.get("EXA_API_KEY"))
        self.num_results = num_results
        self.max_characters = max_characters

    def forward(self, query: str) -> str:
        results = self._client.search_and_contents(
            query,
            type="auto",
            num_results=self.num_results,
            text={"max_characters": self.max_characters},
        )
        parts = []
        for r in results.results:
            text = (r.text or "")[:self.max_characters]
            parts.append(f"[{r.title}]({r.url})\n{text}")
        return "\n\n---\n\n".join(parts) if parts else "No results found."


def main():
    # Load config
    with open("/work/config.json") as f:
        config = json.load(f)

    model_id = config.get("model_id", "gemini/gemini-2.5-flash-lite")
    max_steps = config.get("max_steps", 25)
    api_base = config.get("api_base")

    # Load the full task prompt (question + skills + instructions)
    with open("/work/task_prompt.txt") as f:
        task_prompt = f.read()

    # Set up the LLM with retry on transient errors (SSL, connection, etc.)
    def _is_transient_error(exc: BaseException) -> bool:
        err = str(exc).lower()
        return (
            "429" in err
            or "rate limit" in err
            or "too many requests" in err
            or "rate_limit" in err
            or "ssl" in err
            or "connection" in err
            or "timeout" in err
            or "bad_record_mac" in err
            or "service unavailable" in err
            or "503" in err
        )

    kwargs = {"model_id": model_id}
    if api_base:
        kwargs["api_base"] = api_base
    model = LiteLLMModel(**kwargs)
    model.retryer.retry_predicate = _is_transient_error

    # Create agent with smolagents default tools
    agent = CodeAgent(
        tools=[ExaSearchTool(), VisitWebpageTool()],
        model=model,
        max_steps=max_steps,
        additional_authorized_imports=[
            "csv", "json", "os", "sys", "io", "struct",
            "string", "textwrap", "pathlib",
            "functools", "operator", "copy",
            "hashlib", "base64", "binascii",
            "urllib", "http",
            "decimal", "fractions",
            "glob", "shutil",
            "pandas", "openpyxl",
        ],
    )

    # Run the agent
    try:
        result = agent.run(task_prompt)
        final_answer = str(result) if result is not None else None

        # Build transcript from agent logs
        transcript = _build_transcript(agent)

        output = {
            "final_answer": final_answer,
            "transcript": transcript,
            "error": None,
        }
    except Exception as e:
        output = {
            "final_answer": None,
            "transcript": traceback.format_exc(),
            "error": str(e),
        }

    with open("/work/output.json", "w") as f:
        json.dump(output, f, indent=2)


def _build_transcript(agent):
    """Extract a formatted transcript from smolagents agent memory steps."""
    lines = []
    try:
        for step in agent.memory.steps:
            # Model output (the agent's reasoning / code)
            if hasattr(step, "model_output") and step.model_output:
                output = step.model_output
                if isinstance(output, list):
                    output = str(output)
                lines.append(f"[thought] {str(output)[:2000]}")
            # Code action
            if hasattr(step, "code_action") and step.code_action:
                lines.append(f"[code] {step.code_action[:2000]}")
            # Tool calls
            if hasattr(step, "tool_calls") and step.tool_calls:
                for tc in step.tool_calls:
                    name = getattr(tc, "name", "unknown")
                    args = getattr(tc, "arguments", {})
                    lines.append(f"[tool_use: {name}] {str(args)[:500]}")
            # Observations (tool output / code execution results)
            if hasattr(step, "observations") and step.observations:
                lines.append(f"[observation] {str(step.observations)[:1000]}")
            # Errors
            if hasattr(step, "error") and step.error:
                lines.append(f"[error] {step.error}")
    except Exception as e:
        lines.append(f"[transcript_error] Could not parse agent memory: {e}")
    return "\n".join(lines) if lines else "(no transcript available)"


if __name__ == "__main__":
    main()
