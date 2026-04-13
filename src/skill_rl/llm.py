"""LLM helper: single-turn completions via litellm."""

import litellm


def complete(
    user_content: str,
    system_prompt: str,
    model_id: str,
    api_base: str | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> str:
    """Single-turn LLM completion via litellm.

    Returns the assistant's response text.
    Raises RuntimeError on failure.
    """
    kwargs = dict(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if api_base:
        kwargs["api_base"] = api_base
    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty response")
    return content
