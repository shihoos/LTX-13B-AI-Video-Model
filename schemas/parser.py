import json


def extract_json(
    response: str,
):
    """
    Extract and parse JSON returned by Qwen.

    Handles responses where the model accidentally
    wraps JSON inside Markdown code fences.
    """

    cleaned = response.strip()

    if cleaned.startswith("```"):

        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

    return json.loads(
        cleaned
    )
