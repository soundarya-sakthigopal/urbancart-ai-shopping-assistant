
from typing import Literal


def route_agent(
    state: dict
) -> Literal["tools", "end"]:
    """
    Decide whether to execute tools or finish
    based on the agent's latest response.
    """

    messages = state.get("messages", [])

    if not messages:
        return "end"

    last_message = messages[-1]

    # If the LLM requests a tool, execute it
    if getattr(last_message, "tool_calls", []):
        return "tools"

    # Otherwise, finish the agent turn
    return "end"
