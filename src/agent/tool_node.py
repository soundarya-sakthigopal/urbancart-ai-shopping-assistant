
from langgraph.prebuilt import ToolNode

from src.agent.agent_tools import URBANCART_TOOLS


# Executes tools and handles execution errors
tool_node = ToolNode(
    URBANCART_TOOLS,
    handle_tool_errors=True
)
