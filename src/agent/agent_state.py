
from typing import TypedDict
from typing_extensions import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

"""
Why use add_messages?

It tells LangGraph how to update conversation history. Instead of replacing the existing messages whenever a new message arrives, it adds new messages while preserving the previous conversation.

This is important for multi-turn conversations.
"""
class AgentState(TypedDict, total=False):

    # Conversation history
    messages: Annotated[list[AnyMessage], add_messages]

    # Current customer request
    user_query: str

    # Agent routing decision
    route: str

    # RAG and tool outputs
    rag_result: dict
    tool_result: dict

    # Final response
    final_response: dict

    # Error handling
    error: str
