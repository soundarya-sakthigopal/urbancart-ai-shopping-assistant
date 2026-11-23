
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.agent_state import AgentState
from src.agent.agent_node import agent_node
from src.agent.tool_node import tool_node
from src.agent.agent_router import route_agent


# In-memory conversation checkpoint storage
memory = MemorySaver()


def build_agent_graph():
    """
    Build the UrbanCart LangGraph agent workflow
    with conversation memory.
    """

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "end": END
        }
    )

    workflow.add_edge("tools", "agent")

    # Compile graph with conversation memory
    agent_graph = workflow.compile(
        checkpointer=memory
    )

    return agent_graph
