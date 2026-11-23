from langchain_core.messages import HumanMessage

from src.agent.agent_graph import build_agent_graph


# Build the graph once and reuse it
agent_graph = build_agent_graph()


def invoke_agent(message: str, thread_id: str) -> str:
    """
    Invoke the UrbanCart LangGraph agent
    and return its final response.
    """

    result = agent_graph.invoke(
        {
            "messages": [
                HumanMessage(content=message)
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Extract the latest assistant response
    messages = result.get("messages", [])

    if not messages:
        raise ValueError("Agent returned no messages")

    return messages[-1].content