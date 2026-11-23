
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode


@tool
def simulate_tool_failure():
    """Simulate a database connection failure."""
    raise RuntimeError("Database connection failed")


def mock_agent_node(state: MessagesState):

    last_message = state["messages"][-1]

    # If the last message is a tool error,
    # return a customer-friendly response.
    if getattr(last_message, "status", None) == "error":

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I'm unable to retrieve this information "
                        "right now. Please try again later."
                    )
                )
            ]
        }

    # Otherwise, request the tool.
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simulate_tool_failure",
                        "args": {},
                        "id": "recovery_test_001",
                        "type": "tool_call"
                    }
                ]
            )
        ]
    }


def route_after_agent(state: MessagesState):

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", []):
        return "tools"

    return "end"


def test_agent_error_recovery():

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", mock_agent_node)

    workflow.add_node(
        "tools",
        ToolNode(
            [simulate_tool_failure],
            handle_tool_errors=True
        )
    )

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "end": END
        }
    )

    workflow.add_edge("tools", "agent")

    graph = workflow.compile()

    result = graph.invoke({
        "messages": [
            HumanMessage(
                content="Please check my order status."
            )
        ]
    })

    print("\nAgent Error Recovery Result:")

    for message in result["messages"]:
        print("\nType:", message.type)
        print("Content:", message.content)

    final_message = result["messages"][-1].content

    assert "try again later" in final_message.lower()

    print("\nAgent error recovery test passed!")


if __name__ == "__main__":
    test_agent_error_recovery()
