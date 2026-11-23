
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode


@tool
def simulate_tool_failure():
    """Simulate a technical tool failure for testing."""
    raise RuntimeError("Simulated database connection failure")


def test_tool_error_handling():

    # Create a small graph for testing tool errors
    workflow = StateGraph(MessagesState)

    tool_node = ToolNode(
        [simulate_tool_failure],
        handle_tool_errors=True
    )

    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)

    test_graph = workflow.compile()

    # Simulate an AI tool call
    result = test_graph.invoke({
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simulate_tool_failure",
                        "args": {},
                        "id": "test_error_001",
                        "type": "tool_call"
                    }
                ]
            )
        ]
    })

    print("\nTool Error Handling Result:")

    tool_message = result["messages"][-1]

    print("Message:", tool_message.content)
    print("Status:", tool_message.status)

    assert tool_message.status == "error"

    print("\nTool error handling test passed!")


if __name__ == "__main__":
    test_tool_error_handling()
