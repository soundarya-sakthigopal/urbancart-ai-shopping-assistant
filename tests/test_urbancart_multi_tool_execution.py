
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from src.agent.agent_tools import URBANCART_TOOLS


def test_urbancart_multi_tool_execution():

    # Build a graph using actual UrbanCart tools
    workflow = StateGraph(MessagesState)

    workflow.add_node(
        "tools",
        ToolNode(URBANCART_TOOLS, handle_tool_errors=True)
    )

    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)

    graph = workflow.compile()

    # Simulate an LLM requesting two actual UrbanCart tools
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "track_customer_order",
                "args": {"order_id": "ORD1001"},
                "id": "urbancart_call_001",
                "type": "tool_call"
            },
            {
                "name": "check_product_stock",
                "args": {"sku": "UC-ELE-916693"},
                "id": "urbancart_call_002",
                "type": "tool_call"
            }
        ]
    )

    result = graph.invoke({
        "messages": [ai_message]
    })

    # Collect actual tool responses
    tool_messages = [
        message
        for message in result["messages"]
        if message.type == "tool"
    ]

    print("\nUrbanCart Multi-Tool Results:")

    for message in tool_messages:
        print(f"\nTool: {message.name}")
        print("Result:", message.content)

    # Verify both tools executed
    assert len(tool_messages) == 2

    results_by_name = {
        message.name: message.content
        for message in tool_messages
    }

    assert "Shipped" in results_by_name["track_customer_order"]
    assert "283" in results_by_name["check_product_stock"]

    print("\nUrbanCart multi-tool execution test passed!")


if __name__ == "__main__":
    test_urbancart_multi_tool_execution()
