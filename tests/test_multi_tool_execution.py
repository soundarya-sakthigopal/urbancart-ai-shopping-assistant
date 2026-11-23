
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


@tool
def check_order(order_id: str) -> str:
    """Check a customer's order status."""
    return f"Order {order_id} is Shipped."


@tool
def check_stock(sku: str) -> str:
    """Check product stock."""
    return f"Product {sku} has 283 units in stock."


def test_multi_tool_execution():

    tools = [check_order, check_stock]

    workflow = StateGraph(MessagesState)

    workflow.add_node(
        "tools",
        ToolNode(tools, handle_tool_errors=True)
    )

    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)

    graph = workflow.compile()

    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_order",
                "args": {"order_id": "ORD1001"},
                "id": "call_001",
                "type": "tool_call"
            },
            {
                "name": "check_stock",
                "args": {"sku": "UC-ELE-916693"},
                "id": "call_002",
                "type": "tool_call"
            }
        ]
    )

    result = graph.invoke({
        "messages": [ai_message]
    })

    tool_messages = [
        message
        for message in result["messages"]
        if message.type == "tool"
    ]

    print("\nMultiple Tool Execution Results:")

    for message in tool_messages:
        print(message.content)

    assert len(tool_messages) == 2
    assert "ORD1001" in tool_messages[0].content
    assert "UC-ELE-916693" in tool_messages[1].content

    print("\nMultiple tool execution test passed!")


if __name__ == "__main__":
    test_multi_tool_execution()
