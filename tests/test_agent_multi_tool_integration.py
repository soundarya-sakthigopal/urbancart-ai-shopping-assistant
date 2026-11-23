
from langchain_core.messages import HumanMessage

from src.agent.agent_graph import build_agent_graph


def test_agent_multi_tool_integration():

    # Build the actual UrbanCart agent graph
    graph = build_agent_graph()

    # Ask a question that requires two business tools
    user_query = (
        "Please check order ORD1001 and also check "
        "whether product SKU UC-ELE-916693 is in stock. "
        "Give me both results in one response."
    )

    config = {
    "configurable": {
        "thread_id": "customer_001"
    },
    "recursion_limit": 12
}

    result = graph.invoke(
        {"messages": [HumanMessage(content=user_query)]},
        config=config
    )

    print("\nComplete Agent Multi-Tool Conversation:")

    for message in result["messages"]:

        print(f"\nMessage Type: {message.type}")
        print("Content:", message.content)

        if getattr(message, "tool_calls", []):
            print("Tool Calls:", message.tool_calls)

    # Verify both actual tools were called
    tool_names = [
        message.name
        for message in result["messages"]
        if message.type == "tool"
    ]

    print("\nExecuted Tools:", tool_names)

    assert "track_customer_order" in tool_names
    assert "check_product_stock" in tool_names

    # Verify the agent produced a final response
    final_message = result["messages"][-1]

    assert final_message.type == "ai"
    assert not getattr(final_message, "tool_calls", [])

    print("\nComplete agent multi-tool integration test passed!")


if __name__ == "__main__":
    test_agent_multi_tool_integration()
