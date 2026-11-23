
from langchain_core.messages import AIMessage


from src.agent.agent_router import route_agent


def test_multiple_tool_routing():

    # Simulate an AI response requesting two tools
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "track_customer_order",
                "args": {"order_id": "ORD1001"},
                "id": "call_001",
                "type": "tool_call"
            },
            {
                "name": "check_product_stock",
                "args": {"sku": "UC-ELE-916693"},
                "id": "call_002",
                "type": "tool_call"
            }
        ]
    )

    state = {
        "messages": [ai_message]
    }

    result = route_agent(state)

    print("Multiple tool calls detected:", len(ai_message.tool_calls))
    print("Router decision:", result)

    assert result == "tools"

    print("Multiple tool routing test passed!")


if __name__ == "__main__":
    test_multiple_tool_routing()
