
from src.agent.agent_tools import URBANCART_TOOLS


def test_agent_tool_registration():

    print("\nRegistered UrbanCart Agent Tools:")

    for agent_tool in URBANCART_TOOLS:
        print("\nTool Name:", agent_tool.name)
        print("Description:", agent_tool.description)
        print("Input Schema:", agent_tool.args)

    expected_tools = {
        "search_catalog_and_policies",
        "check_product_stock",
        "track_customer_order",
        "find_similar_products",
        "create_support_escalation"
    }

    actual_tools = {
        agent_tool.name
        for agent_tool in URBANCART_TOOLS
    }

    assert actual_tools == expected_tools
    assert len(URBANCART_TOOLS) == 5

    print("\nAll agent tools registered successfully.")


if __name__ == "__main__":
    test_agent_tool_registration()
