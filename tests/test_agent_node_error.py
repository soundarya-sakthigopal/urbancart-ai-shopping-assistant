
from unittest.mock import patch

from src.agent.agent_node import agent_node


def test_agent_node_error():

    with patch(
        "src.agent.agent_node.get_chat_model",
        side_effect=RuntimeError("Simulated OpenAI API failure")
    ):

        result = agent_node({
            "messages": [
                {
                    "role": "user",
                    "content": "What is my order status?"
                }
            ]
        })

    print("\nAgent Node Error Test:")
    print("Error:", result["error"])
    print("Customer Response:", result["messages"][0].content)

    assert "Simulated OpenAI API failure" in result["error"]

    assert (
        "try again later"
        in result["messages"][0].content.lower()
    )

    print("\nAgent node error handling test passed!")


if __name__ == "__main__":
    test_agent_node_error()
