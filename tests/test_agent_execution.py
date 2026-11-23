
from src.agent.agent_graph import build_agent_graph
from src.rag.rag_initializer import initialize_rag_components
from src.agent.agent_tools import search_catalog_and_policies
from src.agent.agent_graph import build_agent_graph

def test_stock_tool_selection():

    agent_graph = build_agent_graph()

    result = agent_graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Your question here"
            }
        ]
    },
    config={
        "configurable": {
            "thread_id": "customer_001"
        }
    }
)

    print("\nAgent Conversation:")

    for message in result["messages"]:
        print("\nMessage Type:", message.type)
        print("Content:", message.content)

        if hasattr(message, "tool_calls") and message.tool_calls:
            print("Tool Calls:", message.tool_calls)



def test_rag_tool_selection():

    # Initialize RAG components before agent execution
    initialize_rag_components()

    agent_graph = build_agent_graph()

    result = agent_graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": "What is the return window for a smartwatch?"
            }
        ]
    })

    print("\nRAG Agent Conversation:")

    for message in result["messages"]:
        print("\nMessage Type:", message.type)
        print("Content:", message.content)

        if hasattr(message, "tool_calls") and message.tool_calls:
            print("Tool Calls:", message.tool_calls)

def test_order_tool_selection():

    agent_graph = build_agent_graph()

    result = agent_graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Can you check the status of my order ORD1001?"
            }
        ]
    })

    print("\nOrder Tracking Agent Conversation:")

    for message in result["messages"]:
        print("\nMessage Type:", message.type)
        print("Content:", message.content)

        if hasattr(message, "tool_calls") and message.tool_calls:
            print("Tool Calls:", message.tool_calls)

def test_recommendation_tool_selection():

    agent_graph = build_agent_graph()

    result = agent_graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "Can you recommend similar products "
                    "for SKU UC-ELE-891071?"
                )
            }
        ]
    })

    print("\nRecommendation Agent Conversation:")

    for message in result["messages"]:
        print("\nMessage Type:", message.type)
        print("Content:", message.content)

        if hasattr(message, "tool_calls") and message.tool_calls:
            print("Tool Calls:", message.tool_calls)


def test_escalation_tool_selection():

    agent_graph = build_agent_graph()

    result = agent_graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "I have contacted support multiple times "
                    "about my delayed order. Please connect "
                    "me with a human support agent."
                )
            }
        ]
    })

    print("\nEscalation Agent Conversation:")

    for message in result["messages"]:
        print("\nMessage Type:", message.type)
        print("Content:", message.content)

        if hasattr(message, "tool_calls") and message.tool_calls:
            print("Tool Calls:", message.tool_calls)

def test_conversation_memory():
    initialize_rag_components()
    agent_graph = build_agent_graph()
    

    # First customer message
    first_response = agent_graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the return window for a smartwatch?"
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": "memory_test_001"
            }
        }
    )

    print("\nFirst Response:")
    print(first_response["messages"][-1].content)

    # Follow-up message using the SAME thread ID
    second_response = agent_graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What if the original box is opened?"
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": "memory_test_001"
            }
        }
    )

    print("\nFollow-up Response:")
    print(second_response["messages"][-1].content)

    print("\nTotal messages in conversation:",
          len(second_response["messages"]))

if __name__ == "__main__":
    #test_stock_tool_selection()
    #test_rag_tool_selection()
    #test_order_tool_selection()
    #test_recommendation_tool_selection()
    #test_escalation_tool_selection()
    test_conversation_memory()
