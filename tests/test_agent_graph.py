
from src.agent.agent_graph import build_agent_graph


def test_agent_graph_compilation():

    agent_graph = build_agent_graph()

    print("\nAgent Graph Nodes:")

    graph_nodes = agent_graph.get_graph().nodes

    for node_name in graph_nodes:
        print(node_name)

    expected_nodes = {
        "__start__",
        "__end__",
        "agent",
        "tools"
    }

    actual_nodes = set(graph_nodes.keys())

    assert expected_nodes.issubset(actual_nodes)

    print("\nAgent graph compiled successfully.")


if __name__ == "__main__":
    test_agent_graph_compilation()
