
from langchain_core.messages import AIMessage

from src.rag.llm_service import get_chat_model
from src.agent.agent_prompt import AGENT_PROMPT
from src.agent.agent_tools import URBANCART_TOOLS


def agent_node(state: dict) -> dict:
    """
    Process customer messages and decide
    whether a business tool should be called.
    """

    try:
        # 1. Get the LLM
        llm = get_chat_model()

        # 2. Bind available tools
        llm_with_tools = llm.bind_tools(URBANCART_TOOLS)

        # 3. Prepare conversation prompt
        prompt_value = AGENT_PROMPT.invoke({
            "messages": state.get("messages", [])
        })

        # 4. Invoke the LLM
        response = llm_with_tools.invoke(prompt_value)

        # 5. Return updated conversation state
        return {
            "messages": [response],
            "error": ""
        }

    except Exception as e:
        print(f"Agent node error: {e}")

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I'm sorry, but I'm unable to process "
                        "your request right now. Please try again later."
                    )
                )
            ],
            "error": str(e)
        }
