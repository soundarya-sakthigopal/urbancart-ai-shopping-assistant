
from langchain_core.tools import tool

from src.tools.inventory_tool import check_stock_status
from src.tools.order_tool import get_order_status
from src.tools.recommendation_tool import recommend_similar_products
from src.tools.escalation_tool import escalate_to_support
from src.rag.rag_chain import get_rag_chain


@tool
def search_catalog_and_policies(question: str) -> dict:
    """Answer product catalogue and company policy questions using RAG."""

    rag_chain = get_rag_chain()
    result = rag_chain.invoke(question)

    return result["final_response"]


@tool
def check_product_stock(sku: str) -> dict:
    """Check the current stock quantity and availability of a product using its SKU."""

    return check_stock_status(sku)


@tool
def track_customer_order(order_id: str) -> dict:
    """Retrieve order status and expected delivery date using an order ID."""

    return get_order_status(order_id)


@tool
def find_similar_products(sku: str, top_n: int = 3) -> dict:
    """Recommend similar in-stock products based on a given product SKU."""

    return recommend_similar_products(sku, top_n)


@tool
def create_support_escalation(
    customer_query: str,
    reason: str,
    conversation_summary: str
) -> dict:
    """Escalate a customer's issue to human support with a reason and conversation summary."""

    return escalate_to_support(
        customer_query,
        reason,
        conversation_summary
    )


URBANCART_TOOLS = [
    search_catalog_and_policies,
    check_product_stock,
    track_customer_order,
    find_similar_products,
    create_support_escalation
]
