import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORDER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mock_orders.csv"
)
def load_orders():
    """
    Load mock order records for order tracking.
    """

    if not ORDER_FILE.exists():
        raise FileNotFoundError(
            f"Order database not found: {ORDER_FILE}"
        )

    df = pd.read_csv(ORDER_FILE)

    required_columns = {
        "order_id",
        "status",
        "expected_delivery_date",
        "last_updated"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df
def get_order_status(order_id: str) -> dict:
    """
    Retrieve order status using the order ID.
    """

    if not order_id or not order_id.strip():
        return {
            "status": "error",
            "message": "Order ID is required."
        }

    df = load_orders()

    order = df[
        df["order_id"].astype(str).str.strip().str.upper()
        == order_id.strip().upper()
    ]

    if order.empty:
        return {
            "status": "not_found",
            "message": f"No order found for ID: {order_id}"
        }

    order = order.iloc[0]

    return {
        "status": "success",
        "order_id": order["order_id"],
        "order_status": order["status"],
        "expected_delivery_date": (
            None
            if pd.isna(order["expected_delivery_date"])
            else str(order["expected_delivery_date"])
        ),
        "last_updated": str(order["last_updated"]),
        "message": "Order details retrieved successfully."
    }