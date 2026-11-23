import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbancart_product_catalog_cleaned.csv"
)
def load_inventory_catalogue():
    "Load clean data for stock checking"
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Product catalogue not found: {PROCESSED_FILE}") 
    df = pd.read_csv(PROCESSED_FILE)
    required_colums = {"sku","stock_quantity"}
    missing_columns = required_colums-set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return df
def check_stock_status(sku: str, pincode: str = None) -> dict:
    """
    Check product stock availability using SKU.
    """

    if not sku or not sku.strip():
        return {
            "status": "error",
            "message": "SKU is required."
        }

    df = load_inventory_catalogue()

    product = df[
        df["sku"].astype(str).str.strip().str.upper()
        == sku.strip().upper()
    ]

    if product.empty:
        return {
            "status": "not_found",
            "message": f"No product found for SKU: {sku}"
        }

    product = product.iloc[0]

    stock_quantity = product["stock_quantity"]

    if pd.isna(stock_quantity):
        return {
            "status": "unknown",
            "sku": sku,
            "message": "Stock information is unavailable."
        }

    stock_quantity = int(stock_quantity)

    if stock_quantity > 0:
        return {
            "status": "in_stock",
            "sku": sku,
            "product_name": product["product_name"],
            "stock_quantity": stock_quantity,
            "message": "Product is available."
        }

    return {
        "status": "out_of_stock",
        "sku": sku,
        "product_name": product["product_name"],
        "stock_quantity": 0,
        "message": "Product is currently out of stock."
    }