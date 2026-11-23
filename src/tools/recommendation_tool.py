import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbancart_product_catalog_cleaned.csv"
)
def load_product_catalogue():
    """
    Load the cleaned product catalogue for recommendations.
    """

    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Product catalogue not found: {PROCESSED_FILE}"
        )

    df = pd.read_csv(PROCESSED_FILE)

    required_columns = {
        "sku",
        "product_name",
        "category_cleaned",
        "price_numeric",
        "stock_quantity",
        "product_eligible",
        "price_eligible",
        "stock_eligible"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df

def recommend_similar_products(
    sku: str,
    top_n: int = 3
) -> dict:
    """
    Recommend similar products based on category,
    price range, and stock availability.
    """

    # 1. Validate input
    if not isinstance(sku, str) or not sku.strip():
        return {
            "status": "error",
            "message": "SKU is required."
        }

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        return {
            "status": "error",
            "message": "top_n must be a positive integer."
        }

    # 2. Load product catalogue
    df = load_product_catalogue()

    # 3. Find the original product
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

    # 4. Validate original product price
    if (
        not product["price_eligible"]
        or pd.isna(product["price_numeric"])
    ):
        return {
            "status": "error",
            "message": "Price information is unavailable for this product."
        }

    original_price = float(product["price_numeric"])
    category = product["category_cleaned"]

    # 5. Define price range (30% difference)
    min_price = original_price * 0.70
    max_price = original_price * 1.30

    # 6. Filter similar products
    candidates = df[
        (df["category_cleaned"] == category)
        & (
            df["sku"].astype(str).str.strip().str.upper()
            != sku.strip().upper()
        )
        & (df["product_eligible"] == True)
        & (df["price_eligible"] == True)
        & (df["stock_eligible"] == True)
        & (df["stock_quantity"] > 0)
        & (df["price_numeric"] >= min_price)
        & (df["price_numeric"] <= max_price)
    ].copy()

    # 7. Handle no recommendations
    if candidates.empty:
        return {
            "status": "success",
            "sku": sku,
            "product_name": product["product_name"],
            "recommendations": [],
            "message": "No similar products found."
        }

    # 8. Rank by closest price
    candidates["price_difference"] = (
        candidates["price_numeric"] - original_price
    ).abs()

    candidates = candidates.sort_values(
        by="price_difference",
        ascending=True
    )

    # 9. Select top N recommendations
    recommendations = candidates.head(top_n)

    # 10. Prepare structured response
    recommendation_list = []

    for _, row in recommendations.iterrows():
        recommendation_list.append({
            "sku": row["sku"],
            "product_name": row["product_name"],
            "category": row["category_cleaned"],
            "price": float(row["price_numeric"]),
            "stock_quantity": int(row["stock_quantity"])
        })

    return {
        "status": "success",
        "original_sku": sku,
        "original_product": product["product_name"],
        "recommendations": recommendation_list,
        "message": "Similar products retrieved successfully."
    }
