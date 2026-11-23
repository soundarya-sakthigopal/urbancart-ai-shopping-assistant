import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbancart_product_catalog_cleaned.csv"
)
def load_processed_catalogue():

    df = pd.read_csv(PROCESSED_FILE)

    print("Catalogue loaded successfully.")
    print("Total records:", len(df))

    return df
def build_product_documents(df):

    documents = []

    for _, row in df.iterrows():

        if not row["product_eligible"]:
            continue

        content = f"""
        Product Name: {row['product_name']}
        SKU: {row['sku']}
        Category: {row['category_cleaned']}
        Brand: {row['brand'] if pd.notna(row['brand']) else 'Not available'}
        Description: {row['description'] if pd.notna(row['description']) else 'Not available'}
        """

        if row["price_eligible"]:
            content += f"Price: ₹{row['price_numeric']}\n"

        if row["stock_eligible"]:
            content += f"Stock Quantity: {row['stock_quantity']}\n"

        documents.append({
            "page_content": content.strip(),
            "metadata": {
                "sku": row["sku"],
                "category": row["category_cleaned"],
                "brand": row["brand"] if pd.notna(row["brand"]) else "Unknown",
                "source": "product_catalogue"
            }
        })

    print("Product documents created:", len(documents))

    return documents