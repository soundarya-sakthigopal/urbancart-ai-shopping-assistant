
import pandas as pd
from pathlib import Path

from src.preprocessing.price_cleaner import clean_price


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "urbancart_product_catalog_raw.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_catalogue():

    """Load the original product catalogue."""

    df = pd.read_csv(RAW_FILE)

    return df
def prepare_catalogue():

    """Create a separate working copy for preprocessing."""

    df = load_catalogue()

    cleaned_df = df.copy()

    return cleaned_df

def clean_catalogue_prices(cleaned_df):

    """Convert raw product prices into numeric values."""

    cleaned_df["price_numeric"] = (
        cleaned_df["price"].apply(clean_price)
    )

    return cleaned_df
def validate_catalogue_prices(cleaned_df):

    """Identify invalid or suspicious product prices."""

    cleaned_df["price_status"] = "valid"

    # Missing prices
    cleaned_df.loc[
        cleaned_df["price_numeric"].isna(),
        "price_status"
    ] = "missing"

    # Price greater than MRP
    cleaned_df.loc[
        cleaned_df["price_numeric"] > cleaned_df["mrp"],
        "price_status"
    ] = "price_exceeds_mrp"

    return cleaned_df
def validate_catalogue_discounts(cleaned_df):

    """Identify invalid discount percentages."""

    cleaned_df["discount_status"] = "valid"

    invalid_mask = (
        (cleaned_df["discount_pct"] < 0) |
        (cleaned_df["discount_pct"] > 100)
    )

    cleaned_df.loc[
        invalid_mask,
        "discount_status"
    ] = "invalid"

    return cleaned_df
def validate_discount_consistency(cleaned_df):

    """Check whether listed discounts match actual prices."""

    cleaned_df["calculated_discount_pct"] = (
        (cleaned_df["mrp"] - cleaned_df["price_numeric"])
        / cleaned_df["mrp"]
    ) * 100

    cleaned_df["discount_consistency_status"] = "valid"

    # Compare only when price is available
    price_available = cleaned_df["price_numeric"].notna()

    mismatch_mask = (
        price_available &
        (
            (
                cleaned_df["calculated_discount_pct"]
                - cleaned_df["discount_pct"]
            ).abs() > 1
        )
    )

    cleaned_df.loc[
        mismatch_mask,
        "discount_consistency_status"
    ] = "mismatch"

    cleaned_df.loc[
        ~price_available,
        "discount_consistency_status"
    ] = "not_verifiable"

    return cleaned_df
def standardize_categories(cleaned_df):

    """Standardise category names."""

    category_mapping = {
        "electronics": "Electronics",
        "electronic": "Electronics",
        "footwear": "Footwear",
        "shoes": "Footwear",
        "home&kitchen": "Home & Kitchen",
        "home & kitchen": "Home & Kitchen",
        "home and kitchen": "Home & Kitchen",
        "fashion": "Fashion",
        "beauty": "Beauty"
    }

    cleaned_df["category_cleaned"] = (
        cleaned_df["category"]
        .str.strip()
        .str.lower()
        .replace(category_mapping)
    )

    return cleaned_df
def infer_missing_categories(cleaned_df):

    """Infer missing categories using SKU prefixes."""

    sku_category_mapping = {
        "BEA": "Beauty",
        "ELE": "Electronics",
        "FAS": "Fashion",
        "FOO": "Footwear",
        "HOM": "Home & Kitchen"
    }

    # Preserve the category before inference
    cleaned_df["category_original"] = (
        cleaned_df["category_cleaned"]
    )

    # Extract SKU prefix
    cleaned_df["sku_prefix"] = (
        cleaned_df["sku"].str.split("-").str[1]
    )

    # Identify missing categories
    missing_mask = cleaned_df["category_cleaned"].isna()

    # Infer category from SKU prefix
    cleaned_df.loc[missing_mask, "category_cleaned"] = (
        cleaned_df.loc[missing_mask, "sku_prefix"]
        .map(sku_category_mapping)
    )

    # Track where category came from
    cleaned_df["category_source"] = "Original"

    cleaned_df.loc[
        missing_mask &
        cleaned_df["category_cleaned"].notna(),
        "category_source"
    ] = "Inferred from SKU"

    cleaned_df.loc[
        cleaned_df["category_cleaned"].isna(),
        "category_source"
    ] = "Missing"

    return cleaned_df
def validate_product_completeness(cleaned_df):

    """Identify missing brand and description information."""

    cleaned_df["brand_status"] = "available"

    cleaned_df.loc[
        cleaned_df["brand"].isna() |
        cleaned_df["brand"].str.strip().eq(""),
        "brand_status"
    ] = "missing"

    cleaned_df["description_status"] = "available"

    cleaned_df.loc[
        cleaned_df["description"].isna() |
        cleaned_df["description"].str.strip().eq(""),
        "description_status"
    ] = "missing"

    return cleaned_df
def validate_stock_and_rating(cleaned_df):

    """Validate product stock quantities and ratings."""

    # Stock validation
    cleaned_df["stock_status"] = "valid"

    cleaned_df.loc[
        cleaned_df["stock_quantity"] < 0,
        "stock_status"
    ] = "invalid"

    cleaned_df.loc[
        cleaned_df["stock_quantity"] == 0,
        "stock_status"
    ] = "out_of_stock"

    # Rating validation
    cleaned_df["rating_status"] = "valid"

    cleaned_df.loc[
        (cleaned_df["rating"] < 0) |
        (cleaned_df["rating"] > 5),
        "rating_status"
    ] = "invalid"

    return cleaned_df
def validate_return_window(cleaned_df):

    """Validate product return-window values."""

    cleaned_df["return_window_status"] = "valid"

    # Missing return window
    cleaned_df.loc[
        cleaned_df["return_window_days"].isna(),
        "return_window_status"
    ] = "missing"

    # Negative return window
    cleaned_df.loc[
        cleaned_df["return_window_days"] < 0,
        "return_window_status"
    ] = "invalid_placeholder"

    return cleaned_df
def validate_duplicate_skus(cleaned_df):

    """Identify duplicate SKU records for manual review."""

    sku_counts = cleaned_df.groupby("sku").size()

    duplicate_skus = sku_counts[
        sku_counts > 1
    ].index

    cleaned_df["duplicate_status"] = "unique"

    cleaned_df.loc[
        cleaned_df["sku"].isin(duplicate_skus),
        "duplicate_status"
    ] = "duplicate_review"

    return cleaned_df
def validate_product_identity(cleaned_df):

    """Identify SKUs associated with different product identities."""

    identity_conflict_skus = (
        cleaned_df.groupby("sku")["product_name"]
        .nunique()
    )

    identity_conflict_skus = identity_conflict_skus[
        identity_conflict_skus > 1
    ].index

    cleaned_df["identity_status"] = "valid"

    cleaned_df.loc[
        cleaned_df["sku"].isin(identity_conflict_skus),
        "identity_status"
    ] = "identity_conflict"

    return cleaned_df

def validate_product_dates(cleaned_df):

    """Validate product date-added values."""

    cleaned_df["date_added_parsed"] = pd.to_datetime(
        cleaned_df["date_added"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    cleaned_df["date_status"] = "valid"

    cleaned_df.loc[
        cleaned_df["date_added_parsed"].isna(),
        "date_status"
    ] = "invalid"

    return cleaned_df
def run_catalogue_pipeline():

    """Execute the complete catalogue preprocessing pipeline."""

    cleaned_df = prepare_catalogue()

    # Price cleaning and validation
    cleaned_df = clean_catalogue_prices(cleaned_df)
    cleaned_df = validate_catalogue_prices(cleaned_df)

    # Discount validation
    cleaned_df = validate_catalogue_discounts(cleaned_df)
    cleaned_df = validate_discount_consistency(cleaned_df)

    # Category standardisation
    cleaned_df = standardize_categories(cleaned_df)
    cleaned_df = infer_missing_categories(cleaned_df)

    # Product completeness
    cleaned_df = validate_product_completeness(cleaned_df)

    # Stock and rating validation
    cleaned_df = validate_stock_and_rating(cleaned_df)

    # Return window validation
    cleaned_df = validate_return_window(cleaned_df)

    # Duplicate and identity validation
    cleaned_df = validate_duplicate_skus(cleaned_df)
    cleaned_df = validate_product_identity(cleaned_df)

    # Date validation
    cleaned_df = validate_product_dates(cleaned_df)
    cleaned_df = add_product_eligibility(cleaned_df)

    return cleaned_df
def save_processed_catalogue(cleaned_df):

    """Save the cleaned catalogue to the processed directory."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DIR / "urbancart_product_catalog_cleaned.csv"

    cleaned_df.to_csv(output_file, index=False)

    print(f"Cleaned catalogue saved: {output_file}")

    return output_file
def generate_review_report(cleaned_df):

    """Generate a report of records requiring manual review."""

    review_mask = (
        (cleaned_df["price_status"] != "valid") |
        (cleaned_df["discount_status"] != "valid") |
        (cleaned_df["discount_consistency_status"] == "mismatch") |
        (cleaned_df["stock_status"] == "invalid") |
        (cleaned_df["rating_status"] == "invalid") |
        (cleaned_df["return_window_status"] != "valid") |
        (cleaned_df["duplicate_status"] == "duplicate_review") |
        (cleaned_df["identity_status"] == "identity_conflict") |
        (cleaned_df["brand_status"] == "missing") |
        (cleaned_df["description_status"] == "missing") |
        (cleaned_df["date_status"] == "invalid")
    )

    review_df = cleaned_df[review_mask].copy()

    output_file = (
        PROCESSED_DIR / "urbancart_data_quality_review.csv"
    )

    review_df.to_csv(output_file, index=False)

    print("Records requiring review:", len(review_df))
    print("Review report saved:", output_file)

    return review_df
def add_product_eligibility(cleaned_df):

    # Check whether duplicate SKUs have conflicting prices
    price_conflicts = (
        cleaned_df.groupby("sku")["price_numeric"]
        .transform(lambda x: x.nunique(dropna=False) > 1)
    )

    # Check whether duplicate SKUs have conflicting stock values
    stock_conflicts = (
        cleaned_df.groupby("sku")["stock_quantity"]
        .transform(lambda x: x.nunique(dropna=False) > 1)
    )

    # Product retrieval eligibility
    cleaned_df["product_eligible"] = (
        (cleaned_df["identity_status"] == "valid") &
        cleaned_df["product_name"].notna()
    )

    # Price answer eligibility
    cleaned_df["price_eligible"] = (
        (cleaned_df["price_status"] == "valid") &
        (~price_conflicts)
    )

    # Stock answer eligibility
    cleaned_df["stock_eligible"] = (
        cleaned_df["stock_status"].isin(["valid", "out_of_stock"]) &
        (~stock_conflicts)
    )

    # Discount answer eligibility
    cleaned_df["discount_eligible"] = (
        (cleaned_df["discount_status"] == "valid") &
        (cleaned_df["discount_consistency_status"] == "valid") &
        (~price_conflicts)
    )

    return cleaned_df