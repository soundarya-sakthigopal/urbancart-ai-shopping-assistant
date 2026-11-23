import pandas as pd
from pathlib import Path
from src.preprocessing.price_cleaner import clean_price

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Load the raw product catalogue
file_path = PROJECT_ROOT / "data" / "raw" / "urbancart_product_catalog_raw.csv"

df = pd.read_csv(file_path)

# 1. Dataset size
print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# 2. Column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# 3. Data types
print("\n--- Data Types ---")
print(df.dtypes)

# 4. First five records
print("\n--- Sample Records ---")
print(df.head())

# 5. Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 6. Duplicate records
print("\n--- Duplicate Records ---")
print(df.duplicated().sum())
# 7. Duplicate SKU check
print("\n--- Duplicate SKU Check ---")
print("Duplicate SKUs:", df["sku"].duplicated().sum())

# 8. Category distribution
print("\n--- Category Distribution ---")
print(df["category"].value_counts(dropna=False))

# 9. Sample price values
print("\n--- Sample Price Values ---")
print(df["price"].dropna().head(20).tolist())

# 10. Unique product categories
print("\n--- Unique Categories ---")
print(df["category"].dropna().unique())
# 11. Inspect duplicate SKU records
print("\n--- Duplicate SKU Records ---")

duplicate_skus = df[
    df["sku"].duplicated(keep=False)
]

print(
    duplicate_skus[
        ["sku", "product_name", "price",
         "mrp", "stock_quantity", "date_added"]
    ].sort_values("sku").head(20)
)

# 12. Analyse duplicate SKU groups

duplicate_groups = df[
    df["sku"].duplicated(keep=False)
].groupby("sku")

print("\n--- Duplicate SKU Analysis ---")

print("Number of duplicate SKU groups:", len(duplicate_groups))

print(
    "SKUs with different prices:",
    duplicate_groups["price"].nunique().gt(1).sum()
)

print(
    "SKUs with different stock quantities:",
    duplicate_groups["stock_quantity"].nunique().gt(1).sum()
)

print(
    "SKUs with different product names:",
    duplicate_groups["product_name"].nunique().gt(1).sum()
)
# 13. Identify SKUs with different product names

name_conflicts = (
    duplicate_groups["product_name"]
    .nunique()
)

conflicting_skus = name_conflicts[
    name_conflicts > 1
].index

print("\n--- Product Name Conflicts ---")

print(
    df[df["sku"].isin(conflicting_skus)][
        ["sku", "product_name", "category", "brand"]
    ].sort_values("sku")
)
# 14. Price format inspection

print("\n--- Price Data Inspection ---")

print("Total records:", len(df))
print("Missing prices:", df["price"].isnull().sum())
print("Unique price formats:", df["price"].nunique())

print("\nSample of unique price values:")

print(
    df["price"]
    .dropna()
    .drop_duplicates()
    .head(50)
    .tolist()
)

# Create a copy so raw data remains unchanged
cleaned_df = df.copy()

# Convert price into numeric values
cleaned_df["price_numeric"] = cleaned_df["price"].apply(clean_price)

print("\n--- Price Cleaning Results ---")

print("Original missing prices:", df["price"].isnull().sum())

print("Cleaned missing prices:", cleaned_df["price_numeric"].isnull().sum())

print("Cleaned price data type:", cleaned_df["price_numeric"].dtype)

print("\nSample comparison:")

print(cleaned_df[["price", "price_numeric"]].head(20))

# Check failed price conversions

failed_prices = cleaned_df[
    cleaned_df["price"].notna() &
    cleaned_df["price_numeric"].isna()
]

print("\n--- Failed Price Conversions ---")
print("Failed records:", len(failed_prices))

if len(failed_prices) > 0:
    print(failed_prices["price"].unique())
# Price validation against MRP

price_issues = cleaned_df[
    cleaned_df["price_numeric"] > cleaned_df["mrp"]
]

print("\n--- Price Greater Than MRP ---")
print("Number of records:", len(price_issues))

print(
    price_issues[
        ["sku", "product_name", "price", "price_numeric", "mrp"]
    ].head(20)
)
# Calculate expected discount percentage

cleaned_df["calculated_discount_pct"] = (
    (cleaned_df["mrp"] - cleaned_df["price_numeric"])
    / cleaned_df["mrp"]
) * 100

# Find differences greater than 1 percentage point
discount_issues = cleaned_df[
    cleaned_df["price_numeric"].notna() &
    (
        cleaned_df["calculated_discount_pct"]
        - cleaned_df["discount_pct"]
    ).abs().gt(1)
]

print("\n--- Discount Percentage Validation ---")
print("Records with discount mismatch:", len(discount_issues))

print(
    discount_issues[
        ["sku", "price_numeric", "mrp",
         "discount_pct", "calculated_discount_pct"]
    ].head(20)
)
# Validate existing discount percentage

invalid_discount = cleaned_df[
    (cleaned_df["discount_pct"] < 0) |
    (cleaned_df["discount_pct"] > 100)
]

print("\n--- Invalid Discount Values ---")

print("Invalid discount records:", len(invalid_discount))

print(
    invalid_discount[
        ["sku", "product_name", "price_numeric",
         "mrp", "discount_pct"]
    ].head(20)
)
# Category distribution analysis

print("\n--- Category Distribution Analysis ---")

category_counts = cleaned_df["category"].value_counts(
    dropna=False
)

category_summary = pd.DataFrame({
    "product_count": category_counts,
    "percentage": (
        category_counts / len(cleaned_df) * 100
    ).round(2)
})

print(category_summary)
# Identify case and whitespace variations

print("\n--- Normalised Category Names ---")

normalised_categories = (
    cleaned_df["category"]
    .dropna()
    .str.strip()
    .str.lower()
    .unique()
)

print(sorted(normalised_categories))
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

# Standardise category values
cleaned_df["category_cleaned"] = (
    cleaned_df["category"]
    .str.strip()
    .str.lower()
    .replace(category_mapping)
)

print("\n--- Standardised Category Distribution ---")
print(cleaned_df["category_cleaned"].value_counts(dropna=False))

print("\n--- Products with Missing Categories ---")

missing_category = cleaned_df[
    cleaned_df["category_cleaned"].isna()
]

print("Total missing categories:", len(missing_category))

print(
    missing_category[
        ["sku", "product_name", "brand", "description"]
    ].head(20).to_string(index=False)
)
print("\n--- Missing Category: SKU Prefix Analysis ---")

missing_category["sku_prefix"] = (
    missing_category["sku"].str.split("-").str[1]
)

print(
    missing_category["sku_prefix"].value_counts(dropna=False)
)

print("\n--- Prefix vs Product Name ---")

print(
    missing_category[
        ["sku", "sku_prefix", "product_name"]
    ].head(30).to_string(index=False)
)
print("\n--- SKU Prefix vs Existing Category ---")

cleaned_df["sku_prefix"] = (
    cleaned_df["sku"].str.split("-").str[1]
)

print(
    pd.crosstab(
        cleaned_df["sku_prefix"],
        cleaned_df["category_cleaned"],
        dropna=False
    )
)
# Preserve the original category values
cleaned_df["category_original"] = cleaned_df["category_cleaned"]

# Map SKU prefixes to categories
sku_category_mapping = {
    "BEA": "Beauty",
    "ELE": "Electronics",
    "FAS": "Fashion",
    "FOO": "Footwear",
    "HOM": "Home & Kitchen"
}

# Identify missing categories
missing_mask = cleaned_df["category_cleaned"].isna()

# Fill missing categories using SKU prefix
cleaned_df.loc[missing_mask, "category_cleaned"] = (
    cleaned_df.loc[missing_mask, "sku_prefix"]
    .map(sku_category_mapping)
)

# Mark inferred categories
cleaned_df["category_source"] = "Original"

cleaned_df.loc[missing_mask, "category_source"] = "Inferred from SKU"

print("\n--- Category Cleaning Summary ---")
print(cleaned_df["category_cleaned"].value_counts(dropna=False))

print("\n--- Category Source ---")
print(cleaned_df["category_source"].value_counts())

print("\nRemaining missing categories:",
      cleaned_df["category_cleaned"].isna().sum())
print("\n--- Missing Brand and Description Analysis ---")

print("Missing brands:", cleaned_df["brand"].isna().sum())

print("Missing descriptions:", cleaned_df["description"].isna().sum())

print("\n--- Sample Products with Missing Brands ---")

print(
    cleaned_df[
        cleaned_df["brand"].isna()
    ][["sku", "product_name", "category_cleaned"]]
    .head(15)
    .to_string(index=False)
)

print("\n--- Sample Products with Missing Descriptions ---")

print(
    cleaned_df[
        cleaned_df["description"].isna()
    ][["sku", "product_name", "brand", "category_cleaned"]]
    .head(15)
    .to_string(index=False)
)
print("\n--- Brand and Description Missing Combinations ---")

both_missing = (
    cleaned_df["brand"].isna() &
    cleaned_df["description"].isna()
)

brand_missing_only = (
    cleaned_df["brand"].isna() &
    cleaned_df["description"].notna()
)

description_missing_only = (
    cleaned_df["brand"].notna() &
    cleaned_df["description"].isna()
)

print("Both brand and description missing:", both_missing.sum())

print("Only brand missing:", brand_missing_only.sum())

print("Only description missing:", description_missing_only.sum())

print(
    "Both fields available:",
    (
        cleaned_df["brand"].notna() &
        cleaned_df["description"].notna()
    ).sum()
)
print("\n--- Missing Return Window Analysis ---")

print("Missing return windows:",
      cleaned_df["return_window_days"].isna().sum())

print("\nSample records:")

print(
    cleaned_df[
        cleaned_df["return_window_days"].isna()
    ][
        ["sku", "product_name", "category_cleaned",
         "return_window_days"]
    ].head(15).to_string(index=False)
)
print("\n--- Return Window Distribution ---")

print(
    cleaned_df["return_window_days"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\n--- Return Window by Category ---")

print(
    cleaned_df.groupby(
        "category_cleaned",
        dropna=False
    )["return_window_days"]
    .agg(["count", "min", "max", "mean"])
    .round(2)
)
print("\n--- Return Window Value Distribution ---")

print(
    cleaned_df["return_window_days"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\n--- Negative Return Windows ---")

negative_returns = cleaned_df[
    cleaned_df["return_window_days"] < 0
]

print("Negative values:", len(negative_returns))

print(
    negative_returns[
        ["sku", "product_name", "category_cleaned",
         "return_window_days"]
    ].head(20).to_string(index=False)
)

print("\n--- Zero Return Windows ---")

print(
    "Zero values:",
    (cleaned_df["return_window_days"] == 0).sum()
)
print("\n--- Negative Return Windows by Category ---")

print(
    negative_returns["category_cleaned"]
    .value_counts(dropna=False)
)

print("\n--- Negative Return Windows by Product ---")

print(
    negative_returns["product_name"]
    .value_counts()
)
# Create a separate status column
cleaned_df["return_window_status"] = "valid"

# Identify missing return windows
cleaned_df.loc[
    cleaned_df["return_window_days"].isna(),
    "return_window_status"
] = "missing"

# Identify negative return windows
cleaned_df.loc[
    cleaned_df["return_window_days"] < 0,
    "return_window_status"
] = "invalid_placeholder"

print("\n--- Return Window Quality Summary ---")

print(cleaned_df["return_window_status"].value_counts())
print("\n--- Stock Quantity Validation ---")

print("Minimum stock:", cleaned_df["stock_quantity"].min())
print("Maximum stock:", cleaned_df["stock_quantity"].max())

print("Negative stock records:",
      (cleaned_df["stock_quantity"] < 0).sum())

print("Zero stock records:",
      (cleaned_df["stock_quantity"] == 0).sum())

print("\n--- Rating Validation ---")

print("Minimum rating:", cleaned_df["rating"].min())
print("Maximum rating:", cleaned_df["rating"].max())

print("Missing ratings:", cleaned_df["rating"].isna().sum())

print("Ratings below 0:",
      (cleaned_df["rating"] < 0).sum())

print("Ratings above 5:",
      (cleaned_df["rating"] > 5).sum())

print("\n--- Rating Distribution ---")

print(cleaned_df["rating"].describe())
print("\n--- Negative Stock Records ---")

negative_stock = cleaned_df[
    cleaned_df["stock_quantity"] < 0
]

print(
    negative_stock[
        ["sku", "product_name", "stock_quantity"]
    ].head(20).to_string(index=False)
)

print("\n--- Invalid Rating Records ---")

invalid_ratings = cleaned_df[
    cleaned_df["rating"] > 5
]

print(
    invalid_ratings[
        ["sku", "product_name", "rating"]
    ].head(20).to_string(index=False)
)
# Inspect negative stock
negative_stock = cleaned_df[
    cleaned_df["stock_quantity"] < 0
]

print("\n--- Negative Stock Samples ---")
print(
    negative_stock[
        ["sku", "product_name", "stock_quantity"]
    ].head(20).to_string(index=False)
)

# Inspect ratings above 5
invalid_ratings = cleaned_df[
    cleaned_df["rating"] > 5
]

print("\n--- Invalid Rating Samples ---")
print(
    invalid_ratings[
        ["sku", "product_name", "rating"]
    ].head(20).to_string(index=False)
)

# Stock quantity validation
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


# Display validation summary
print("\n--- Stock Quality Summary ---")
print(cleaned_df["stock_status"].value_counts())

print("\n--- Rating Quality Summary ---")
print(cleaned_df["rating_status"].value_counts())

print("\n--- MRP Validation ---")

print("Minimum MRP:", cleaned_df["mrp"].min())
print("Maximum MRP:", cleaned_df["mrp"].max())

print("Zero or negative MRP:",
      (cleaned_df["mrp"] <= 0).sum())

print("\n--- Discount Percentage Validation ---")

print("Minimum discount:", cleaned_df["discount_pct"].min())
print("Maximum discount:", cleaned_df["discount_pct"].max())

print("Negative discounts:",
      (cleaned_df["discount_pct"] < 0).sum())

print("Discounts above 100:",
      (cleaned_df["discount_pct"] > 100).sum())
cleaned_df["discount_status"] = "valid"

cleaned_df.loc[
    (cleaned_df["discount_pct"] < 0) |
    (cleaned_df["discount_pct"] > 100),
    "discount_status"
] = "invalid"

print("\n--- Discount Quality Summary ---")

print(cleaned_df["discount_status"].value_counts())
print("\n--- Date Added Inspection ---")

print("Data type:", cleaned_df["date_added"].dtype)

print("\nSample dates:")
print(cleaned_df["date_added"].head(20).tolist())

print("\nUnique date formats / values:")
print(cleaned_df["date_added"].dropna().unique()[:20])

print("\nMissing dates:")
print(cleaned_df["date_added"].isna().sum())
cleaned_df["date_added_parsed"] = pd.to_datetime(
    cleaned_df["date_added"],
    format="%Y-%m-%d",
    errors="coerce"
)

print("\n--- Date Validation Summary ---")

print(
    "Invalid dates:",
    cleaned_df["date_added_parsed"].isna().sum()
)

print(
    "Earliest date:",
    cleaned_df["date_added_parsed"].min()
)

print(
    "Latest date:",
    cleaned_df["date_added_parsed"].max()
)

print("\n--- Catalogue Quality Summary ---")

print("Total records:", len(cleaned_df))

print("\nMissing values:")
print(cleaned_df.isnull().sum())

print("\nDuplicate SKU records:")
print(cleaned_df["sku"].duplicated().sum())

print("\nValidation status summaries:")

for column in [
    "return_window_status",
    "stock_status",
    "rating_status",
    "discount_status"
]:
    print(f"\n{column}:")
    print(cleaned_df[column].value_counts(dropna=False))
print("\n--- SKU Identity Conflicts ---")

duplicate_groups = cleaned_df[
    cleaned_df["sku"].duplicated(keep=False)
].groupby("sku")

identity_conflicts = duplicate_groups.filter(
    lambda group:
        group["product_name"].nunique() > 1
        or group["brand"].nunique(dropna=True) > 1
)

print("Records involved:", len(identity_conflicts))

print(
    identity_conflicts[
        ["sku", "product_name", "category_cleaned", "brand"]
    ].sort_values("sku").to_string(index=False)
)

# Identify SKUs associated with multiple product identities

identity_conflict_skus = (
    cleaned_df.groupby("sku")["product_name"]
    .nunique()
)

identity_conflict_skus = identity_conflict_skus[
    identity_conflict_skus > 1
].index

# Create status column
cleaned_df["identity_status"] = "valid"

cleaned_df.loc[
    cleaned_df["sku"].isin(identity_conflict_skus),
    "identity_status"
] = "identity_conflict"

print("\n--- Identity Status Summary ---")
print(cleaned_df["identity_status"].value_counts())

print("\n--- Records Requiring Identity Review ---")
print(
    cleaned_df[
        cleaned_df["identity_status"] == "identity_conflict"
    ][["sku", "product_name", "category_cleaned", "brand"]]
)

print("\n--- Duplicate SKU Attribute Conflicts ---")

duplicate_records = cleaned_df[
    cleaned_df["sku"].duplicated(keep=False)
]

same_identity_duplicates = duplicate_records[
    duplicate_records["identity_status"] == "valid"
]

duplicate_groups = same_identity_duplicates.groupby("sku")

print(
    "Duplicate SKU groups with same product identity:",
    duplicate_groups.ngroups
)

print(
    "Groups with different prices:",
    duplicate_groups["price_numeric"].nunique(dropna=False).gt(1).sum()
)

print(
    "Groups with different stock quantities:",
    duplicate_groups["stock_quantity"].nunique().gt(1).sum()
)

print(
    "Groups with different brands:",
    duplicate_groups["brand"].nunique(dropna=False).gt(1).sum()
)
# Count distinct records for each SKU
sku_counts = cleaned_df.groupby("sku").size()

# Identify duplicated SKUs
duplicate_skus = sku_counts[sku_counts > 1].index

# Assign duplicate status
cleaned_df["duplicate_status"] = "unique"

cleaned_df.loc[
    cleaned_df["sku"].isin(duplicate_skus),
    "duplicate_status"
] = "duplicate_review"

print("\n--- Duplicate Status Summary ---")
print(cleaned_df["duplicate_status"].value_counts())
print("\n--- Duplicate SKU Frequency ---")

print(
    cleaned_df["sku"]
    .value_counts()
    .value_counts()
    .sort_index()
)