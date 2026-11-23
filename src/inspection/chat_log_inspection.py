import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

file_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "urbancart_chat_query_log_raw.csv"
)

df = pd.read_csv(file_path)

print("\n--- Chat Log Dataset Shape ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Sample Records ---")
print(df.head(10).to_string(index=False))

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Records ---")
print(df.duplicated().sum())
print("\n--- Intent Distribution ---")

intent_counts = df["true_intent"].value_counts()

intent_summary = pd.DataFrame({
    "query_count": intent_counts,
    "percentage": (
        intent_counts / len(df) * 100
    ).round(2)
})

print(intent_summary)
print("\n--- Language Distribution ---")

language_counts = df["language_mix"].value_counts()

language_summary = pd.DataFrame({
    "query_count": language_counts,
    "percentage": (
        language_counts / len(df) * 100
    ).round(2)
})

print(language_summary)
print("\n--- Channel Distribution ---")

channel_counts = df["channel"].value_counts()

channel_summary = pd.DataFrame({
    "query_count": channel_counts,
    "percentage": (
        channel_counts / len(df) * 100
    ).round(2)
})

print(channel_summary)
print("\n--- Existing Bot Resolution ---")

resolution_counts = (
    df["resolved_by_bot_currently"]
    .value_counts()
)

resolution_summary = pd.DataFrame({
    "query_count": resolution_counts,
    "percentage": (
        resolution_counts / len(df) * 100
    ).round(2)
})

print(resolution_summary)
print("\n--- Intent vs Bot Resolution ---")

intent_resolution = pd.crosstab(
    df["true_intent"],
    df["resolved_by_bot_currently"],
    normalize="index"
) * 100

print(intent_resolution.round(2))
print("\n--- Language vs Bot Resolution ---")

language_resolution = pd.crosstab(
    df["language_mix"],
    df["resolved_by_bot_currently"],
    normalize="index"
) * 100

print(language_resolution.round(2))
print("\n--- Query Length Analysis ---")

df["query_length"] = df["query_text"].str.len()

print(df["query_length"].describe())
print("\n--- Sample Customer Queries ---")

for query in df["query_text"].head(20):
    print("-", query)
    print("\n--- Sample Queries by Intent ---")

for intent in df["true_intent"].unique():

    print(f"\nIntent: {intent}")

    sample_queries = df[
        df["true_intent"] == intent
    ]["query_text"].head(5)

    for query in sample_queries:
        print("-", query)
print("\n--- Duplicate Query Analysis ---")

query_counts = df["query_text"].value_counts()

duplicate_queries = query_counts[
    query_counts > 1
]

print("Unique repeated queries:", len(duplicate_queries))

print("\nTop 10 repeated queries:")
print(duplicate_queries.head(10))