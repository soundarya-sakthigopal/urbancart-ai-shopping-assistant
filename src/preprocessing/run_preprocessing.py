from src.preprocessing.catalog_cleaner import (
    run_catalogue_pipeline,
    save_processed_catalogue,
    generate_review_report
)


def main():

    print("Starting catalogue preprocessing...")

    # Run cleaning and validation
    cleaned_df = run_catalogue_pipeline()

    # Save cleaned catalogue
    save_processed_catalogue(cleaned_df)

    # Generate data quality report
    review_df = generate_review_report(cleaned_df)

    print("\nPreprocessing completed successfully.")
    print("Total products:", len(cleaned_df))
    print("Records requiring review:", len(review_df))


if __name__ == "__main__":
    main()