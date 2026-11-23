from src.tools.recommendation_tool import (
    load_product_catalogue,
    recommend_similar_products
)
def test_valid_sku():

    df = load_product_catalogue()

    sku = df[
        (df["product_eligible"] == True)
        & (df["price_eligible"] == True)
        & (df["stock_eligible"] == True)
        & (df["stock_quantity"] > 0)
    ].iloc[0]["sku"]

    result = recommend_similar_products(sku)

    print("\nValid SKU Recommendation Test:")
    print(result)

    assert result["status"] == "success"
    assert "recommendations" in result

    for recommendation in result["recommendations"]:
        assert recommendation["sku"] != sku
        assert recommendation["stock_quantity"] > 0
def test_invalid_sku():

    result = recommend_similar_products("UC-INVALID-000000")

    print("\nInvalid SKU Recommendation Test:")
    print(result)

    assert result["status"] == "not_found"
    assert "recommendations" not in result
def test_empty_sku():

    result = recommend_similar_products("")

    print("\nEmpty SKU Recommendation Test:")
    print(result)

    assert result["status"] == "error"
def test_custom_top_n():

    df = load_product_catalogue()

    sku = df[
        (df["product_eligible"] == True)
        & (df["price_eligible"] == True)
        & (df["stock_eligible"] == True)
        & (df["stock_quantity"] > 0)
    ].iloc[0]["sku"]

    result = recommend_similar_products(sku, top_n=2)

    print("\nCustom Top N Test:")
    print(result)

    assert result["status"] == "success"
    assert len(result["recommendations"]) <= 2
if __name__ == "__main__":
    test_valid_sku()
    test_invalid_sku()
    test_empty_sku()
    test_custom_top_n()

    print("\nAll recommendation tool tests passed.")