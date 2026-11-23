from src.tools.inventory_tool import (
    load_inventory_catalogue,
    check_stock_status
)


def test_valid_sku():
    df = load_inventory_catalogue()

    sku = df.iloc[0]["sku"]

    result = check_stock_status(sku)

    print("\nValid SKU Test:")
    print(result)

    assert result["status"] in [
        "in_stock",
        "out_of_stock",
        "unknown"
    ]


def test_invalid_sku():

    result = check_stock_status("UC-INVALID-000000")

    print("\nInvalid SKU Test:")
    print(result)

    assert result["status"] == "not_found"


def test_empty_sku():

    result = check_stock_status("")

    print("\nEmpty SKU Test:")
    print(result)

    assert result["status"] == "error"


def test_invalid_sku_for_agent():
    from src.tools.inventory_tool import check_stock_status

    result = check_stock_status("INVALID-SKU-123")

    print("\nInvalid SKU Result:")
    print(result)

    assert result["status"] == "not_found"

if __name__ == "__main__":
    test_valid_sku()
    test_invalid_sku()
    test_empty_sku()
    test_invalid_sku_for_agent()

    print("\nAll inventory tool tests passed.")