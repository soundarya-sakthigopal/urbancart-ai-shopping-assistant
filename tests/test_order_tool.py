from src.tools.order_tool import (
    load_orders,
    get_order_status
)
def test_valid_order_id():

    df = load_orders()

    order_id = df.iloc[0]["order_id"]

    result = get_order_status(order_id)

    print("\nValid Order ID Test:")
    print(result)

    assert result["status"] == "success"
    assert result["order_id"] == order_id
    assert result["order_status"] == "Shipped"

def test_invalid_order_id():

    result = get_order_status("ORD9999")

    print("\nInvalid Order ID Test:")
    print(result)

    assert result["status"] == "not_found"
def test_empty_order_id():

    result = get_order_status("")

    print("\nEmpty Order ID Test:")
    print(result)

    assert result["status"] == "error"
if __name__ == "__main__":
    test_valid_order_id()
    test_invalid_order_id()
    test_empty_order_id()

    print("\nAll order tool tests passed.")