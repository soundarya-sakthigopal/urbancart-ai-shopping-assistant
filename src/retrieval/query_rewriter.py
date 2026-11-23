
import re

PRODUCT_POLICY_MAPPING = {
    "apparel": [
        "kurti", "shirt", "shirts", "jeans",
        "dress", "trousers", "footwear",
        "shoes", "sandals"
    ],
    "electronics": [
        "smartwatch", "watch", "mobile",
        "phone", "laptop", "earbuds",
        "headphones", "tablet"
    ],
    "home_kitchen": [
        "cookware", "furniture", "mixer",
        "kitchen", "utensils"
    ],
    "beauty_personal_care": [
        "makeup", "skincare", "cosmetics",
        "lipstick", "moisturizer"
    ]
}
POLICY_SEARCH_TERMS = {
    "apparel": "apparel footwear",
    "electronics": "electronics",
    "home_kitchen": "home kitchen",
    "beauty_personal_care": "beauty personal care"
}
POLICY_INTENT_MAPPING = {
    "return_refund": [
        "return", "refund", "exchange",
        "replace", "returnable", "non-returnable"
    ],
    "warranty": [
        "warranty", "guarantee", "manufacturing defect",
        "repair", "warranty claim"
    ],
    "shipping_delivery": [
    "delivery", "shipping", "delayed",
    "late", "tracking", "not delivered",
    "order not received", "undelivered",
    "undelivered order", "not received",
    "hasn't arrived", "has not arrived",
    "order not delivered"
    ],
    "payment_cod": [
        "cod", "cash on delivery",
        "payment", "debited", "duplicate payment",
        "failed payment", "payment failed"
    ]
}
POLICY_SOURCE_MAPPING = {
    "return_refund": "return_refund_policy.txt",
    "warranty": "warranty_policy.txt",
    "shipping_delivery": "shipping_delivery_policy.txt",
    "payment_cod": "payment_cod_policy.txt"
}

def rewrite_query(query: str) -> str:
    normalized_query = query.lower().strip()

    expanded_terms = []

    for policy_category, product_terms in PRODUCT_POLICY_MAPPING.items():
        for term in product_terms:
            pattern = rf"\b{re.escape(term)}\b"

            if re.search(pattern, normalized_query):
                expanded_terms.append(POLICY_SEARCH_TERMS[policy_category])
                break

    if expanded_terms:
        normalized_query += " " + " ".join(expanded_terms)

    return normalized_query
def detect_policy_intent(query: str):
    normalized_query = query.lower().strip()

    detected_intents = []

    for policy_type, keywords in POLICY_INTENT_MAPPING.items():
        for keyword in keywords:
            pattern = rf"\b{re.escape(keyword)}\b"

            if re.search(pattern, normalized_query):
                detected_intents.append(policy_type)
                break

    return detected_intents
def get_policy_sources(query: str) -> list[str]:
    intents = detect_policy_intent(query)

    return [
        POLICY_SOURCE_MAPPING[intent]
        for intent in intents
    ]