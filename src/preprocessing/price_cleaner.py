import pandas as pd
import re
def clean_price(value):
    #Hande missing values
    if pd.isna(value):
        return None
    #Convert to string and remove spaces
    value =str(value).strip().lower()
    # remove currency symbols and prefix
    value = value.replace("₹", "")
    value = value.replace("rs", "")
    value = value.replace("/-", "")
    value = value.replace(",", "")

    if value.endswith("k"):
        number = float(value[:-1])
        return number*1000
    try:
        return float(value)

    except ValueError:
        return None
