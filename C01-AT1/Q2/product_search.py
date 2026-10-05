import re


# Read products from external file
with open("products.txt", "r") as file:
    products = [line.strip() for line in file if line.strip()]


# Display matching products
def display_results(search_type, keyword, matches):

    print("\n========================================")
    print("Search Type :", search_type)
    print("Keyword     :", keyword)
    print("========================================")

    if matches:
        print("Matching Products:")

        for product in matches:
            print("-", product)
    else:
        print("No matching products found.")

    print("\nTotal Matching Products:", len(matches))


# Exact keyword search
def exact_search(keyword):

    pattern = r"\b" + re.escape(keyword) + r"\b"

    return [
        product
        for product in products
        if re.search(pattern, product, re.IGNORECASE)
    ]


# Prefix search
def prefix_search(keyword):

    pattern = r"\b" + re.escape(keyword) + r"\w*"

    return [
        product
        for product in products
        if re.search(pattern, product, re.IGNORECASE)
    ]


# Suffix search
def suffix_search(keyword):

    pattern = r"\w*" + re.escape(keyword) + r"\b"

    return [
        product
        for product in products
        if re.search(pattern, product, re.IGNORECASE)
    ]


# Partial keyword search
def partial_search(keyword):

    pattern = re.escape(keyword)

    return [
        product
        for product in products
        if re.search(pattern, product, re.IGNORECASE)
    ]


# -------------------------------
# Test the search system
# -------------------------------

# 1. Exact keyword search
result = exact_search("Laptop")
display_results("Exact Keyword Search", "Laptop", result)


# 2. Prefix search
result = prefix_search("Apple")
display_results("Prefix Search", "Apple", result)


# 3. Suffix search
result = suffix_search("Pro")
display_results("Suffix Search", "Pro", result)


# 4. Partial keyword search
result = partial_search("Galaxy")
display_results("Partial Keyword Search", "Galaxy", result)


# 5. Case-insensitive search
result = exact_search("APPLE")
display_results("Case-Insensitive Search", "APPLE", result)