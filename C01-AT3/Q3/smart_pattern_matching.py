import re

# Get text from user
text = input("Enter the text: ")

while True:
    print("\n===== SMART PATTERN MATCHING ENGINE =====")
    print("1. Search Word")
    print("2. Search Date")
    print("3. Search Phone Number")
    print("4. Search Hashtag")
    print("5. Search Mention")
    print("6. Search Prefix")
    print("7. Search Suffix")
    print("8. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        word = input("Enter word to search: ")
        pattern = r"\b" + re.escape(word) + r"\b"
        result = re.findall(pattern, text, re.IGNORECASE)

        if result:
            print("Matching Words:", result)
        else:
            print("No matching word found.")

    elif choice == "2":
        pattern = r"\b\d{2}/\d{2}/\d{4}\b"
        result = re.findall(pattern, text)

        if result:
            print("Dates Found:", result)
        else:
            print("No date found.")

    elif choice == "3":
        pattern = r"\b[6-9]\d{9}\b"
        result = re.findall(pattern, text)

        if result:
            print("Phone Numbers Found:", result)
        else:
            print("No phone number found.")

    elif choice == "4":
        pattern = r"#\w+"
        result = re.findall(pattern, text)

        if result:
            print("Hashtags Found:", result)
        else:
            print("No hashtag found.")

    elif choice == "5":
        pattern = r"@\w+"
        result = re.findall(pattern, text)

        if result:
            print("Mentions Found:", result)
        else:
            print("No mention found.")

    elif choice == "6":
        prefix = input("Enter prefix: ")
        pattern = r"\b" + re.escape(prefix) + r"\w*"
        result = re.findall(pattern, text, re.IGNORECASE)

        if result:
            print("Prefix Matches:", result)
        else:
            print("No prefix match found.")

    elif choice == "7":
        suffix = input("Enter suffix: ")
        pattern = r"\b\w*" + re.escape(suffix) + r"\b"
        result = re.findall(pattern, text, re.IGNORECASE)

        if result:
            print("Suffix Matches:", result)
        else:
            print("No suffix match found.")

    elif choice == "8":
        print("Exiting program...")
        break

    else:
        print("Invalid choice. Please try again.")