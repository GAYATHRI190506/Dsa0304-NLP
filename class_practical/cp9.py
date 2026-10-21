
import re

patterns = [
    (r".*ing$", "VBG"),
    (r".*ed$", "VBD"),
    (r".*ly$", "RB"),
    (r".*ous$", "JJ"),
    (r".*s$", "NNS"),
    (r".*", "NN")
]

text = input("Enter a sentence: ")
words = re.findall(r"\b\w+\b", text.lower())

for word in words:
    for pattern, tag in patterns:
        if re.fullmatch(pattern, word):
            print(word, "->", tag)
            break
