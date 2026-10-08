word = input("Enter a noun: ")

if word.endswith(("s", "x", "z", "ch", "sh")):
    plural = word + "es"
elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
    plural = word[:-1] + "ies"
else:
    plural = word + "s"

print("Plural form:", plural)