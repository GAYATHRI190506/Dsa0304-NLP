
words = ["The", "students", "are", "playing"]

# Initial tags
tags = ["DT", "NN", "NN", "NN"]

print("Initial Tags:")
print(list(zip(words, tags)))

# Transformation rules
for i in range(len(words)):
    # Rule 1: Words ending in 's' are plural nouns
    if words[i].lower().endswith("s"):
        tags[i] = "NNS"

    # Rule 2: 'are' is a verb
    if words[i].lower() == "are":
        tags[i] = "VBP"

    # Rule 3: Words ending in 'ing' are present participles
    if words[i].lower().endswith("ing"):
        tags[i] = "VBG"

print("Final Tags:")
print(list(zip(words, tags)))
