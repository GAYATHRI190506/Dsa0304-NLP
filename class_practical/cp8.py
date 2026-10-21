
# Word probabilities for POS tags
tag_prob = {
    "book": {"NN": 0.8, "VB": 0.2},
    "play": {"NN": 0.4, "VB": 0.6},
    "runs": {"NN": 0.2, "VB": 0.8},
    "student": {"NN": 0.9, "VB": 0.1}
}

text = input("Enter a sentence: ")
words = text.lower().split()

for word in words:
    if word in tag_prob:
        tag = max(tag_prob[word], key=tag_prob[word].get)
    else:
        tag = "NN"

    print(word, "->", tag)
