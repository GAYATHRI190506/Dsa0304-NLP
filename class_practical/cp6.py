
import random

text = "I love natural language processing"
words = text.split()

bigrams = list(zip(words, words[1:]))

print("Bigrams:", bigrams)

current = "I"
generated = [current]

for i in range(4):
    next_words = [b[1] for b in bigrams if b[0] == current]

    if not next_words:
        break

    current = random.choice(next_words)
    generated.append(current)

print("Generated Text:", " ".join(generated))
