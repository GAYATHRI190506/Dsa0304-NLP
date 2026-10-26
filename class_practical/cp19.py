import nltk
from nltk.wsd import lesk

nltk.download('wordnet')

sentence = input("Enter sentence: ")
word = input("Enter target word: ")

sense = lesk(sentence.split(), word)

if sense:
    print("Word:", word)
    print("Meaning:", sense.definition())
else:
    print("Meaning not found")
