import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

text = input("Enter a sentence: ")

words = word_tokenize(text)
stemmer = PorterStemmer()

print("\nMorphological Analysis:")
for word in words:
    print(word, "->", stemmer.stem(word))