from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

n = int(input("Enter number of documents: "))
documents = []

for i in range(n):
    doc = input("Enter document: ")
    documents.append(doc)

query = input("Enter search query: ")

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents + [query])

scores = cosine_similarity(
    tfidf[-1], tfidf[:-1]
).flatten()

for i in scores.argsort()[::-1]:
    print("Document:", documents[i])
    print("Score:", round(scores[i], 3))
