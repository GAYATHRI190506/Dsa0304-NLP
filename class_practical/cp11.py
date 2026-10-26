import nltk
from nltk import CFG

grammar = CFG.fromstring("""
S -> NP VP
NP -> 'Ram' | 'Ravi'
VP -> 'runs' | 'plays'
""")

sentence = input("Enter sentence: ").split()
parser = nltk.RecursiveDescentParser(grammar)

found = False
for tree in parser.parse(sentence):
    tree.pretty_print()
    found = True

if not found:
    print("Invalid sentence")
